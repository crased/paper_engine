"""
Configuration parser for Paper Engine
Reads .ini configuration files and provides easy access to settings
"""

import configparser
import os
from pathlib import Path


class ConfigParser:
    """Parser for .ini configuration files with type conversion support"""

    def __init__(self, config_file):
        """
        Initialize the configuration parser

        Args:
            config_file: Path to the .ini configuration file
        """
        self.config_file = config_file
        self.parser = configparser.ConfigParser()

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        self.parser.read(config_file)
        self._config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from ini file and convert types"""
        for section in self.parser.sections():
            for key, value in self.parser.items(section):
                # Convert key to uppercase to match original Python config format
                config_key = key.upper()
                self._config[config_key] = self._convert_value(value)

    def _convert_value(self, value):
        """
        Convert string values to appropriate Python types

        Args:
            value: String value from ini file

        Returns:
            Converted value (bool, int, float, list, dict, or str)
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Handle lists (comma-separated values)
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        # Try to convert to number
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get(self, key, default=None):
        """Get configuration value by key"""
        return self._config.get(key, default)

    def __getattr__(self, name):
        """Allow attribute-style access to configuration values"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._config.get(name.upper())

    def __getitem__(self, key):
        """Allow dictionary-style access to configuration values"""
        return self._config[key.upper()]

    def __contains__(self, key):
        """Check if key exists in configuration"""
        return key.upper() in self._config

    def keys(self):
        """Get all configuration keys"""
        return self._config.keys()

    def values(self):
        """Get all configuration values"""
        return self._config.values()

    def items(self):
        """Get all configuration key-value pairs"""
        return self._config.items()

    def to_dict(self):
        """Return configuration as dictionary"""
        return self._config.copy()


class MainConfig(ConfigParser):
    """Configuration for main.py"""

    def __init__(self):
        config_dir = Path(__file__).parent
        super().__init__(config_dir / "main_conf.ini")

    @property
    def LABEL_STUDIO_ENV(self):
        """Return Label Studio environment variables as dict"""
        enabled = self.get("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", True)
        return {
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true" if enabled else "false"
        }


class ScreencaptureConfig(ConfigParser):
    """Configuration for screencapture.py"""

    def __init__(self):
        config_dir = Path(__file__).parent
        super().__init__(config_dir / "screencapture_conf.ini")

    @property
    def LINUX_ENV_VARS(self):
        """Return Linux environment variables as dict"""
        return {
            "XDG_CURRENT_DESKTOP": self.get("XDG_CURRENT_DESKTOP", "sway"),
            "QT_QPA_PLATFORM": self.get("QT_QPA_PLATFORM", "wayland")
        }

    @property
    def FLAMESHOT_COMMAND(self):
        """Return flameshot command as list"""
        command = self.get("COMMAND", "flameshot")
        args = self.get("ARGS", "screen,-n,1,-r")
        if isinstance(args, str):
            args_list = [arg.strip() for arg in args.split(',')]
        else:
            args_list = args
        return [command] + args_list


class TrainingConfig(ConfigParser):
    """Configuration for training_model.py"""

    def __init__(self):
        config_dir = Path(__file__).parent
        super().__init__(config_dir / "training_conf.ini")

    @property
    def EXPORT_FORMATS(self):
        """Return export formats as list"""
        formats = self.get("EXPORT_FORMATS", "torchscript,onnx")
        if isinstance(formats, str):
            return [fmt.strip() for fmt in formats.split(',')]
        return formats


class PrepareDatasetConfig(ConfigParser):
    """Configuration for prepare_local_dataset.py"""

    def __init__(self):
        config_dir = Path(__file__).parent
        super().__init__(config_dir / "prepare_dataset_conf.ini")

    @property
    def LABEL_GROUPS(self):
        """
        Return label groups as OrderedDict
        Format: {group_name: [label1, label2, ...]}
        """
        from collections import OrderedDict

        if not self.parser.has_section('Groups'):
            return OrderedDict()

        groups = OrderedDict()
        for group_name, labels_str in self.parser.items('Groups'):
            if labels_str.strip():  # Only include non-empty groups
                labels = [label.strip() for label in labels_str.split(',') if label.strip()]
                if labels:
                    groups[group_name] = labels

        return groups

    @property
    def LABEL_MAPPING(self):
        """
        Return label mapping for renaming labels
        Format: {original_name: new_name}
        """
        if not self.parser.has_section('LabelMapping'):
            return {}

        mapping = {}
        for original, new_name in self.parser.items('LabelMapping'):
            if new_name.strip():
                mapping[original] = new_name.strip()

        return mapping

    def get_class_mapping(self, discovered_classes=None):
        """
        Generate class mapping based on configuration

        Args:
            discovered_classes: Set of classes found in annotations (optional)

        Returns:
            dict: {class_name: class_id}
        """
        # Apply label renaming if configured
        label_mapping = self.LABEL_MAPPING

        if self.ENABLE_CUSTOM_GROUPING:
            # Use custom groups
            label_groups = self.LABEL_GROUPS

            if not label_groups:
                print("WARNING: enable_custom_grouping=true but no groups defined")
                print("Falling back to auto-discovery")
                if discovered_classes:
                    return {cls: idx for idx, cls in enumerate(sorted(discovered_classes))}
                return {}

            # Create mapping with sequential IDs
            class_mapping = {}
            class_id = 0

            for group_name, labels in label_groups.items():
                for label in labels:
                    # Apply renaming if configured
                    final_label = label_mapping.get(label, label)
                    class_mapping[final_label] = class_id
                    class_id += 1

            return class_mapping
        else:
            # Auto-discover and sort alphabetically
            if discovered_classes:
                # Apply renaming
                renamed_classes = set()
                for cls in discovered_classes:
                    renamed_classes.add(label_mapping.get(cls, cls))

                return {cls: idx for idx, cls in enumerate(sorted(renamed_classes))}
            return {}


# Create singleton instances for easy import
main_conf = MainConfig()
screencapture_conf = ScreencaptureConfig()
training_conf = TrainingConfig()
prepare_dataset_conf = PrepareDatasetConfig()


# For backwards compatibility, expose all config values at module level
def _expose_config_values():
    """Expose config values as module-level variables for backwards compatibility"""
    import sys
    module = sys.modules[__name__]

    # Expose main config
    for key, value in main_conf.items():
        setattr(module, key, value)


# Uncomment below if you want module-level access to config values
# _expose_config_values()
