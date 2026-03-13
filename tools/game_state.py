"""
Generic game state reader for Unity/Mono games via external memory reading.

Provides a config-driven system to read arbitrary game state from any Unity
game running under Mono. The user defines "state fields" as pointer chains
starting from Mono class static/instance fields, and the reader automatically:
  - Resolves Mono classes and fields by name
  - Distinguishes static vs instance fields (from Mono metadata)
  - Follows pointer chains through objects, collections (List/Dict), arrays
  - Reads the final value as the correct type (int, float, bool, string)

Usage:
    from tools.game_state import GameStateReader, StateFieldDef, ChainStep

    config = [
        StateFieldDef("hp", "int", [
            ChainStep.static("PlayerManager", "players"),
            ChainStep.dict_value(0),
            ChainStep.field("_stats"),
            ChainStep.field("<Health>k__BackingField"),
        ]),
        StateFieldDef("scene", "string", [
            ChainStep.static("SceneLoader", "<SceneName>k__BackingField"),
        ]),
    ]

    reader = GameStateReader(mono_external, config, "Assembly-CSharp")
    reader.resolve_classes()
    state = reader.read_all()
    print(state)  # {"hp": 3, "scene": "scene_level_platforming_1_1F", ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from tools.mono_external import (
    MonoExternal,
    MonoClassInfo,
    MonoFieldInfo,
    MonoArrayOffsets,
    MonoTypeEnum,
    CSharpListOffsets,
    CSharpDictOffsets,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain step definitions — describe how to traverse a pointer chain
# ---------------------------------------------------------------------------


class StepKind(Enum):
    """Type of a chain traversal step."""

    STATIC = auto()  # Read a static field from a class
    FIELD = auto()  # Read an instance field from the current pointer
    LIST_ITEM = auto()  # Index into a C# List<T>
    DICT_VALUE = auto()  # Get a value from a C# Dictionary<K,V> by slot scan
    ARRAY_ITEM = auto()  # Index into a raw MonoArray


@dataclass
class ChainStep:
    """A single step in a pointer chain.

    Each step describes one dereference / traversal operation. Steps are
    executed left-to-right; the output pointer of one step becomes the
    context for the next.

    Use the class methods (static, field, list_item, dict_value, array_item)
    to construct steps — don't create instances directly.
    """

    kind: StepKind
    class_name: str = ""  # For STATIC: which class to read from
    field_name: str = ""  # For STATIC/FIELD: the field name
    index: int = 0  # For LIST_ITEM/ARRAY_ITEM: element index
    max_scan: int = 4  # For DICT_VALUE: max slots to scan for non-null
    owner_class: str = ""  # For FIELD: explicit class that owns this field
    #                        (needed when context is lost, e.g. after collection access)

    @classmethod
    def static(cls, class_name: str, field_name: str) -> ChainStep:
        """Read a static field from a named class.

        This is always the first step in a chain. If the field is a value type
        (int, float, bool), the chain should end here. If it's a reference type,
        the chain continues by following the pointer.
        """
        return cls(kind=StepKind.STATIC, class_name=class_name, field_name=field_name)

    @classmethod
    def field(cls, field_name: str, owner_class: str = "") -> ChainStep:
        """Read an instance field from the current object pointer.

        Args:
            field_name: The field name to read.
            owner_class: Optional class name that defines this field. Use this
                         when the class context is lost (e.g. after a
                         dict_value or list_item step). The class will be
                         auto-resolved during resolve_classes().
        """
        return cls(kind=StepKind.FIELD, field_name=field_name, owner_class=owner_class)

    @classmethod
    def list_item(cls, index: int = 0) -> ChainStep:
        """Index into a C# List<T>._items array.

        Reads _items (MonoArray*), then reads element at the given index.
        """
        return cls(kind=StepKind.LIST_ITEM, index=index)

    @classmethod
    def dict_value(cls, index: int = 0, max_scan: int = 4) -> ChainStep:
        """Get a value from a C# Dictionary<K,V>.

        Reads valueSlots (MonoArray*), then scans up to max_scan slots
        starting at `index` for the first non-null entry.
        """
        return cls(kind=StepKind.DICT_VALUE, index=index, max_scan=max_scan)

    @classmethod
    def array_item(cls, index: int = 0) -> ChainStep:
        """Index into a raw MonoArray (T[]).

        Reads element at index from a MonoArray's data region.
        """
        return cls(kind=StepKind.ARRAY_ITEM, index=index)

    def __repr__(self) -> str:
        if self.kind == StepKind.STATIC:
            return f"Static({self.class_name}.{self.field_name})"
        elif self.kind == StepKind.FIELD:
            prefix = f"{self.owner_class}." if self.owner_class else "."
            return f"Field({prefix}{self.field_name})"
        elif self.kind == StepKind.LIST_ITEM:
            return f"List[{self.index}]"
        elif self.kind == StepKind.DICT_VALUE:
            return f"DictValue(start={self.index}, scan={self.max_scan})"
        elif self.kind == StepKind.ARRAY_ITEM:
            return f"Array[{self.index}]"
        return f"Step({self.kind})"


# ---------------------------------------------------------------------------
# State field definition — one entry in the game state config
# ---------------------------------------------------------------------------


# Supported read types
READ_TYPES = {"int", "float", "bool", "string", "ptr"}


@dataclass
class StateFieldDef:
    """Definition of a single game state value to read.

    Args:
        name: Key in the output state dictionary.
        read_type: How to interpret the final value ("int", "float", "bool",
                   "string", "ptr").
        chain: Sequence of ChainSteps describing how to reach the value.
        default: Default value if the chain fails (None = use type default).
        description: Human-readable description (optional, for docs/debug).
    """

    name: str
    read_type: str
    chain: list[ChainStep]
    default: Any = None
    description: str = ""

    def __post_init__(self):
        if self.read_type not in READ_TYPES:
            raise ValueError(
                f"Invalid read_type '{self.read_type}' for '{self.name}'. "
                f"Must be one of: {READ_TYPES}"
            )
        if not self.chain:
            raise ValueError(f"Empty chain for state field '{self.name}'")
        if self.default is None:
            self.default = _type_default(self.read_type)

    def __repr__(self) -> str:
        chain_str = " -> ".join(str(s) for s in self.chain)
        return f"StateField({self.name}: {self.read_type} = {chain_str})"


def _type_default(read_type: str) -> Any:
    """Return the default value for a read type."""
    return {
        "int": 0,
        "float": 0.0,
        "bool": False,
        "string": "",
        "ptr": 0,
    }[read_type]


# ---------------------------------------------------------------------------
# GameStateReader — the main generic reader
# ---------------------------------------------------------------------------


class GameStateReader:
    """Config-driven game state reader for Unity/Mono games.

    Takes a MonoExternal instance and a list of StateFieldDefs, resolves
    all referenced Mono classes, and provides read_all() to produce a
    dict of game state values.

    Args:
        mono: Attached MonoExternal instance.
        fields: List of state field definitions.
        assembly: Assembly name to look up classes in (e.g. "Assembly-CSharp").
    """

    def __init__(
        self,
        mono: MonoExternal,
        fields: list[StateFieldDef],
        assembly: str = "Assembly-CSharp",
    ):
        self.mono = mono
        self.pm = mono.pm
        self.fields = fields
        self.assembly = assembly

        # Resolved class cache: class_name -> MonoClassInfo
        self._classes: dict[str, Optional[MonoClassInfo]] = {}

        # Track which classes are needed
        self._needed_classes: set[tuple[str, str]] = set()  # (namespace, name)
        self._extract_needed_classes()

    def _extract_needed_classes(self):
        """Scan all field definitions to find which Mono classes we need."""
        for fdef in self.fields:
            for step in fdef.chain:
                class_name = ""
                if step.kind == StepKind.STATIC:
                    class_name = step.class_name
                elif step.kind == StepKind.FIELD and step.owner_class:
                    class_name = step.owner_class

                if class_name:
                    if "." in class_name:
                        ns, cn = class_name.rsplit(".", 1)
                    else:
                        ns, cn = "", class_name
                    self._needed_classes.add((ns, cn))

    def resolve_classes(self) -> dict[str, Optional[MonoClassInfo]]:
        """Resolve all Mono classes referenced by the field definitions.

        Returns a dict of class_name -> MonoClassInfo (or None if not found).
        """
        if not self._needed_classes:
            return {}

        results = self.mono.find_classes_batch(
            self.assembly, list(self._needed_classes)
        )
        self._classes = results

        found = [k for k, v in results.items() if v is not None]
        missing = [k for k, v in results.items() if v is None]
        if found:
            logger.info("Resolved classes: %s", found)
        if missing:
            logger.warning("Missing classes: %s", missing)

        return results

    def get_class(self, name: str) -> Optional[MonoClassInfo]:
        """Get a resolved class by name (with or without namespace)."""
        # Try exact match first
        if name in self._classes:
            return self._classes[name]
        # Try without namespace
        for key, klass in self._classes.items():
            if key.endswith(f".{name}") or key == name:
                return klass
        return None

    # -- Chain execution --

    def _execute_chain(self, fdef: StateFieldDef) -> Any:
        """Execute a pointer chain and read the final value.

        Returns the value, or the field's default if any step fails.
        """
        ptr = 0  # Current pointer (context for next step)
        current_class: Optional[MonoClassInfo] = None  # Class context for FIELD steps

        for i, step in enumerate(fdef.chain):
            is_last = i == len(fdef.chain) - 1

            if step.kind == StepKind.STATIC:
                klass = self.get_class(step.class_name)
                if klass is None:
                    logger.debug(
                        "%s: class '%s' not resolved", fdef.name, step.class_name
                    )
                    return fdef.default

                current_class = klass
                field_info = klass.get_field(step.field_name)
                if field_info is None:
                    logger.debug(
                        "%s: field '%s' not found in %s",
                        fdef.name,
                        step.field_name,
                        step.class_name,
                    )
                    return fdef.default

                if field_info.is_static:
                    # Read from static data block
                    if is_last:
                        return self._read_static_leaf(klass, field_info, fdef.read_type)
                    else:
                        ptr = self._read_static_ptr(klass, field_info)
                        if not ptr:
                            logger.debug(
                                "%s: static field '%s.%s' is null",
                                fdef.name,
                                step.class_name,
                                step.field_name,
                            )
                            return fdef.default
                else:
                    # Instance field on a static-less context — shouldn't happen
                    # as STATIC step implies reading from the class's static data.
                    # But if the field is instance, we need an object pointer first.
                    logger.warning(
                        "%s: field '%s' is instance, not static (in STATIC step)",
                        fdef.name,
                        step.field_name,
                    )
                    return fdef.default

            elif step.kind == StepKind.FIELD:
                if ptr == 0:
                    logger.debug("%s: null pointer before FIELD step", fdef.name)
                    return fdef.default

                # Find the field — prefer explicit owner_class, then current
                # class context, then search all resolved classes.
                field_info = None

                if step.owner_class:
                    owner = self.get_class(step.owner_class)
                    if owner is not None:
                        field_info = owner.get_field(step.field_name)
                        current_class = owner

                if field_info is None and current_class is not None:
                    field_info = current_class.get_field(step.field_name)

                if field_info is None:
                    # Fallback: search all resolved classes
                    field_info = self._find_field_any_class(step.field_name)

                if field_info is None:
                    logger.debug(
                        "%s: field '%s' not found in any resolved class",
                        fdef.name,
                        step.field_name,
                    )
                    return fdef.default

                if is_last:
                    return self._read_instance_leaf(ptr, field_info, fdef.read_type)
                else:
                    # Follow the pointer
                    next_ptr = self.pm.read_pointer(ptr + field_info.offset)
                    if not next_ptr:
                        logger.debug(
                            "%s: instance field '%s' at 0x%x+0x%x is null",
                            fdef.name,
                            step.field_name,
                            ptr,
                            field_info.offset,
                        )
                        return fdef.default
                    ptr = next_ptr

            elif step.kind == StepKind.LIST_ITEM:
                if ptr == 0:
                    return fdef.default
                ptr = self._read_list_element(ptr, step.index)
                if not ptr:
                    logger.debug("%s: List element [%d] is null", fdef.name, step.index)
                    return fdef.default
                current_class = None  # Lost class context after collection access

            elif step.kind == StepKind.DICT_VALUE:
                if ptr == 0:
                    return fdef.default
                ptr = self._read_dict_first_value(ptr, step.index, step.max_scan)
                if not ptr:
                    logger.debug(
                        "%s: Dict has no non-null value (start=%d, scan=%d)",
                        fdef.name,
                        step.index,
                        step.max_scan,
                    )
                    return fdef.default
                current_class = None  # Lost class context after collection access

            elif step.kind == StepKind.ARRAY_ITEM:
                if ptr == 0 or step.index < 0:
                    return fdef.default
                element = self.pm.read_pointer(
                    ptr + MonoArrayOffsets.DATA + step.index * 8
                )
                if not element:
                    logger.debug("%s: Array[%d] is null", fdef.name, step.index)
                    return fdef.default
                ptr = element
                current_class = None

        # If we somehow get here without returning, chain was all pointer-following
        # and the last step wasn't a leaf read. Return the final pointer.
        return ptr if ptr else fdef.default

    def _find_field_any_class(self, field_name: str) -> Optional[MonoFieldInfo]:
        """Search all resolved classes for a field by name."""
        for klass in self._classes.values():
            if klass is None:
                continue
            f = klass.get_field(field_name)
            if f is not None:
                return f
        return None

    # -- Leaf value readers --

    def _read_static_leaf(
        self, klass: MonoClassInfo, field_info: MonoFieldInfo, read_type: str
    ) -> Any:
        """Read a static field as a final value (leaf of the chain)."""
        if not self.mono._ensure_static_data(klass):
            logger.debug("No static data for %s", klass.name)
            return _type_default(read_type)

        addr = klass.static_data_addr + field_info.offset
        return self._read_value_at(addr, field_info, read_type)

    def _read_static_ptr(self, klass: MonoClassInfo, field_info: MonoFieldInfo) -> int:
        """Read a static field as a pointer (intermediate chain step)."""
        if not self.mono._ensure_static_data(klass):
            return 0

        addr = klass.static_data_addr + field_info.offset
        size = MonoTypeEnum.size(field_info.type_code)
        if size <= 4:
            # Value type in a pointer context — read as pointer anyway
            # (e.g., could be a small struct or just a non-pointer field
            #  being followed; but in a chain, a non-last static step
            #  must yield a pointer)
            val = self.pm.read_pointer(addr)
            return val or 0
        return self.pm.read_pointer(addr) or 0

    def _read_instance_leaf(
        self, obj_addr: int, field_info: MonoFieldInfo, read_type: str
    ) -> Any:
        """Read an instance field as a final value (leaf of the chain)."""
        addr = obj_addr + field_info.offset
        return self._read_value_at(addr, field_info, read_type)

    def _read_value_at(
        self, addr: int, field_info: MonoFieldInfo, read_type: str
    ) -> Any:
        """Read a value at a memory address based on the desired read_type."""
        if read_type == "int":
            val = self.pm.read_int32(addr)
            return val if val is not None else 0
        elif read_type == "float":
            val = self.pm.read_float(addr)
            return val if val is not None else 0.0
        elif read_type == "bool":
            val = self.pm.read_uint8(addr)
            return bool(val) if val is not None else False
        elif read_type == "string":
            # String is a pointer to a MonoString object
            str_ptr = self.pm.read_pointer(addr)
            if not str_ptr:
                return ""
            return self.mono._read_mono_string(str_ptr) or ""
        elif read_type == "ptr":
            val = self.pm.read_pointer(addr)
            return val if val is not None else 0
        return _type_default(read_type)

    # -- Collection readers --

    def _read_list_element(self, list_ptr: int, index: int) -> int:
        """Read an element from a C# List<T>.

        Returns the element pointer, or 0 if not found.
        """
        if index < 0:
            return 0

        # List<T>._items is a MonoArray* at +0x10
        items_arr = self.pm.read_pointer(list_ptr + CSharpListOffsets.ITEMS)
        if not items_arr:
            return 0

        # Check _size
        size = self.pm.read_int32(list_ptr + CSharpListOffsets.SIZE)
        if size is None or index >= size:
            return 0

        # Read element from MonoArray data region
        element = self.pm.read_pointer(items_arr + MonoArrayOffsets.DATA + index * 8)
        return element or 0

    def _read_dict_first_value(
        self, dict_ptr: int, start_index: int, max_scan: int
    ) -> int:
        """Find the first non-null value in a C# Dictionary<K,V>.

        Scans valueSlots from start_index up to start_index + max_scan.
        Returns the value pointer, or 0 if none found.
        """
        if start_index < 0 or max_scan <= 0:
            return 0

        # Check count first
        count = self.pm.read_int32(dict_ptr + CSharpDictOffsets.COUNT)
        if count is None or count < 1:
            return 0

        # Read valueSlots array pointer
        value_slots = self.pm.read_pointer(dict_ptr + CSharpDictOffsets.VALUE_SLOTS)
        if not value_slots:
            return 0

        # Read touchedSlots to know how far to scan
        touched = self.pm.read_int32(dict_ptr + CSharpDictOffsets.TOUCHED_SLOTS)
        if touched is None or touched < 1:
            return 0

        # Scan for first non-null value
        scan_end = min(start_index + max_scan, touched)
        for i in range(start_index, scan_end):
            val = self.pm.read_pointer(value_slots + MonoArrayOffsets.DATA + i * 8)
            if val:
                return val

        return 0

    # -- Main API --

    def read_all(self) -> dict[str, Any]:
        """Read all configured state fields and return a dict.

        Keys are the field names from the config. Values are the read values,
        or defaults if the read failed.

        This method never raises — individual field read failures are logged
        at DEBUG level and the default value is used.
        """
        state: dict[str, Any] = {}
        for fdef in self.fields:
            try:
                state[fdef.name] = self._execute_chain(fdef)
            except (OSError, ValueError, OverflowError) as e:
                # Expected errors from memory reads (process gone, bad address, etc.)
                logger.debug("%s: read error: %s", fdef.name, e)
                state[fdef.name] = fdef.default
            except Exception as e:
                # Unexpected errors — log louder so bugs aren't silently swallowed
                logger.warning("%s: unexpected error during read: %s", fdef.name, e)
                state[fdef.name] = fdef.default
        return state

    def read_field(self, name: str) -> Any:
        """Read a single state field by name.

        Returns the value, or the field's default if not found / read fails.
        """
        for fdef in self.fields:
            if fdef.name == name:
                try:
                    return self._execute_chain(fdef)
                except (OSError, ValueError, OverflowError) as e:
                    logger.debug("%s: read error: %s", name, e)
                    return fdef.default
                except Exception as e:
                    logger.warning("%s: unexpected error during read: %s", name, e)
                    return fdef.default
        raise KeyError(f"No state field named '{name}'")

    def list_fields(self) -> list[str]:
        """List all configured state field names."""
        return [f.name for f in self.fields]

    def describe(self) -> str:
        """Human-readable description of all configured fields and their chains."""
        lines = [
            f"GameStateReader ({len(self.fields)} fields, assembly={self.assembly}):"
        ]
        for fdef in self.fields:
            chain_str = " -> ".join(str(s) for s in fdef.chain)
            desc = f" ({fdef.description})" if fdef.description else ""
            lines.append(f"  {fdef.name}: {fdef.read_type} = {chain_str}{desc}")
        return "\n".join(lines)
