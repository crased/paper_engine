import subprocess



def launch_label_studio(env):
    """Launch Label Studio for annotation.

    Args:
        env: Environment dictionary with LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED set

    Returns:
        subprocess.Popen object for the label-studio process
    """
    try:
        command = ["label-studio", "start", "--port", "8080"]
        label_process = subprocess.Popen(command, env=env, start_new_session=True)
        print("\nLabel Studio started on http://localhost:8080")
        return label_process
    except FileNotFoundError:
        print("\nERROR: label-studio command not found.")
        print("Install with: pip install label-studio")
        print("Then ensure it's in your PATH.")
        sys.exit(1)
    except subprocess.SubprocessError as e:
        print(f"\nERROR: Failed to start Label Studio: {e}")
        print("Try running manually: label-studio start --port 8080")
        sys.exit(1)