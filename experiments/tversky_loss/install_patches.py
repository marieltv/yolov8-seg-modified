import os
import shutil
import sys


def find_ultralytics_path() -> str:
    """
    Locate the installation path of the Ultralytics package.

    This function attempts to import the `ultralytics` package and returns
    the absolute directory path where it is installed. If Ultralytics is not
    installed in the current Python environment, the script will terminate
    with an informative error message.

    Returns:
        str: Absolute path to the Ultralytics package directory.

    Raises:
        SystemExit: If Ultralytics is not installed.
    """
    try:
        import ultralytics
        return os.path.dirname(ultralytics.__file__)
    except ImportError:
        print(" Ultralytics is not installed.")
        print("Install it first with:\n  pip install ultralytics")
        sys.exit(1)


def install_losses_tversky(ultra_utils_path: str) -> None:
    """
    Install the custom Tversky/Focal-Tversky loss module into Ultralytics.

    This function copies `losses_tversky.py` from the local experiment directory
    into the official Ultralytics `utils` directory so that it can be imported
    internally by the modified YOLO loss pipeline.

    Source:
        experiments/tversky/losses_tversky.py

    Destination:
        ultralytics/utils/losses_tversky.py

    Args:
        ultra_utils_path (str): Path to `ultralytics/utils` directory.

    Raises:
        FileNotFoundError: If the source `losses_tversky.py` file is missing.
    """
    src = os.path.join("experiments", "tversky_loss", "losses_tversky.py")
    dst = os.path.join(ultra_utils_path, "losses_tversky.py")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing source file: {src}")

    shutil.copy(src, dst)
    print(f" Installed losses_tversky.py → {dst}")


def patch_loss_py(ultra_utils_path: str) -> None:
    """
    Patch the original Ultralytics loss file with a modified version.

    This function performs the following operations:
    1. Creates a backup of the original `loss.py` as `loss_original.py`
       (only if the backup does not already exist).
    2. Replaces the active `loss.py` with a modified version that integrates
       the Focal-Tversky loss for segmentation.

    Backup:
        ultralytics/utils/loss_original.py

    Replacement:
        experiments/tversky_loss/modified_loss.py → ultralytics/utils/loss.py

    Args:
        ultra_utils_path (str): Path to `ultralytics/utils` directory.

    Raises:
        FileNotFoundError: If either the original YOLO loss file or the modified
                           loss file is missing.
    """
    original_loss = os.path.join(ultra_utils_path, "loss.py")
    backup_loss = os.path.join(ultra_utils_path, "loss_original.py")
    modified_loss = os.path.join("experiments", "tversky_loss", "modified_loss.py")

    if not os.path.exists(modified_loss):
        raise FileNotFoundError(f"Missing modified loss file: {modified_loss}")

    if not os.path.exists(original_loss):
        raise FileNotFoundError(
            f"Original Ultralytics loss.py not found at: {original_loss}"
        )

    # Backup original loss only once
    if not os.path.exists(backup_loss):
        shutil.copy(original_loss, backup_loss)
        print(f" Backup created: {backup_loss}")
    else:
        print(" Backup already exists: loss_original.py (not overwritten)")

    # Replace active loss.py with custom implementation
    shutil.copy(modified_loss, original_loss)
    print(f" Installed modified loss.py → {original_loss}")


def main() -> None:
    """
    Main entry point for the YOLO custom patch installer.

    This function:
    - Detects the Ultralytics installation path
    - Locates the `utils` directory
    - Installs the custom Tversky/Focal-Tversky loss module
    - Replaces the official YOLO loss with a modified implementation

    After successful execution, Ultralytics will use the modified loss logic
    for all future training runs within this Python environment.
    """
    print("\n==============================")
    print(" YOLOv8 CUSTOM PATCH INSTALL ")
    print("==============================\n")

    ultra_path = find_ultralytics_path()
    ultra_utils_path = os.path.join(ultra_path, "utils")

    print(f" Ultralytics detected at:\n  {ultra_path}\n")

    if not os.path.isdir(ultra_utils_path):
        raise NotADirectoryError(
            f"Ultralytics utils directory not found: {ultra_utils_path}"
        )

    install_losses_tversky(ultra_utils_path)
    patch_loss_py(ultra_utils_path)

    print("\n INSTALLATION COMPLETE")
    print("You can now run Tversky / Focal-Tversky experiments safely.")
    print("\nIf needed, the original loss is preserved as:")
    print("  ultralytics/utils/loss_original.py\n")


if __name__ == "__main__":
    main()
