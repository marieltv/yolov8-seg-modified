"""
Ultralytics Runtime Patch for Custom CoordAtt Module Registration.

This file injects a custom Coordinate Attention (CoordAtt) module into
the Ultralytics YOLOv8 runtime so that it can be referenced inside
model YAML files.

Why this is required:
---------------------
Ultralytics parses model YAML files dynamically and resolves layer
classes using global lookups inside:

- ultralytics.nn.modules
- ultralytics.nn.tasks

Custom modules that are not part of the official Ultralytics codebase
are therefore NOT visible to the YAML parser by default.

This script manually registers the CoordAtt class in both namespaces,
allowing YAML entries such as:

    - [-1, 1, CoordAtt, [128, 64]]

to be parsed correctly without modifying the Ultralytics source files.

Scope:
------
- This patch affects ONLY the current Python environment.
- It must be executed before model creation (YOLO(model_yaml)).
- It does not permanently modify Ultralytics installation files.

Intended usage:
---------------
Call this script once before training or import it at the top of the
training script that uses CoordAtt.

Example:
--------
    python experiments/coordinate_attention/install_patches.py
"""

from typing import Type
import inspect

from coordinate_attention import CoordAtt

import ultralytics.nn.modules as M
import ultralytics.nn.tasks as T


def register_coordatt() -> None:
    """
    Register the custom CoordAtt module into Ultralytics namespaces.

    This function injects CoordAtt into:
        - ultralytics.nn.modules
        - ultralytics.nn.tasks

    This is necessary because:
        - Ultralytics uses wildcard imports:
              from .modules import *
        - YAML parsing relies on globals() lookup inside tasks.py

    After registration, CoordAtt becomes visible to:
        - YOLO YAML model parser
        - model.task and model.model builders

    Returns:
        None
    """
    # Register for module-level resolution
    M.CoordAtt = CoordAtt

    # Register for task-level global resolution
    T.CoordAtt = CoordAtt


def verify_registration() -> None:
    """
    Verify that CoordAtt is correctly registered in Ultralytics.

    Prints the constructor signature from both namespaces to ensure:
        - The class is reachable
        - The expected __init__ signature is preserved

    Returns:
        None
    """
    print("modules CoordAtt:", inspect.signature(M.CoordAtt))
    print("tasks   CoordAtt:", inspect.signature(T.CoordAtt))


# ----------------------------
# Execute patch on import/run
# ----------------------------
if __name__ == "__main__":
    register_coordatt()
    verify_registration()
