"""M0 smoke test: the package imports and the core scientific stack is present.

The real physics/model/metric/API tests arrive in later milestones
(test_physics.py in M1, etc.). This just gives pytest something to collect and
confirms the environment is wired up.
"""

import importlib


def test_shocklab_imports():
    import shocklab

    assert shocklab.__version__ == "0.1.0"


def test_subpackages_import():
    for mod in ("physics", "data", "models", "train", "eval", "server"):
        importlib.import_module(f"shocklab.{mod}")


def test_scientific_stack_present():
    import numpy  # noqa: F401
    import scipy  # noqa: F401
