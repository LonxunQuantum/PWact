"""
NEP-based DIRECT sampling — re-exports from MatPL's src.sampling.

This module exists for backward compatibility with code that imported from
pwact.agent.direct. New code should import directly from src.sampling.
"""
import os
import sys
from pathlib import Path

_matpl_root = os.environ.get("MATPL_ROOT")
if _matpl_root is None:
    _matpl_root = str(Path(__file__).resolve().parents[3] / "MatPL")
    if os.path.exists(_matpl_root):
        os.environ["MATPL_ROOT"] = _matpl_root
if _matpl_root not in sys.path:
    sys.path.insert(0, _matpl_root)

from src.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
from src.sampling.nep_describer import NEPStructure

__all__ = ["DIRECTSampler", "BirchClustering", "SelectKFromClusters", "NEPStructure"]
