"""Run the stdlib regression suite from any working directory."""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
