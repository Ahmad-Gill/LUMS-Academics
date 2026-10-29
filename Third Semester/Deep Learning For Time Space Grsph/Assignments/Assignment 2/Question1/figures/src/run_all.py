"""Regenerate every Question 1 figure: python figures/src/run_all.py"""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in sorted(HERE.glob("fig*.py")):
    runpy.run_path(str(script), run_name="__main__")
