"""Measure executable line coverage of the simulator using stdlib trace."""
from pathlib import Path
import sys
import trace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = sorted((ROOT / "kv_cache").glob("*.py"))

suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
tracer = trace.Trace(count=True, trace=False)
result = tracer.runfunc(unittest.TextTestRunner(verbosity=0).run, suite)
if not result.wasSuccessful():
    raise SystemExit(1)

counts = tracer.results().counts
total = covered = 0
for path in SOURCES:
    executable = set(trace._find_executable_linenos(str(path)))
    hit = {line for (filename, line), count in counts.items()
           if Path(filename).resolve() == path and count > 0}
    observed = len(executable & hit)
    total += len(executable)
    covered += observed
    print(f"{path.name}: {observed}/{len(executable)} ({observed / len(executable):.1%})")
print(f"TOTAL: {covered}/{total} ({covered / total:.1%})")
if covered / total < 0.70:
    raise SystemExit(2)
