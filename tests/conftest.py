import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Compile the Nim test modules into extension modules next to this file.
subprocess.run(
    [sys.executable, "-m", "nimlang", "build-ext", str(HERE / "vander.nim"), str(HERE / "checks.nim")],
    check=True,
)
sys.path.insert(0, str(HERE))
