import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Install the Nim dependencies pinned in nimlang.lock, then compile the Nim test
# modules into extension modules next to this file.
subprocess.run([sys.executable, "-m", "nimlang", "sync"], check=True, cwd=HERE.parent)
subprocess.run(
    [sys.executable, "-m", "nimlang", "build-ext", str(HERE / "vander.nim"), str(HERE / "checks.nim")],
    check=True,
)
sys.path.insert(0, str(HERE))
