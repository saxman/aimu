from pathlib import Path

root = Path(__file__).parents[1]

# Path to the tests directory
tests = root / "tests"

# Path to the aimu package directory
package = root / "aimu"

# Path to transient output directory
output = root / "output"

# Path to runnable examples directory
examples = root / "examples"

# Path to Agent skills directory (demo skills live under examples/)
skills = examples / "skills"
