# Usage: interactive run in an ipython / jupyter session to introspect
# the impact of code formatting rules.
import pandas as pd
from pathlib import Path


pyfiles = sorted((Path(__file__).parent.parent / "sklearn").glob("**/*.py"))
lines = []
for pyfile in pyfiles:
    print(f"processing {str(pyfile)}")
    for i, line in enumerate(pyfile.read_text(encoding="utf-8").split("\n")):
        lines.append({
            "filename": str(pyfile),
            "number": i + 1,
            "length": len(line),
        })

lines = pd.DataFrame(lines)
print(lines.sort_values("length", ascending=False).head(30))
