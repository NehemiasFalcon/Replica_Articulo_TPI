from pathlib import Path
import sys

bad = []
for path in sorted(Path("src").rglob("*.py")):
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.rstrip(" \t") != line:
            bad.append(f"{path.as_posix()}:{lineno}: TRAILING_WHITESPACE")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print("Lint gate passed")
