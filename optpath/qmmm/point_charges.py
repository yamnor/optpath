"""Point charge helpers."""

from __future__ import annotations

from pathlib import Path


def load_xyzq(path: str | Path) -> list[dict]:
    """Load MM point charges from an xyzq file.

    Supported formats (whitespace-separated, one charge per line):

        x  y  z  charge                      (legacy, no element)
        symbol  x  y  z  charge              (preferred)

    Lines starting with '#' and blank lines are ignored.
    The 'symbol' field is used for vdW parameter lookup; it defaults to 'X'
    (unknown) when not present.
    """
    charges: list[dict] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 4:
            # legacy: x y z charge
            x, y, z, q = map(float, parts)
            symbol = "X"
        elif len(parts) >= 5:
            # preferred: symbol x y z charge
            symbol = parts[0].capitalize()
            x, y, z, q = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        else:
            continue
        charges.append({"symbol": symbol, "x": x, "y": y, "z": z, "charge": q})
    return charges
