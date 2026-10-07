"""Center-crop scene photos to the 1600x1600 format used by letter assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", nargs="?", default="assets_scenes/source_photos")
    parser.add_argument("output_dir", nargs="?", default="assets_scenes/square_1600")
    parser.add_argument("--size", type=int, default=1600)
    args = parser.parse_args()

    source_dir, output_dir = Path(args.source_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_dir.glob("*.jpg"))
    if len(sources) != 20:
        raise ValueError(f"Expected exactly 20 scene sources, found {len(sources)}")

    for source in sources:
        with Image.open(source) as image:
            square = ImageOps.fit(
                image.convert("RGB"), (args.size, args.size),
                method=Image.Resampling.LANCZOS, centering=(0.5, 0.5),
            )
            destination = output_dir / f"{source.stem}.png"
            square.save(destination, optimize=True)
            print(f"{destination}: {square.size[0]}x{square.size[1]}")


if __name__ == "__main__":
    main()
