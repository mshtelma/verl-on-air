#!/usr/bin/env python3
"""Image size gate: fail unless the image exists, its size is measurable, and it is under the gate.

    python3 scripts/image_size.py <image> <max_gb>

Integer bytes throughout; ``max_gb`` is DECIMAL gigabytes (10^9 bytes) -- the stricter reading of
AI Runtime's "20 GB" registration limit. GiB is printed alongside for comparison only.
Replaces a `docker inspect | bc` pipeline that printed "OK" when the image was missing.
"""
from __future__ import annotations

import subprocess
import sys
from decimal import Decimal, InvalidOperation

GB, GIB = 10**9, 2**30


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    image, max_gb = argv[1], argv[2]
    try:
        limit = int(Decimal(max_gb) * GB)
    except InvalidOperation:
        print(f"size gate: MAX_IMAGE_GB={max_gb!r} is not a number", file=sys.stderr)
        return 2
    r = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Size}}"],
                       capture_output=True, text=True)
    raw = r.stdout.strip()
    if r.returncode != 0 or not raw.isdigit():
        print(f"size gate: cannot measure {image}: {(r.stderr or raw).strip() or 'no output'}",
              file=sys.stderr)
        print("  build it first (make build), or check the tag in config.env", file=sys.stderr)
        return 1
    size = int(raw)
    print(f"image size: {size:,} bytes = {size / GB:.2f} GB ({size / GIB:.2f} GiB); "
          f"gate {limit:,} bytes ({max_gb} GB)")
    if size > limit:
        print("TOO LARGE. Registration will time out replicating this image. Size levers, cheapest first:\n"
              "  * confirm UV_NO_CACHE=1 took effect (the uv cache is ~11 GB)\n"
              "  * --build-arg WITH_VIDEO=0 (drops ffmpeg + torchcodec)\n"
              "  * drop nvidia-modelopt if Megatron-Bridge tolerates it\n"
              f"  * make layers   (largest layers of {image})", file=sys.stderr)
        return 1
    print("OK — under the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
