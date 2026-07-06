"""Convert an MP4 clip to a README-friendly GIF (resize + fps cap)."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mp4", type=Path)
    parser.add_argument("output_gif", type=Path)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max_frames", type=int, default=120)
    args = parser.parse_args()

    reader = imageio.get_reader(args.input_mp4)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps") or 30.0)
    step = max(1, int(round(src_fps / args.fps)))

    frames: list[np.ndarray] = []
    for i, frame in enumerate(reader):
        if i % step != 0:
            continue
        h, w = frame.shape[:2]
        new_w = args.width
        new_h = int(h * new_w / w)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frames.append(np.asarray(img))
        if len(frames) >= args.max_frames:
            break
    reader.close()

    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output_gif, frames, fps=args.fps, loop=0)
    print(f"Wrote {args.output_gif} ({len(frames)} frames, {args.output_gif.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
