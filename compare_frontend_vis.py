#!/usr/bin/env python3
"""Build a 2x2 comparison grid from four frontend run vis directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=/abs/path/to/run/room0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-step", type=int, default=20)
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec: {spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = ImageFont.load_default()
    title_h = 24
    canvas = Image.new("RGB", (image.width, image.height + title_h), color=(255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, title_h), fill=(245, 245, 245))
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    return canvas


def blank_tile(size: tuple[int, int], title: str) -> Image.Image:
    canvas = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    draw.text((8, 28), "missing", fill=(128, 128, 128), font=font)
    return canvas


def save_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 4, tile_width: int = 480) -> None:
    if not image_paths:
        return
    loaded: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            loaded.append(image.convert("RGB").copy())
    tile_height = int(tile_width * loaded[0].height / loaded[0].width)
    rows = (len(loaded) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(255, 255, 255))
    for idx, image in enumerate(loaded):
        thumb = image.resize((tile_width, tile_height))
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_height
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    runs = [parse_run_spec(spec) for spec in args.run]
    if len(runs) != 4:
        raise ValueError("Expected exactly four --run specs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dirs = [(label, room0_dir / "local_memory_audit" / "vis") for label, room0_dir in runs]
    frame_names: set[str] = set()
    for _, vis_dir in vis_dirs:
        frame_names.update(path.name for path in vis_dir.glob("*.png"))
    frame_list = sorted(frame_names)
    if not frame_list:
        raise RuntimeError("No vis pngs found in any run.")

    combined_dir = args.output_dir / "frames"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_paths: list[Path] = []

    first_tiles: list[Image.Image] = []
    for label, vis_dir in vis_dirs:
        first_path = vis_dir / frame_list[0]
        if first_path.exists():
            with Image.open(first_path) as image:
                first_tiles.append(add_title(image.convert("RGB"), label))
        else:
            raise RuntimeError(f"First frame missing for run {label}: {first_path}")
    tile_w, tile_h = first_tiles[0].size

    for frame_name in frame_list:
        tiles: list[Image.Image] = []
        for label, vis_dir in vis_dirs:
            image_path = vis_dir / frame_name
            if image_path.exists():
                with Image.open(image_path) as image:
                    tiles.append(add_title(image.convert("RGB"), label))
            else:
                tiles.append(blank_tile((tile_w, tile_h), label))

        canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), color=(255, 255, 255))
        canvas.paste(tiles[0], (0, 0))
        canvas.paste(tiles[1], (tile_w, 0))
        canvas.paste(tiles[2], (0, tile_h))
        canvas.paste(tiles[3], (tile_w, tile_h))
        out_path = combined_dir / frame_name
        canvas.save(out_path)
        combined_paths.append(out_path)

    sample_paths = [path for idx, path in enumerate(combined_paths) if idx % max(1, args.sample_step) == 0]
    save_contact_sheet(sample_paths, args.output_dir / "contact_sheet.png", columns=2, tile_width=720)

    summary_path = args.output_dir / "summary.md"
    summary_lines = [
        "# Frontend Threshold Compare",
        "",
        "Runs:",
        *(f"- `{label}`: `{room0_dir}`" for label, room0_dir in runs),
        "",
        f"- frame_count: `{len(frame_list)}`",
        f"- combined_frames_dir: `{combined_dir}`",
        f"- contact_sheet: `{args.output_dir / 'contact_sheet.png'}`",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
