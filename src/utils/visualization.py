"""Visualization helpers for RGB-D frontend inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.data_structures import Proposal2D


def proposal_overlay_image(
    rgb: np.ndarray,
    proposals: Iterable[Proposal2D],
    alpha: float = 0.35,
) -> Image.Image:
    """Render masks and bounding boxes on top of an RGB frame."""
    proposals = list(proposals)
    base = rgb.astype(np.uint8).copy()
    canvas = base.astype(np.float32)

    for proposal in proposals:
        color = np.asarray(_proposal_color(proposal.proposal_id), dtype=np.float32)
        mask = proposal.mask.astype(bool)
        if mask.any():
            canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for proposal in proposals:
        color = _proposal_color(proposal.proposal_id)
        x1, y1, x2, y2 = _bbox_for_draw(proposal)
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"#{proposal.proposal_id} {proposal.confidence:.2f}"
        text_bbox = _text_bbox(draw, font, label, x1, y1)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)

    return image


def runtime_proposal_class_overlay_image(
    rgb: np.ndarray,
    proposals: Iterable[Proposal2D],
    proposal_profiles: Iterable[Any],
    alpha: float = 0.35,
) -> Image.Image:
    """Render runtime proposal classes (object/background/uncertain) overlays."""
    proposals = list(proposals)
    profile_map: dict[int, Any] = {
        int(getattr(profile, "proposal_id", -1)): profile for profile in proposal_profiles
    }

    base = rgb.astype(np.uint8).copy()
    canvas = base.astype(np.float32)

    for proposal in proposals:
        profile = profile_map.get(int(proposal.proposal_id))
        cls_name = _profile_field(profile, "proposal_class", "uncertain")
        color = np.asarray(_proposal_class_color(cls_name), dtype=np.float32)
        mask = proposal.mask.astype(bool)
        if mask.any():
            canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for proposal in proposals:
        profile = profile_map.get(int(proposal.proposal_id))
        cls_name = _profile_field(profile, "proposal_class", "uncertain")
        objectness = float(_profile_field(profile, "objectness_score", 0.0))
        backgroundness = float(_profile_field(profile, "backgroundness_score", 0.0))
        color = _proposal_class_color(cls_name)

        x1, y1, x2, y2 = _bbox_for_draw(proposal)
        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"#{proposal.proposal_id} {cls_name} O{objectness:.2f} B{backgroundness:.2f}"
        text_bbox = _text_bbox(draw, font, label, x1, y1)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)

    return image


def save_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int = 4,
    tile_width: int = 360,
) -> None:
    """Create a compact contact sheet for a list of saved images."""
    if not image_paths:
        return

    loaded: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            loaded.append(image.convert("RGB").copy())

    tile_height = int(tile_width * loaded[0].height / loaded[0].width)
    rows = (len(loaded) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(255, 255, 255))

    for index, image in enumerate(loaded):
        thumb = image.resize((tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)


def runtime_group_overlay_image(
    rgb: np.ndarray,
    groups: Iterable[Any],
    alpha: float = 0.35,
) -> Image.Image:
    """Render merged runtime groups with object-link annotations."""
    groups = list(groups)
    base = rgb.astype(np.uint8).copy()
    canvas = base.astype(np.float32)

    for group in groups:
        color = np.asarray(_proposal_color(int(getattr(group, "group_id"))), dtype=np.float32)
        mask = np.asarray(getattr(group, "merged_mask")).astype(bool)
        if mask.any():
            canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for group in groups:
        group_id = int(getattr(group, "group_id"))
        color = _proposal_color(group_id)
        bbox = np.asarray(getattr(group, "merged_bbox_xyxy")).astype(np.int32)
        x1, y1, x2, y2 = bbox.tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        linked_object = getattr(group, "linked_object_id")
        whole_prior_used = bool(getattr(group, "whole_prior_used"))
        label = f"G{group_id}"
        if linked_object is not None:
            label += f" O{linked_object}"
        if whole_prior_used:
            label += " P"

        text_bbox = _text_bbox(draw, font, label, x1, y1)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)

    return image


def save_side_by_side(left: Image.Image, right: Image.Image, output_path: Path) -> None:
    """Save two images as a horizontal side-by-side panel."""
    left_rgb = left.convert("RGB")
    right_rgb = right.convert("RGB")
    canvas = Image.new("RGB", (left_rgb.width + right_rgb.width, max(left_rgb.height, right_rgb.height)))
    canvas.paste(left_rgb, (0, 0))
    canvas.paste(right_rgb, (left_rgb.width, 0))
    canvas.save(output_path)


def _proposal_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (0, 191, 255),
        (60, 179, 113),
        (255, 215, 0),
        (186, 85, 211),
        (255, 140, 0),
        (64, 224, 208),
        (220, 20, 60),
        (123, 104, 238),
        (46, 139, 87),
    ]
    return palette[index % len(palette)]


def _proposal_class_color(class_name: str) -> tuple[int, int, int]:
    if class_name == "object_like":
        return (88, 201, 126)
    if class_name == "background_like":
        return (248, 119, 84)
    return (244, 209, 96)


def _profile_field(profile: Any, name: str, default: Any) -> Any:
    if profile is None:
        return default
    if isinstance(profile, dict):
        return profile.get(name, default)
    return getattr(profile, name, default)


def _bbox_for_draw(proposal: Proposal2D) -> tuple[int, int, int, int]:
    xs = np.nonzero(proposal.mask)[1]
    ys = np.nonzero(proposal.mask)[0]
    if xs.size == 0 or ys.size == 0:
        x1, y1, x2, y2 = proposal.bbox_xyxy.astype(np.int32).tolist()
        return x1, y1, x2, y2

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    label: str,
    x: int,
    y: int,
) -> tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        try:
            return draw.textbbox((x, y), label, font=font)
        except Exception:
            pass

    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(label)
        return x, y, x + (right - left), y + (bottom - top)

    if hasattr(font, "getsize"):
        width, height = font.getsize(label)
        return x, y, x + width, y + height

    width = 6 * len(label)
    height = 11
    return x, y, x + width, y + height
