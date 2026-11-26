"""Procedural kitchen and table textures for realistic fruit composites."""

from __future__ import annotations

import math
import random
from PIL import Image, ImageDraw, ImageFilter

STYLES = ("wood", "marble", "countertop", "linen")


def generate_wood_texture(width: int, height: int, rng: random.Random) -> Image.Image:
    """Generate wooden cutting board / tabletop texture with natural grain."""
    # Generate on lower resolution and upscale for speed and soft organic look
    sw, sh = max(100, width // 4), max(100, height // 4)
    base_r = rng.randint(160, 205)
    base_g = rng.randint(110, 155)
    base_b = rng.randint(65, 105)

    base = Image.new("RGB", (sw, sh), (base_r, base_g, base_b))
    draw = ImageDraw.Draw(base)

    num_rings = rng.randint(8, 16)
    for _ in range(num_rings):
        y = rng.randint(0, sh)
        darkness = rng.uniform(0.75, 1.25)
        color = (
            max(0, min(255, int(base_r * darkness))),
            max(0, min(255, int(base_g * darkness))),
            max(0, min(255, int(base_b * darkness))),
        )
        line_w = rng.randint(1, 4)
        freq = rng.uniform(0.02, 0.08)
        offset = rng.uniform(0, 100)
        points = []
        for x in range(0, sw, 8):
            wavy_y = y + int(math.sin(x * freq + offset) * 6)
            points.append((x, wavy_y))
        if len(points) > 1:
            draw.line(points, fill=color, width=line_w)

    base = base.filter(ImageFilter.GaussianBlur(radius=1.5))
    return base.resize((width, height), Image.Resampling.BILINEAR)


def generate_marble_texture(width: int, height: int, rng: random.Random) -> Image.Image:
    """Generate kitchen marble or quartz surface texture."""
    sw, sh = max(100, width // 4), max(100, height // 4)
    base_val = rng.randint(215, 240)
    base = Image.new("RGB", (sw, sh), (base_val, base_val, base_val - 5))
    draw = ImageDraw.Draw(base)

    num_veins = rng.randint(3, 6)
    for _ in range(num_veins):
        cur_x = rng.randint(0, sw)
        cur_y = 0
        points = [(cur_x, cur_y)]
        vein_color = rng.randint(150, 195)
        color = (vein_color, vein_color + 5, vein_color + 10)
        while cur_y < sh:
            cur_y += rng.randint(8, 20)
            cur_x += rng.randint(-10, 10)
            points.append((cur_x, cur_y))
        if len(points) > 1:
            draw.line(points, fill=color, width=rng.randint(1, 3))

    base = base.filter(ImageFilter.GaussianBlur(radius=2.0))
    return base.resize((width, height), Image.Resampling.BILINEAR)


def generate_countertop_texture(width: int, height: int, rng: random.Random) -> Image.Image:
    """Generate modern speckled kitchen countertop with directional lighting."""
    sw, sh = max(100, width // 4), max(100, height // 4)
    r_val = rng.randint(180, 220)
    g_val = rng.randint(180, 220)
    b_val = rng.randint(175, 215)
    base = Image.new("RGB", (sw, sh), (r_val, g_val, b_val))
    draw = ImageDraw.Draw(base)

    num_speckles = rng.randint(100, 300)
    for _ in range(num_speckles):
        x = rng.randint(0, sw - 1)
        y = rng.randint(0, sh - 1)
        speckle_delta = rng.choice([-30, -15, 15, 30])
        s_color = (
            max(0, min(255, r_val + speckle_delta)),
            max(0, min(255, g_val + speckle_delta)),
            max(0, min(255, b_val + speckle_delta)),
        )
        draw.point((x, y), fill=s_color)

    base = base.filter(ImageFilter.GaussianBlur(radius=1.0))
    return base.resize((width, height), Image.Resampling.BILINEAR)


def generate_linen_texture(width: int, height: int, rng: random.Random) -> Image.Image:
    """Generate woven tablecloth texture."""
    sw, sh = max(100, width // 4), max(100, height // 4)
    base_tone = rng.choice([
        (225, 220, 210),  # beige
        (215, 225, 220),  # sage tint
        (220, 220, 225),  # light cool gray
    ])
    base = Image.new("RGB", (sw, sh), base_tone)
    draw = ImageDraw.Draw(base)

    step = rng.randint(3, 6)
    for x in range(0, sw, step):
        delta = rng.randint(-10, 10)
        c = (
            max(0, min(255, base_tone[0] + delta)),
            max(0, min(255, base_tone[1] + delta)),
            max(0, min(255, base_tone[2] + delta)),
        )
        draw.line([(x, 0), (x, sh)], fill=c, width=1)

    for y in range(0, sh, step):
        delta = rng.randint(-10, 10)
        c = (
            max(0, min(255, base_tone[0] + delta)),
            max(0, min(255, base_tone[1] + delta)),
            max(0, min(255, base_tone[2] + delta)),
        )
        draw.line([(0, y), (sw, y)], fill=c, width=1)

    base = base.filter(ImageFilter.GaussianBlur(radius=0.8))
    return base.resize((width, height), Image.Resampling.BILINEAR)


def generate_texture(width: int, height: int, style: str, rng: random.Random) -> Image.Image:
    """Master procedural texture generator dispatch."""
    if style == "wood":
        return generate_wood_texture(width, height, rng)
    elif style == "marble":
        return generate_marble_texture(width, height, rng)
    elif style == "countertop":
        return generate_countertop_texture(width, height, rng)
    elif style == "linen":
        return generate_linen_texture(width, height, rng)
    else:
        return generate_wood_texture(width, height, rng)
