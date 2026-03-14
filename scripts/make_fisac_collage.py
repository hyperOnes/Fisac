from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


SOURCE_IMAGE = Path("/Users/sebastian/Downloads/CleanShot 2026-03-14 at 12.28.37@2x.png")
OUTPUT_IMAGE = Path("/Users/sebastian/Fisac/artifacts/miguel-fisac-collage.png")

# Manually verified image-only bounds for each thumbnail in the screenshot.
CROPS = {
    "pagoda_sepia": (36, 31, 416, 390),
    "balconies": (457, 31, 753, 390),
    "pagoda_bw": (794, 31, 1083, 390),
    "interior": (1124, 31, 1629, 390),
    "steps": (1670, 31, 2042, 390),
    "texture": (2083, 31, 2451, 390),
    "portrait": (2492, 31, 2781, 390),
    "pagoda_trees": (36, 527, 394, 886),
    "concrete_detail": (435, 527, 865, 886),
    "construction": (906, 527, 1206, 886),
    "beams": (1247, 527, 1611, 886),
    "roofline": (1652, 527, 2276, 886),
    "pagoda_exterior": (2317, 527, 2781, 886),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a poster-style collage from Fisac thumbnails.")
    parser.add_argument("--source", type=Path, default=SOURCE_IMAGE)
    parser.add_argument("--output", type=Path, default=OUTPUT_IMAGE)
    return parser.parse_args()


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def make_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    base = Image.new("RGBA", size, (243, 239, 232, 255))
    px = base.load()

    top = (249, 246, 240)
    bottom = (230, 225, 218)
    for y in range(height):
        mix = y / max(height - 1, 1)
        row = tuple(int(top[i] * (1 - mix) + bottom[i] * mix) for i in range(3))
        for x in range(width):
            px[x, y] = (*row, 255)

    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.ellipse((60, 40, 1160, 1140), fill=(214, 191, 170, 90))
    draw.ellipse((1820, 760, 3180, 2100), fill=(188, 196, 202, 58))
    draw.rounded_rectangle((1820, 160, 3050, 620), radius=48, fill=(255, 255, 255, 72))
    draw.rounded_rectangle((160, 1350, 1180, 1740), radius=56, fill=(255, 250, 244, 74))
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    base.alpha_composite(glow)

    noise = Image.effect_noise(size, 10).convert("L")
    noise = ImageEnhance.Contrast(noise).enhance(1.8)
    grain = Image.new("RGBA", size, (120, 108, 94, 0))
    grain.putalpha(noise.point(lambda value: int(value * 0.07)))
    base.alpha_composite(grain)
    return base


def prepare_crop(source: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    crop = source.crop(box).convert("RGB")
    crop = ImageEnhance.Contrast(crop).enhance(1.06)
    crop = ImageEnhance.Color(crop).enhance(0.92)
    crop = ImageEnhance.Sharpness(crop).enhance(1.08)
    return crop


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return fitted


def make_card(
    image: Image.Image,
    max_size: tuple[int, int],
    matte: int = 18,
    radius: int = 30,
) -> Image.Image:
    framed = fit_image(image, *max_size)
    width = framed.width + matte * 2
    height = framed.height + matte * 2

    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=radius,
        fill=(248, 245, 239, 255),
        outline=(224, 216, 206, 255),
        width=2,
    )
    card.paste(framed, (matte, matte))

    inner = Image.new("RGBA", card.size, (0, 0, 0, 0))
    inner_draw = ImageDraw.Draw(inner)
    inner_draw.rounded_rectangle(
        (matte - 2, matte - 2, width - matte + 1, height - matte + 1),
        radius=max(radius - 8, 16),
        outline=(235, 229, 221, 160),
        width=2,
    )
    card.alpha_composite(inner)
    return card


def alpha_shadow(alpha: Image.Image, color: tuple[int, int, int, int], blur: int) -> Image.Image:
    shadow = Image.new("RGBA", alpha.size, (*color[:3], 0))
    shadow_alpha = alpha.point(lambda value: int(value * color[3] / 255))
    shadow.putalpha(shadow_alpha)
    return shadow.filter(ImageFilter.GaussianBlur(blur))


def paste_with_shadow(
    canvas: Image.Image,
    overlay: Image.Image,
    center: tuple[int, int],
    angle: float,
    shadow_offset: tuple[int, int] = (0, 22),
    shadow_color: tuple[int, int, int, int] = (33, 24, 16, 86),
    shadow_blur: int = 26,
) -> None:
    rotated = overlay.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    alpha = rotated.getchannel("A")
    shadow = alpha_shadow(alpha, shadow_color, shadow_blur)

    left = int(center[0] - rotated.width / 2)
    top = int(center[1] - rotated.height / 2)

    canvas.alpha_composite(shadow, (left + shadow_offset[0], top + shadow_offset[1]))
    canvas.alpha_composite(rotated, (left, top))


def draw_title(canvas: Image.Image) -> None:
    title_font = load_font("/System/Library/Fonts/Supplemental/Baskerville.ttc", 118)
    subtitle_font = load_font("/System/Library/Fonts/Supplemental/GillSans.ttc", 30)
    small_font = load_font("/System/Library/Fonts/Supplemental/GillSans.ttc", 22)

    draw = ImageDraw.Draw(canvas)
    draw.text((148, 128), "Miguel Fisac", font=title_font, fill=(53, 42, 33, 255))
    draw.rounded_rectangle((154, 312, 358, 316), radius=2, fill=(134, 106, 82, 255))
    draw.text(
        (154, 336),
        "Concrete rhythms, pagodas, and structural experiments",
        font=subtitle_font,
        fill=(88, 74, 62, 255),
    )
    draw.text(
        (156, 388),
        "Collage assembled from the original search thumbnails",
        font=small_font,
        fill=(120, 104, 90, 255),
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.source).convert("RGB")
    canvas = make_background((3200, 1900))

    prepared = {name: prepare_crop(source, box) for name, box in CROPS.items()}

    placements = [
        ("pagoda_sepia", (540, 540), (710, 770), -4.0),
        ("balconies", (370, 370), (1230, 470), 3.0),
        ("pagoda_bw", (300, 420), (1335, 910), 6.0),
        ("interior", (760, 520), (2180, 450), -1.2),
        ("steps", (340, 340), (2300, 860), -3.2),
        ("texture", (320, 400), (2710, 690), 2.6),
        ("portrait", (260, 360), (2930, 335), 1.6),
        ("pagoda_trees", (320, 480), (420, 1320), -6.0),
        ("concrete_detail", (440, 360), (1030, 1365), -2.4),
        ("construction", (250, 360), (1410, 1245), 2.0),
        ("beams", (330, 420), (1720, 1195), 5.0),
        ("roofline", (720, 430), (2140, 1430), 1.2),
        ("pagoda_exterior", (430, 350), (2785, 1305), 3.6),
    ]

    draw_title(canvas)
    for name, max_size, center, angle in placements:
        card = make_card(prepared[name], max_size=max_size)
        paste_with_shadow(canvas, card, center=center, angle=angle)

    vignette = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.rounded_rectangle((28, 28, 3172, 1872), radius=48, outline=(120, 98, 78, 46), width=2)
    canvas.alpha_composite(vignette)

    final = canvas.convert("RGB")
    final.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
