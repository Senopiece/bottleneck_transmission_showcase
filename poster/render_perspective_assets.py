from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#08090B"
DARK = "#242833"
LINE = "#B9C5D8"
WHITE = "#FFFFFF"
CYAN = "#5AA9E6"
SCREEN = "#E9EEF6"
SCENE_SCALE = 3


def perspective_coefficients(destination, source):
    """Return Pillow output-to-input homography coefficients."""
    matrix = []
    values = []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype=float), np.asarray(values, dtype=float))


def project(image, output_size, destination):
    source = [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]
    coefficients = perspective_coefficients(destination, source)
    return image.transform(
        output_size,
        Image.Transform.PERSPECTIVE,
        tuple(coefficients),
        Image.Resampling.BICUBIC,
    )


def normalize(vector):
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / length


class PerspectiveCamera:
    """Small pinhole camera used to project every sprite in one 3D scene."""

    def __init__(self, position, target, focal_length, viewport):
        self.position = np.asarray(position, dtype=float)
        forward = normalize(np.asarray(target, dtype=float) - self.position)
        self.right = normalize(np.cross(forward, np.asarray((0.0, 1.0, 0.0))))
        self.up = normalize(np.cross(self.right, forward))
        self.forward = forward
        self.focal_length = float(focal_length)
        self.viewport = viewport

    def project(self, point):
        relative = np.asarray(point, dtype=float) - self.position
        depth = float(np.dot(relative, self.forward))
        if depth <= 0:
            raise ValueError("Sprite is behind the scene camera")
        width, height = self.viewport
        return (
            width * 0.5 + self.focal_length * np.dot(relative, self.right) / depth,
            height * 0.5 - self.focal_length * np.dot(relative, self.up) / depth,
        )


def plane_corners(center, width, height, yaw_degrees=0.0, roll_degrees=0.0):
    """Return clockwise world-space corners for a vertical textured plane."""
    yaw = np.deg2rad(yaw_degrees)
    roll = np.deg2rad(roll_degrees)
    horizontal = np.asarray((np.cos(yaw), 0.0, -np.sin(yaw)))
    vertical = np.asarray((0.0, 1.0, 0.0))
    rolled_horizontal = horizontal * np.cos(roll) + vertical * np.sin(roll)
    rolled_vertical = -horizontal * np.sin(roll) + vertical * np.cos(roll)
    center = np.asarray(center, dtype=float)
    half_horizontal = rolled_horizontal * width * 0.5
    half_vertical = rolled_vertical * height * 0.5
    return [
        center - half_horizontal + half_vertical,
        center + half_horizontal + half_vertical,
        center + half_horizontal - half_vertical,
        center - half_horizontal - half_vertical,
    ]


def render_sprite_plane(
    canvas,
    sprite,
    camera,
    corners,
    shadow_offset=(7, 9),
    shadow_opacity=55,
    shadow_radius=9,
):
    destination = [camera.project(corner) for corner in corners]
    warped = project(sprite, canvas.size, destination)
    shadow = Image.new("RGBA", canvas.size, (8, 9, 11, 0))
    shadow_alpha = warped.getchannel("A").filter(ImageFilter.GaussianBlur(shadow_radius)).point(
        lambda value: value * shadow_opacity // 255
    )
    shadow.putalpha(shadow_alpha)
    canvas.alpha_composite(shadow, shadow_offset)
    canvas.alpha_composite(warped)


def with_shadow(image, offset=(8, 10), radius=10, opacity=95):
    alpha = image.getchannel("A")
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius)).point(
        lambda value: value * opacity // 255
    )
    shadow = Image.new("RGBA", image.size, (8, 9, 11, 0))
    shadow.putalpha(shadow_alpha)
    result = Image.new("RGBA", image.size, (0, 0, 0, 0))
    result.alpha_composite(shadow, offset)
    result.alpha_composite(image)
    return result


def draw_triangle(draw, center, size, fill, outline, width):
    cx, cy = center
    height = size * 0.88
    points = [
        (cx, cy - height / 2),
        (cx - size / 2, cy + height / 2),
        (cx + size / 2, cy + height / 2),
    ]
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def make_flat_pattern():
    image = Image.new("RGBA", (1000, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 992, 212), radius=42, fill=DARK, outline=LINE, width=6)
    draw.rounded_rectangle((62, 78, 126, 142), radius=5, fill=WHITE)

    for index, bit in enumerate("10101"):
        x = 255 + index * 125
        fill = CYAN if bit == "1" else "#26394F"
        outline = WHITE if bit == "1" else CYAN
        draw.ellipse((x - 35, 75, x + 35, 145), fill=fill, outline=outline, width=7)

    draw_triangle(draw, (915, 110), 82, WHITE, WHITE, 5)
    return image


def make_flat_phone():
    image = Image.new("RGBA", (440, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 14, 422, 706), radius=58, fill="#111318", outline=INK, width=8)
    draw.rounded_rectangle((34, 32, 406, 688), radius=45, fill="#F4F7FB", outline="#343A46", width=4)
    draw.ellipse((209, 48, 231, 70), fill="#171A20")
    draw.ellipse((216, 55, 224, 63), fill="#405B7A")

    # Live camera preview of the target behind the phone.
    draw.rounded_rectangle((62, 205, 378, 382), radius=22, fill=SCREEN, outline=LINE, width=4)
    draw.rounded_rectangle((86, 257, 354, 331), radius=14, fill=DARK, outline=LINE, width=4)
    draw.rectangle((104, 279, 128, 303), fill=WHITE)
    for index, bit in enumerate("10101"):
        x = 166 + index * 32
        fill = CYAN if bit == "1" else "#26394F"
        draw.ellipse(
            (x - 9, 282, x + 9, 300),
            fill=fill,
            outline=WHITE if bit == "1" else CYAN,
            width=2,
        )
    draw_triangle(draw, (332, 291), 25, WHITE, WHITE, 2)
    draw.rounded_rectangle((76, 238, 364, 350), radius=10, outline=CYAN, width=4)
    draw.line((220, 224, 220, 364), fill=CYAN, width=2)
    draw.line((68, 294, 372, 294), fill=CYAN, width=2)
    draw.rounded_rectangle((146, 624, 294, 638), radius=7, fill="#C6CFDB")
    return image


def make_sprite_scene(pattern_sprite, phone_sprite):
    """Render two textured planes through one perspective camera."""
    viewport = (1200 * SCENE_SCALE, 500 * SCENE_SCALE)
    canvas = Image.new("RGBA", viewport, (0, 0, 0, 0))
    camera = PerspectiveCamera(
        position=(-1.8, 1.15, 8.6),
        target=(0.0, 0.02, 0.05),
        focal_length=1120 * SCENE_SCALE,
        viewport=viewport,
    )

    # Both centers lie on one optical axis. The small target is farther away;
    # the side camera reveals it through parallax instead of an artificial offset.
    pattern_corners = plane_corners(
        center=(0.0, 0.02, -2.55),
        width=1.08,
        height=0.24,
        yaw_degrees=6.0,
        roll_degrees=-1.0,
    )
    phone_corners = plane_corners(
        center=(0.0, 0.02, 0.82),
        width=0.92,
        height=1.52,
        yaw_degrees=6.0,
        roll_degrees=1.0,
    )

    render_sprite_plane(
        canvas,
        pattern_sprite,
        camera,
        pattern_corners,
        (5 * SCENE_SCALE, 7 * SCENE_SCALE),
        42,
        7 * SCENE_SCALE,
    )
    render_sprite_plane(
        canvas,
        phone_sprite,
        camera,
        phone_corners,
        (8 * SCENE_SCALE, 10 * SCENE_SCALE),
        62,
        9 * SCENE_SCALE,
    )
    bounds = canvas.getchannel("A").getbbox()
    if bounds is None:
        return canvas
    padding = 28 * SCENE_SCALE
    left = max(0, bounds[0] - padding)
    top = max(0, bounds[1] - padding)
    right = min(canvas.width, bounds[2] + padding)
    bottom = min(canvas.height, bounds[3] + padding)
    return canvas.crop((left, top, right, bottom))


flat_pattern = make_flat_pattern()
flat_phone = make_flat_phone()
flat_pattern.save(ASSETS / "optical_pattern_flat.png")
flat_phone.save(ASSETS / "camera_phone_flat.png")

# Individual projected assets are retained for easy inspection; the combined
# illustration below is rendered independently from one shared 3D camera.
preview_camera = PerspectiveCamera((-1.8, 1.15, 8.6), (0.0, 0.02, 0.05), 1120, (1200, 500))
pattern_projected = project(
    flat_pattern,
    (1200, 500),
    [preview_camera.project(point) for point in plane_corners((0.0, 0.02, -2.55), 1.08, 0.24, 6.0, -1.0)],
)
phone_projected = project(
    flat_phone,
    (1200, 500),
    [preview_camera.project(point) for point in plane_corners((0.0, 0.02, 0.82), 0.92, 1.52, 6.0, 1.0)],
)

with_shadow(pattern_projected, offset=(5, 6), radius=7, opacity=55).save(
    ASSETS / "optical_pattern_perspective.png"
)
with_shadow(phone_projected, offset=(7, 9), radius=9, opacity=75).save(
    ASSETS / "camera_phone_perspective.png"
)
make_sprite_scene(flat_pattern, flat_phone).save(ASSETS / "optical_capture_3d.png")
