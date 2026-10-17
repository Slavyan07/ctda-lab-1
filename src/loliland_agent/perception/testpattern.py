"""An explicitly artificial visual test scene. No Minecraft training data or game API."""

from __future__ import annotations

from typing import Literal

from PIL import Image, ImageDraw

from loliland_agent.models import Landmark, VisualState

COLORS = {
    "anchor_a": (251, 0, 251),
    "anchor_b": (0, 251, 251),
    "anchor_c": (10, 30, 251),
    "player": (251, 210, 0),
    "waypoint": (20, 220, 50),
    "button": (180, 70, 240),
    "intervention": (250, 30, 30),
}


def render_test_frame(
    width: int = 800,
    height: int = 450,
    player_x: float = 0.2,
    inventory: bool = False,
    selected: int = 0,
    intervention: bool = False,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (26, 42, 52))
    draw = ImageDraw.Draw(image)

    def rect(box: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
        draw.rectangle(
            tuple(int(value * (width if i % 2 == 0 else height)) for i, value in enumerate(box)), fill=color
        )

    rect((0.02, 0.02, 0.06, 0.06), COLORS["anchor_a"])
    rect((0.94, 0.02, 0.98, 0.06), COLORS["anchor_b"])
    rect((0.02, 0.94, 0.06, 0.98), COLORS["anchor_c"])
    draw.text(
        (int(width * 0.1), int(height * 0.04)), "EXTERNAL AGENT TEST WINDOW - NOT MINECRAFT", fill="white"
    )
    draw.text(
        (int(width * 0.1), int(height * 0.12)),
        "W: advance / E: inventory / 1-9: hotbar / F8: stop",
        fill="white",
    )
    rect((0.78, 0.46, 0.82, 0.54), COLORS["waypoint"])
    rect((player_x - 0.015, 0.475, player_x + 0.015, 0.525), COLORS["player"])
    for slot in range(9):
        x = 0.25 + slot * 0.055
        rect((x, 0.85, x + 0.04, 0.9), (100, 100, 100) if slot != selected else (240, 240, 240))
    if inventory:
        rect((0.25, 0.25, 0.75, 0.75), (50, 55, 60))
        rect((0.45, 0.45, 0.55, 0.55), COLORS["button"])
        draw.text((int(width * 0.3), int(height * 0.3)), "INVENTORY / TEST BUTTON", fill="white")
    if intervention:
        rect((0.2, 0.2, 0.8, 0.8), COLORS["intervention"])
        draw.text((int(width * 0.3), int(height * 0.4)), "CAPTCHA - HUMAN REQUIRED", fill="white")
    return image


def detect_test_frame(image: Image.Image) -> VisualState | None:
    image = image.convert("RGB").resize((320, 180), Image.Resampling.NEAREST)
    pixels = image.load()
    assert pixels is not None

    def matches(x: float, y: float, name: str) -> bool:
        pixel = image.getpixel((int(x * 320), int(y * 180)))
        return isinstance(pixel, tuple) and all(abs(int(pixel[i]) - COLORS[name][i]) < 12 for i in range(3))

    if not all(
        matches(x, y, name)
        for x, y, name in [(0.04, 0.04, "anchor_a"), (0.96, 0.04, "anchor_b"), (0.04, 0.96, "anchor_c")]
    ):
        return None
    if matches(0.5, 0.5, "intervention"):
        return VisualState(
            scene="verification",
            confidence=1.0,
            human_required=True,
            visible_text=("CAPTCHA - HUMAN REQUIRED",),
            summary="Искусственное окно: требуется человек",
        )
    landmarks: list[Landmark] = []
    kinds: list[tuple[str, Literal["player", "waypoint", "gui_button"]]] = [
        ("player", "player"),
        ("waypoint", "waypoint"),
        ("button", "gui_button"),
    ]
    for name, kind in kinds:
        points = []
        for y in range(30, 150):
            for x in range(30, 290):
                value = pixels[x, y]
                if isinstance(value, tuple) and all(abs(value[i] - COLORS[name][i]) < 12 for i in range(3)):
                    points.append((x, y))
        if points:
            landmarks.append(
                Landmark(
                    name=name,
                    kind=kind,
                    x=sum(p[0] for p in points) / len(points) / 320,
                    y=sum(p[1] for p in points) / len(points) / 180,
                    confidence=1.0,
                )
            )
    inventory = any(item.kind == "gui_button" for item in landmarks)
    selected = next(
        (
            slot
            for slot in range(9)
            if image.getpixel((int((0.27 + slot * 0.055) * 320), 157)) == (240, 240, 240)
        ),
        None,
    )
    return VisualState(
        scene="inventory" if inventory else "test_window",
        confidence=1.0,
        human_required=False,
        selected_slot=selected,
        landmarks=tuple(landmarks),
        summary="Искусственное тестовое окно; распознано локально по пикселям",
    )
