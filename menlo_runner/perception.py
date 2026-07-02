# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HFOV_HALF_DEG = 30.0
MIN_BLOB_AREA = 200
DEFAULT_COMPRESSED_MAX_WIDTH = 800
DEFAULT_COMPRESSED_JPEG_QUALITY = 70


@dataclass(frozen=True)
class ColorDetection:
    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    depth_score: float | None = None


def _cv2_np() -> tuple[Any, Any]:
    import cv2
    import numpy as np

    return cv2, np


def color_ranges() -> dict[str, list[tuple[Any, Any]]]:
    _, np = _cv2_np()
    return {
        "red": [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([160, 50, 50]), np.array([180, 255, 255])),
        ],
        "green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
        "blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        "yellow": [(np.array([20, 50, 50]), np.array([35, 255, 255]))],
    }


def decode_jpeg(jpeg_bytes: bytes) -> Any:
    cv2, np = _cv2_np()
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG bytes.")
    return image


def compress_jpeg(
    jpeg_bytes: bytes,
    *,
    max_width: int = DEFAULT_COMPRESSED_MAX_WIDTH,
    quality: int = DEFAULT_COMPRESSED_JPEG_QUALITY,
) -> bytes:
    """Resize/re-encode a camera JPEG before optional VLM calls.

    Use a higher max_width/quality for reading small signs, and lower values
    when the VLM only needs coarse scene context.
    """
    if max_width <= 0:
        raise ValueError("max_width must be greater than zero.")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100.")

    cv2, _ = _cv2_np()
    image = decode_jpeg(jpeg_bytes)
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, max(1, int(height * scale))))

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode compressed JPEG.")
    return encoded.tobytes()


def detect_color_blobs(
    jpeg_bytes: bytes,
    *,
    min_area: int = MIN_BLOB_AREA,
    depth_map: Any | None = None,
) -> list[ColorDetection]:
    cv2, np = _cv2_np()
    image = decode_jpeg(jpeg_bytes)
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections: list[ColorDetection] = []

    for color, ranges in color_ranges().items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, low, high)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
            angle = (cx - width / 2) / (width / 2) * HFOV_HALF_DEG
            depth_score = None
            if depth_map is not None:
                dy = min(max(cy, 0), depth_map.shape[0] - 1)
                dx = min(max(cx, 0), depth_map.shape[1] - 1)
                depth_score = float(depth_map[dy, dx])
            detections.append(
                ColorDetection(
                    color=color,
                    angle_deg=round(angle, 1),
                    blob_area=int(area),
                    centroid=(cx, cy),
                    bbox=(x, y, bbox_width, bbox_height),
                    depth_score=depth_score,
                )
            )

    return sorted(detections, key=lambda item: item.blob_area, reverse=True)


def perceive_jpeg(jpeg_bytes: bytes, *, min_area: int = MIN_BLOB_AREA) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for detection in detect_color_blobs(jpeg_bytes, min_area=min_area):
        if detection.color in result:
            continue
        result[detection.color] = {
            "angle_deg": detection.angle_deg,
            "blob_area": detection.blob_area,
        }
    return result


async def perceive(ctx: Any, *, min_area: int = MIN_BLOB_AREA) -> dict[str, dict[str, float | int]]:
    """Return the Workshop 2 perception format: {color: {angle_deg, blob_area}}."""
    jpeg = await ctx.get_vision("pov")
    return perceive_jpeg(jpeg, min_area=min_area)


def annotate_detections(jpeg_bytes: bytes, detections: list[ColorDetection] | None = None) -> bytes:
    cv2, _ = _cv2_np()
    image = decode_jpeg(jpeg_bytes)
    detections = detections if detections is not None else detect_color_blobs(jpeg_bytes)
    bgr = {
        "red": (0, 0, 255),
        "green": (0, 200, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 200, 200),
    }
    for item in detections:
        x, y, width, height = item.bbox
        color = bgr.get(item.color, (255, 255, 255))
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            image,
            f"{item.color} {item.angle_deg:+.1f}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode annotated image.")
    return encoded.tobytes()


def estimate_depth_map(jpeg_bytes: bytes, depth_pipe: Any) -> Any:
    cv2, np = _cv2_np()
    from PIL import Image

    image = decode_jpeg(jpeg_bytes)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.array(depth_pipe(pil_image)["depth"])



