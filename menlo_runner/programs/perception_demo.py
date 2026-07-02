# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

from pathlib import Path

from menlo_runner.perception import (
    annotate_detections,
    detect_color_blobs,
)


async def run(ctx) -> None:
    jpeg = await ctx.get_vision("pov")
    detections = detect_color_blobs(jpeg)
    print("Color blob detections:")
    if not detections:
        print("  none")
    for item in detections:
        print(
            f"  {item.color}: angle={item.angle_deg:+.1f} deg "
            f"area={item.blob_area}px^2 centroid={item.centroid}"
        )

    out = Path("outputs/perception-annotated.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(annotate_detections(jpeg, detections))
    print(f"Saved annotated image: {out}")

