"""Copy a LeRobot dataset with all videos resized to a target resolution.

Uses ffmpeg via multiprocessing to resize all videos in parallel across CPU cores.
ffmpeg processes frames one at a time so memory usage is constant regardless of
video length.

Usage (from repo root):
    uv run python lerobot_baseline/resize_dataset.py

Source: recordings/redcubes_bluecup   (480x640)
Dest:   recordings_96x96/redcubes_bluecup  (96x96)

frame_cache/ is skipped -- it is resolution-specific and not used by LeRobot.
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
from pathlib import Path

import av
from PIL import Image

SRC = Path("recordings/redcubes_bluecup")
DST = Path("recordings_96x96/redcubes_bluecup")
TARGET_H = 96
TARGET_W = 96


def resize_video(args: tuple[Path, Path]) -> str:
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as in_container:
        in_stream = in_container.streams.video[0]
        fps = float(in_stream.average_rate)
        with av.open(str(dst), "w") as out_container:
            out_stream = out_container.add_stream("libx264", rate=int(fps))
            out_stream.width = TARGET_W
            out_stream.height = TARGET_H
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "18", "preset": "fast"}
            for frame in in_container.decode(video=0):
                resized = frame.to_image().resize((TARGET_W, TARGET_H), Image.BILINEAR)
                out_frame = av.VideoFrame.from_image(resized)
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
            for packet in out_stream.encode():
                out_container.mux(packet)
    return str(src.name)


def update_info_json(dst: Path) -> None:
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feature in info.get("features", {}).values():
        if feature.get("dtype") != "video":
            continue
        feature["shape"] = [TARGET_H, TARGET_W, feature["shape"][2]]
        vi = feature.get("info", {})
        vi["video.height"] = TARGET_H
        vi["video.width"] = TARGET_W
        vi["video.codec"] = "h264"
        vi["video.crf"] = 18
        vi.pop("video.preset", None)
    info_path.write_text(json.dumps(info, indent=4))


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"Destination already exists: {DST}  (delete it first)")

    DST.mkdir(parents=True)
    for item in SRC.iterdir():
        if item.name in {"videos", "frame_cache"}:
            continue
        dst_item = DST / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    mp4_files = sorted((SRC / "videos").rglob("*.mp4"))
    jobs = [(src, DST / "videos" / src.relative_to(SRC / "videos")) for src in mp4_files]

    num_workers = min(len(jobs), multiprocessing.cpu_count())
    print(f"Re-encoding {len(mp4_files)} videos to {TARGET_W}x{TARGET_H} using {num_workers} workers ...")
    with multiprocessing.Pool(num_workers) as pool:
        for i, name in enumerate(pool.imap_unordered(resize_video, jobs), 1):
            print(f"  [{i}/{len(jobs)}] done: {name}")

    update_info_json(DST)
    print(f"\nDone. Resized dataset at: {DST}")


if __name__ == "__main__":
    main()
