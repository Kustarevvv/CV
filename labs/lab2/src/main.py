import os
import cv2
import numpy as np

from tracker import build_reference, find_object

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "..", "results", "videos")


def process_video(path, out_path, label):
    cap = cv2.VideoCapture(path)
    ok, first = cap.read()
    if not ok:
        print("не удалось открыть видео:", path)
        return [], []

    ref_kp, ref_des, ref_corners = build_reference(first)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(first)

    matches_count = []
    modes = []
    state = {"prev_gray": None, "ref_pts": None, "cur_pts": None}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        box, n_good, pts, mode = find_object(ref_kp, ref_des, ref_corners, frame, state)
        matches_count.append(n_good)
        modes.append(mode)

        if mode is not None:
            for x, y in pts.reshape(-1, 2):
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
            cv2.polylines(frame, [np.int32(box)], True, (0, 255, 0), 3)
            corner = tuple(np.int32(box[0, 0]))
            cv2.putText(frame, label, corner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "object lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    return matches_count, modes


VIDEOS = [
    ("video_Jack_London", "video_jl.MOV", "Jack London"),
    ("test_videos", "mona-lisa.avi", "Mona Lisa"),
    ("test_videos", "mona-lisa-blur.avi", "Mona Lisa"),
    ("test_videos", "mona-lisa-blur-extra-credit.avi", "Mona Lisa"),
]


def main():
    for folder, name, label in VIDEOS:
        video_in = os.path.join(BASE, "..", folder, name)
        out_name = os.path.splitext(name)[0] + "_tracked.mp4"
        video_out = os.path.join(OUT_DIR, out_name)

        matches, modes = process_video(video_in, video_out, label)

        total = len(modes)
        n_homography = modes.count("homography")
        n_flow = modes.count("flow")
        n_lost = modes.count(None)
        print(name)
        print(f"  кадров всего: {total}")
        print(f"  гомография: {n_homography} ({100 * n_homography / total:.1f}%)")
        print(f"  оптический поток: {n_flow} ({100 * n_flow / total:.1f}%)")
        print(f"  потерян: {n_lost} ({100 * n_lost / total:.1f}%)")
        print(f"  среднее число совпадений: {sum(matches) / total:.1f}")


if __name__ == "__main__":
    main()
