import cv2
import numpy as np

detector = cv2.SIFT_create(contrastThreshold=0.02)
matcher = cv2.BFMatcher(cv2.NORM_L2)

MIN_MATCHES = 12
MIN_INLIERS = 12
MIN_FLOW_INLIERS = 8
RATIO = 0.75
MIN_AREA_ABS = 1200
MAX_AREA_RATIO = 1.05
BOUNDS_MARGIN = 0.25
MIN_SIDE_RATIO = 0.1

TILT_SIDES = ("top", "bottom", "left", "right")
TILT_STRENGTHS = (0.3, 0.5)


def tilted_view(gray, side, strength):
    h, w = gray.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()
    margin_x = w * strength / 2
    margin_y = h * strength / 2
    if side == "top":
        dst[0][0] += margin_x
        dst[1][0] -= margin_x
    elif side == "bottom":
        dst[2][0] -= margin_x
        dst[3][0] += margin_x
    elif side == "left":
        dst[0][1] += margin_y
        dst[3][1] -= margin_y
    else:
        dst[1][1] += margin_y
        dst[2][1] -= margin_y

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, H, (w, h))
    return warped, np.linalg.inv(H)


def build_reference(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    all_pts = []
    all_des = []

    kp, des = detector.detectAndCompute(gray, None)
    all_pts.append(np.float32([p.pt for p in kp]))
    all_des.append(des)

    # синтетические ракурсы эталона (ASIFT-подобный приём) для устойчивости к углу съёмки
    for side in TILT_SIDES:
        for strength in TILT_STRENGTHS:
            warped, H_inv = tilted_view(gray, side, strength)
            kp_w, des_w = detector.detectAndCompute(warped, None)
            if des_w is None or len(kp_w) == 0:
                continue
            pts_w = np.float32([p.pt for p in kp_w]).reshape(-1, 1, 2)
            pts_back = cv2.perspectiveTransform(pts_w, H_inv).reshape(-1, 2)
            all_pts.append(pts_back)
            all_des.append(des_w)

    ref_pts = np.vstack(all_pts)
    ref_des = np.vstack(all_des)
    return ref_pts, ref_des, corners


def box_is_valid(box, w, h):
    pts = box.reshape(-1, 2)
    if not cv2.isContourConvex(np.int32(box)):
        return False

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    if x_min < -BOUNDS_MARGIN * w or x_max > (1 + BOUNDS_MARGIN) * w:
        return False
    if y_min < -BOUNDS_MARGIN * h or y_max > (1 + BOUNDS_MARGIN) * h:
        return False

    area = cv2.contourArea(pts.astype(np.float32))
    if area < MIN_AREA_ABS or area > MAX_AREA_RATIO * w * h:
        return False

    # отсекаем вырожденные "иголки" - оптический поток иногда сносит точки
    # в сторону, и гомография по ним растягивает рамку в тонкий клин
    rect_w, rect_h = cv2.minAreaRect(pts.astype(np.float32))[1]
    if min(rect_w, rect_h) < MIN_SIDE_RATIO * max(rect_w, rect_h):
        return False

    return True


def match_reference(ref_pts, ref_des, ref_corners, gray, w, h):
    kp, des = detector.detectAndCompute(gray, None)
    if des is None or len(kp) < 2:
        return None, 0

    pairs = matcher.knnMatch(ref_des, des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_MATCHES:
        return None, len(good)

    src = ref_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask.sum() < MIN_INLIERS:
        return None, len(good)

    inliers = mask.ravel() == 1
    box = cv2.perspectiveTransform(ref_corners, H)
    if not box_is_valid(box, w, h):
        return None, len(good)

    return (box, src[inliers], dst[inliers]), len(good)


def track_flow(ref_corners, prev_gray, gray, prev_ref_pts, prev_cur_pts, w, h):
    if prev_cur_pts is None or len(prev_cur_pts) < MIN_FLOW_INLIERS:
        return None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_cur_pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    status = status.reshape(-1).astype(bool)
    if status.sum() < MIN_FLOW_INLIERS:
        return None

    src = prev_ref_pts[status]
    dst = next_pts[status]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask.sum() < MIN_FLOW_INLIERS:
        return None

    inliers = mask.ravel() == 1
    box = cv2.perspectiveTransform(ref_corners, H)
    if not box_is_valid(box, w, h):
        return None

    return box, src[inliers], dst[inliers]


def find_object(ref_pts, ref_des, ref_corners, frame, state):
    # state - словарь prev_gray/ref_pts/cur_pts с прошлого кадра, main.py ведёт его по циклу
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result, n_good = match_reference(ref_pts, ref_des, ref_corners, gray, w, h)
    mode = "homography"

    if result is None and state["prev_gray"] is not None:
        result = track_flow(ref_corners, state["prev_gray"], gray, state["ref_pts"], state["cur_pts"], w, h)
        mode = "flow"

    state["prev_gray"] = gray

    if result is None:
        state["ref_pts"] = None
        state["cur_pts"] = None
        return None, n_good, None, None

    box, new_ref_pts, new_cur_pts = result
    state["ref_pts"] = new_ref_pts
    state["cur_pts"] = new_cur_pts
    return box, n_good, new_cur_pts, mode
