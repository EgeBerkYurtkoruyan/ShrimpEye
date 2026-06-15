"""
Tracking strategy
=================
ACQUIRE uses background subtraction, which is only valid while the robot is
standing still. Once an object is locked we LEARN its colour in HSV and from
then on find it by colour segmentation (cv2.inRange), which does not depend on
any stored background, so it survives the camera moving.

On top of the colour lock, three gates keep the lock on the RIGHT object even
when a similar-looking thing appears elsewhere:

  POSITION GATE (always on):
    A candidate blob is only accepted if it is close to where the object is
    predicted to be this frame. The gate is centred on a short-term velocity
    prediction, so it travels WITH a moving object; a stationary same-colour
    background patch falls outside the gate the moment the object moves. On a
    miss the gate widens only GRADUALLY over successive frames, so the lock can
    never instantly jump to a totally different location.

  SIZE GATE:
    A candidate whose diameter is very different from the tracked diameter is
    rejected, so a large same-colour wall or a tiny same-colour speck cannot
    steal the lock.

  COLOUR GATE:
    The learned hue band (or brightness band for white/grey/black objects).

Distance control
================
The object's DIAMETER (min-enclosing-circle) is the distance measure. Diameter
is linear with apparent size, so it tracks distance far more smoothly than area
(which changes with the square). At lock time we record target_diam; while
following we drive forward when the diameter is smaller than target (object got
farther) and back up when it is larger (object got closer), until the diameter
matches the locked value. Turn and drive speeds are PROPORTIONAL to the error,
so motion is gentle near the set-point and firmer when far from it.

State machine
=============
  ACQUIRE  :  robot still; waits for a new object to appear and hold still.
  TRACK    :  follows the locked object by colour; keeps the locked diameter.
  LOST     :  object left the frame (or nothing showed up for a while);
              robot stops, re-captures a fresh background, returns to ACQUIRE.

Startup procedure
=================
  1. Run with the object NOT in view; wait for "Background captured."
  2. Place the object in front of the camera, hold still ~1.2 s until "LOCKED".
  3. Move the object; the robot follows and keeps the locked distance.
  4. Remove the object; the robot stops and waits for the next one.
  5. ESC in the debug window or Ctrl-C in the terminal to quit.
"""

import time
from collections import deque

import cv2
import numpy as np
from picamera2 import Picamera2
import picar_4wd as fc


# ---------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------

FRAME_W = 160
FRAME_H = 120
IMAGE_CENTER_X = FRAME_W // 2

# Background capture (used only while the robot is still, in ACQUIRE)
BACKGROUND_SECONDS = 1.5   # seconds to sample the empty scene

# Acquisition - locking on to a new object
LOCK_SECONDS            = 1.2   # object must stay still for this many seconds
STABLE_RADIUS_PX        = 12    # max pixel drift while "holding still"
DIFF_THRESHOLD          = 28    # pixel brightness change to count as foreground
MIN_AREA                = 200   # smallest candidate blob (px^2)
MAX_AREA                = 9000  # largest candidate blob (px^2)
ACQUIRE_TIMEOUT_SECONDS = 5.0   # empty view this long -> refresh background

# Dynamic colour lock (learned from the locked object)
SAT_MIN_FOR_HUE       = 40    # below this saturation a pixel's hue is unstable
CHROMATIC_PIXEL_RATIO = 0.25  # fraction of ROI that must be saturated to be "colourful"
HUE_TOLERANCE         = 12    # +/- hue band around the dominant hue (OpenCV 0..179)
CHROMA_SAT_LOWER      = 55    # min saturation accepted as the object (colourful case)
CHROMA_VAL_LOWER      = 45    # min value accepted as the object (colourful case)
ACHROMA_SAT_MAX       = 65    # max saturation for the white/grey/black fallback
ACHROMA_VAL_TOL       = 45    # +/- brightness band for the white/grey/black fallback

# Colour blob acceptance
COLOR_MIN_AREA = 120   # smallest colour blob (px^2) considered at all

# POSITION GATE (ask 1: only follow blobs near the predicted position)
BASE_GATE_PX   = 45    # acceptance radius while solidly tracking
GATE_GROWTH_PX = 16    # how much the radius grows per consecutive missed frame
MAX_GATE_PX    = 110   # cap so the gate never spans the whole frame
VEL_ALPHA      = 0.40  # smoothing for the velocity (motion) predictor

# SIZE GATE (reject same-colour blobs of the wrong size)
SIZE_GATE_LO = 0.50    # accept diameter >= this * tracked diameter
SIZE_GATE_HI = 1.90    # accept diameter <= this * tracked diameter

MAX_LOST_FRAMES = 12   # consecutive misses before declaring LOST

# EMA smoothing on the detected centre and diameter
EMA_ALPHA_POS  = 0.35
EMA_ALPHA_DIAM = 0.30

# Distance control (target is the object's locked DIAMETER)
DEADZONE_RATIO    = 0.15   # +/- 15% of target diameter = "close enough", hold
MIN_DIAM_DEADZONE = 4.0    # absolute floor for the dead-zone (anti-jitter, px)
SAFETY_RATIO      = 1.80   # diameter > target * this -> back up hard (too close)

# Steering / centring
CENTER_DEADZONE = 12    # px error ignored to prevent jitter

# Proportional speeds (smoothness)
TURN_SPEED_MIN  = 12
TURN_SPEED_MAX  = 28
TURN_GAIN       = 0.45   # added turn speed per px of centring error past the deadzone
DRIVE_SPEED_MIN = 12
DRIVE_SPEED_MAX = 26
DRIVE_GAIN      = 32     # added drive speed per unit of normalised diameter error

SHOW_DEBUG = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_backward(speed):
    """Drive backward, tolerating either picar_4wd naming convention."""
    for name in ("backward", "back"):
        fn = getattr(fc, name, None)
        if callable(fn):
            fn(speed)
            return
    fc.stop()   # library exposes neither -> at least don't crash


# ---------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    return picam2


def capture_small_bgr(picam2):
    """Grab a frame from the camera and resize to the working resolution."""
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(frame_bgr, (FRAME_W, FRAME_H))


def make_gray(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


# ---------------------------------------------------------------------
# Background / acquisition (static-camera only)
# ---------------------------------------------------------------------

def capture_background(picam2):
    print("Capturing background - keep the object OUT of view...")
    frames = []
    t0 = time.time()
    while time.time() - t0 < BACKGROUND_SECONDS:
        frames.append(make_gray(capture_small_bgr(picam2)))
        time.sleep(0.03)
    bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    print("Background captured. Place the object in view and hold it still to lock.")
    return bg


def find_new_object(frame_gray, background_gray):
    """
    Frame-difference detector (only used while the robot is standing still).
    Returns (diff_mask, candidate_dict or None).
    """
    diff  = cv2.absdiff(frame_gray, background_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    valid = [
        (cv2.contourArea(c), c)
        for c in contours
        if MIN_AREA <= cv2.contourArea(c) <= MAX_AREA
    ]
    if not valid:
        return mask, None

    area, contour = max(valid, key=lambda t: t[0])
    x, y, w, h   = cv2.boundingRect(contour)
    return mask, {
        "x": x, "y": y, "w": w, "h": h,
        "cx": x + w // 2, "cy": y + h // 2,
        "area": area,
    }


def stable_enough(history):
    """True when the candidate has barely moved for LOCK_SECONDS."""
    if len(history) < 5:
        return False
    if history[-1][0] - history[0][0] < LOCK_SECONDS:
        return False
    centers    = np.array([(h[1], h[2]) for h in history], dtype=np.float32)
    mean_c     = centers.mean(axis=0)
    max_radius = np.max(np.linalg.norm(centers - mean_c, axis=1))
    return max_radius <= STABLE_RADIUS_PX


# ---------------------------------------------------------------------
# Dynamic colour lock
# ---------------------------------------------------------------------

def learn_color(frame_bgr, bbox):
    """
    Learn an HSV colour signature from the centre of the locked object's box.
    Returns a 'colour' dict, or None if the region is empty.

      chromatic = True  : coloured object -> lock the dominant HUE (most stable
                          channel under lighting / motion) with a tolerance band.
      chromatic = False : white / grey / black object -> lock a BRIGHTNESS band.
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # Sample only the central 60% of the box so we get the object, not the
    # background that often clings to the box edges.
    cx0 = clamp(int(x + w * 0.2), 0, FRAME_W - 1)
    cy0 = clamp(int(y + h * 0.2), 0, FRAME_H - 1)
    cx1 = clamp(int(x + w * 0.8), cx0 + 1, FRAME_W)
    cy1 = clamp(int(y + h * 0.8), cy0 + 1, FRAME_H)

    roi = frame_bgr[cy0:cy1, cx0:cx1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].ravel()
    S = hsv[:, :, 1].ravel()
    V = hsv[:, :, 2].ravel()

    sat_pixels = S >= SAT_MIN_FOR_HUE
    chromatic  = bool(S.size) and (sat_pixels.mean() >= CHROMATIC_PIXEL_RATIO)

    if chromatic:
        hh = H[sat_pixels]
        if hh.size > 0:
            # Dominant hue via a circularly-smoothed histogram peak (handles
            # red, which wraps around the 0/179 boundary).
            hist = np.bincount(hh, minlength=180).astype(np.float32)
            ext  = np.concatenate([hist[-2:], hist, hist[:2]])
            ker  = np.array([1, 2, 3, 2, 1], dtype=np.float32)
            ker /= ker.sum()
            sm   = np.convolve(ext, ker, mode="same")[2:-2]
            peak = int(np.argmax(sm))
            return {
                "chromatic": True,
                "hue":       peak,
                "hue_tol":   HUE_TOLERANCE,
                "s_lo":      CHROMA_SAT_LOWER,
                "v_lo":      CHROMA_VAL_LOWER,
            }

    # Achromatic fallback: lock brightness, accept only low saturation.
    v_med = float(np.median(V)) if V.size else 128.0
    return {
        "chromatic": False,
        "v_lo":      clamp(int(v_med - ACHROMA_VAL_TOL), 0, 255),
        "v_hi":      clamp(int(v_med + ACHROMA_VAL_TOL), 0, 255),
        "s_hi":      ACHROMA_SAT_MAX,
    }


def describe_color(color):
    """Short human-readable label for the locked colour, for the terminal."""
    if not color["chromatic"]:
        return "achromatic (V %d-%d)" % (color["v_lo"], color["v_hi"])
    h = color["hue"]
    names = [
        (0,   "red"), (10, "orange"), (20, "yellow"), (35, "green"),
        (85,  "cyan"), (100, "blue"), (130, "purple"), (160, "pink/red"),
    ]
    label = "red"
    for thr, name in names:
        if h >= thr:
            label = name
    return "%s (hue %d)" % (label, h)


def color_mask(frame_bgr, color):
    """Binary mask of the locked colour over the whole frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    if color["chromatic"]:
        h    = color["hue"]
        tol  = color["hue_tol"]
        s_lo = color["s_lo"]
        v_lo = color["v_lo"]
        lo   = h - tol
        hi   = h + tol
        if lo < 0:                         # wraps past 0 (e.g. red)
            m1 = cv2.inRange(hsv, (0, s_lo, v_lo),        (hi, 255, 255))
            m2 = cv2.inRange(hsv, (180 + lo, s_lo, v_lo), (179, 255, 255))
            mask = cv2.bitwise_or(m1, m2)
        elif hi > 179:                     # wraps past 179 (e.g. red)
            m1 = cv2.inRange(hsv, (lo, s_lo, v_lo),  (179, 255, 255))
            m2 = cv2.inRange(hsv, (0, s_lo, v_lo),   (hi - 180, 255, 255))
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, (lo, s_lo, v_lo), (hi, 255, 255))
    else:
        mask = cv2.inRange(hsv, (0, 0, color["v_lo"]),
                                (179, color["s_hi"], color["v_hi"]))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


# ---------------------------------------------------------------------
# Colour tracker (with position gate, size gate, velocity prediction)
# ---------------------------------------------------------------------

def create_tracker(frame_bgr, bbox):
    color = learn_color(frame_bgr, bbox)
    if color is None:
        return None

    cx0 = float(bbox.get("cx", bbox["x"] + bbox["w"] / 2))
    cy0 = float(bbox.get("cy", bbox["y"] + bbox["h"] / 2))
    diam0 = float(max(bbox["w"], bbox["h"]))   # rough first guess

    tracker = {
        "color":       color,
        "lost_count":  0,
        "ema_cx":      cx0,
        "ema_cy":      cy0,
        "vel_cx":      0.0,
        "vel_cy":      0.0,
        "ema_diam":    diam0,
        "target_diam": diam0,
    }

    # Refine the centre/diameter using the actual colour blob right now, so the
    # set-point is measured exactly the way it will be measured while tracking.
    detection, _ = track_color(frame_bgr, tracker)
    if detection is not None:
        tracker["ema_cx"]     = detection["cx"]
        tracker["ema_cy"]     = detection["cy"]
        tracker["ema_diam"]   = detection["diam"]
        tracker["target_diam"] = detection["diam"]
    tracker["vel_cx"] = 0.0
    tracker["vel_cy"] = 0.0
    tracker["lost_count"] = 0
    return tracker


def track_color(frame_bgr, tracker):
    """
    Find the locked colour, then choose the blob that passes the position gate
    and the size gate and is closest to the predicted position.
    Returns (detection_dict or None, colour_mask).
    """
    mask = color_mask(frame_bgr, tracker["color"])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        if cv2.contourArea(c) < COLOR_MIN_AREA:
            continue
        (ccx, ccy), r = cv2.minEnclosingCircle(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        candidates.append({
            "cx": float(ccx), "cy": float(ccy),
            "diam": float(2.0 * r),
            "x": bx, "y": by, "w": bw, "h": bh,
        })

    if not candidates:
        tracker["lost_count"] += 1
        return None, mask

    # Predict where the object should be this frame (last centre + velocity).
    pred_x = clamp(tracker["ema_cx"] + tracker["vel_cx"], 0, FRAME_W)
    pred_y = clamp(tracker["ema_cy"] + tracker["vel_cy"], 0, FRAME_H)

    # POSITION GATE: radius grows gradually with consecutive misses, but is
    # capped so the lock can never jump to a totally different location.
    gate = min(MAX_GATE_PX, BASE_GATE_PX + tracker["lost_count"] * GATE_GROWTH_PX)

    in_gate = [
        cand for cand in candidates
        if np.hypot(cand["cx"] - pred_x, cand["cy"] - pred_y) <= gate
    ]
    if not in_gate:
        tracker["lost_count"] += 1
        return None, mask

    # SIZE GATE: reject blobs whose diameter is far from the tracked diameter.
    lo = SIZE_GATE_LO * tracker["ema_diam"]
    hi = SIZE_GATE_HI * tracker["ema_diam"]
    sized = [cand for cand in in_gate if lo <= cand["diam"] <= hi]
    if not sized:
        sized = in_gate   # fall back rather than lose a borderline-size object

    # Among survivors, pick the one closest to the predicted position.
    chosen = min(
        sized,
        key=lambda cand: np.hypot(cand["cx"] - pred_x, cand["cy"] - pred_y),
    )

    # Update velocity (before the position EMA, using the raw measurement).
    meas_vx = chosen["cx"] - tracker["ema_cx"]
    meas_vy = chosen["cy"] - tracker["ema_cy"]
    tracker["vel_cx"] = (1 - VEL_ALPHA) * tracker["vel_cx"] + VEL_ALPHA * meas_vx
    tracker["vel_cy"] = (1 - VEL_ALPHA) * tracker["vel_cy"] + VEL_ALPHA * meas_vy

    # Smooth the centre and diameter.
    tracker["ema_cx"]   = (1 - EMA_ALPHA_POS)  * tracker["ema_cx"]   + EMA_ALPHA_POS  * chosen["cx"]
    tracker["ema_cy"]   = (1 - EMA_ALPHA_POS)  * tracker["ema_cy"]   + EMA_ALPHA_POS  * chosen["cy"]
    tracker["ema_diam"] = (1 - EMA_ALPHA_DIAM) * tracker["ema_diam"] + EMA_ALPHA_DIAM * chosen["diam"]
    tracker["lost_count"] = 0

    return {
        "x":    chosen["x"],
        "y":    chosen["y"],
        "w":    chosen["w"],
        "h":    chosen["h"],
        "cx":   tracker["ema_cx"],     # smoothed
        "cy":   tracker["ema_cy"],     # smoothed
        "diam": tracker["ema_diam"],   # smoothed
    }, mask


# ---------------------------------------------------------------------
# Motion control (diameter-based distance, proportional speeds)
# ---------------------------------------------------------------------

def control_robot(detection, target_diam):
    """
    Priority order:
      1. No detection           :  stop
      2. Far too close (safety) :  back up regardless of centring
      3. Off-centre             :  turn to re-centre (speed proportional to error)
      4. Too far                :  drive forward  (speed proportional to error)
      5. Too close              :  drive backward (speed proportional to error)
      6. Inside dead-zone       :  hold still
    """
    if detection is None:
        fc.stop()
        return "STOP: target lost"

    error_x  = detection["cx"] - IMAGE_CENTER_X
    diam     = detection["diam"]
    deadzone = max(MIN_DIAM_DEADZONE, target_diam * DEADZONE_RATIO)

    # 1. Safety: object alarmingly close -> back away immediately.
    if diam > target_diam * SAFETY_RATIO:
        safe_backward(DRIVE_SPEED_MAX)
        return "BACK!: too close  diam=%.1f target=%.1f" % (diam, target_diam)

    # 2. Re-centre first (proportional turn speed).
    if abs(error_x) > CENTER_DEADZONE:
        turn_speed = int(clamp(
            TURN_SPEED_MIN + (abs(error_x) - CENTER_DEADZONE) * TURN_GAIN,
            TURN_SPEED_MIN, TURN_SPEED_MAX,
        ))
        if error_x < 0:
            fc.turn_left(turn_speed)
            return "LEFT:  err=%+.0f diam=%.1f spd=%d" % (error_x, diam, turn_speed)
        fc.turn_right(turn_speed)
        return "RIGHT: err=%+.0f diam=%.1f spd=%d" % (error_x, diam, turn_speed)

    # 3. Centred -> maintain the locked diameter (= distance).
    diam_err = diam - target_diam                  # >0 too close, <0 too far
    norm     = abs(diam_err) / max(1.0, target_diam)
    drive_speed = int(clamp(
        DRIVE_SPEED_MIN + norm * DRIVE_GAIN,
        DRIVE_SPEED_MIN, DRIVE_SPEED_MAX,
    ))

    if diam_err < -deadzone:                        # object farther -> approach
        fc.forward(drive_speed)
        return "FWD:   diam=%.1f target=%.1f spd=%d" % (diam, target_diam, drive_speed)

    if diam_err > deadzone:                         # object closer -> back up
        safe_backward(drive_speed)
        return "BACK:  diam=%.1f target=%.1f spd=%d" % (diam, target_diam, drive_speed)

    # 4. Centred and at the right distance.
    fc.stop()
    return "HOLD:  diam=%.1f target=%.1f" % (diam, target_diam)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def main():
    picam2     = setup_camera()
    background = capture_background(picam2)

    candidate_history = deque(maxlen=60)
    tracker = None
    state   = "ACQUIRE"          # ACQUIRE | TRACK | LOST
    acquire_start = time.time()  # when the current empty-view wait began

    print("Running. Press ESC in the debug window or Ctrl-C to quit.")

    try:
        while True:
            frame = capture_small_bgr(picam2)
            gray  = make_gray(frame)
            mask  = None
            detection = None
            status    = ""

            # -- ACQUIRE ----------------------------------------------
            if state == "ACQUIRE":
                mask, candidate = find_new_object(gray, background)
                fc.stop()

                if candidate is not None:
                    # Object in view -> reset the empty-view timer so the
                    # background is never refreshed while an object is present.
                    acquire_start = time.time()

                    now = time.time()
                    candidate_history.append(
                        (now, candidate["cx"], candidate["cy"],
                         candidate["area"], candidate)
                    )
                    if stable_enough(candidate_history):
                        tracker = create_tracker(frame, candidate_history[-1][4])
                        candidate_history.clear()
                        if tracker is not None:
                            state  = "TRACK"
                            status = ("LOCKED - colour=%s target_diam=%.1f"
                                      % (describe_color(tracker["color"]),
                                         tracker["target_diam"]))
                            print(status)
                        else:
                            status = "ACQUIRE: lock failed, retrying"
                    else:
                        status = "ACQUIRE: hold still  area=%.0f" % candidate["area"]
                    detection = candidate
                else:
                    candidate_history.clear()
                    if time.time() - acquire_start > ACQUIRE_TIMEOUT_SECONDS:
                        state  = "LOST"
                        status = "ACQUIRE timeout: refreshing background"
                        print(status)
                    else:
                        waited = time.time() - acquire_start
                        status = ("ACQUIRE: waiting for object (%.1f/%.1f s)"
                                  % (waited, ACQUIRE_TIMEOUT_SECONDS))

            # -- TRACK ------------------------------------------------
            elif state == "TRACK":
                detection, mask = track_color(frame, tracker)
                status = control_robot(detection, tracker["target_diam"])

                if detection is None and tracker["lost_count"] > MAX_LOST_FRAMES:
                    fc.stop()
                    tracker = None
                    candidate_history.clear()
                    state  = "LOST"
                    status = "LOST: object left view"
                    print(status)

            # -- LOST -------------------------------------------------
            elif state == "LOST":
                fc.stop()
                # Robot may have moved, so the old background is stale - grab a
                # fresh one. Keep the scene empty during this ~1.5 s window.
                background = capture_background(picam2)
                candidate_history.clear()
                state  = "ACQUIRE"
                acquire_start = time.time()
                status = "Reacquiring..."
                print(status)

            # -- Debug display ----------------------------------------
            if SHOW_DEBUG:
                view = frame.copy()

                if detection is not None:
                    bx = int(detection["x"])
                    by = int(detection["y"])
                    bw = int(detection["w"])
                    bh = int(detection["h"])
                    cx = int(detection["cx"])
                    cy = int(detection.get("cy", by + bh / 2))
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.circle(view, (cx, cy), 3, (0, 0, 255), -1)
                    if "diam" in detection:
                        cv2.circle(view, (cx, cy),
                                   int(detection["diam"] / 2), (0, 255, 255), 1)

                cv2.line(view,
                         (IMAGE_CENTER_X, 0), (IMAGE_CENTER_X, FRAME_H),
                         (0, 0, 255), 1)
                cv2.putText(view, "[%s] %s" % (state, status), (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                cv2.imshow("PiCar tracker", view)
                if mask is not None:
                    cv2.imshow("mask", mask)

                if (cv2.waitKey(1) & 0xFF) == 27:   # ESC
                    break

    except KeyboardInterrupt:
        pass

    finally:
        fc.stop()
        picam2.stop()
        cv2.destroyAllWindows()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
