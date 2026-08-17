# -*- coding: utf-8 -*-
"""EchoPath device runtime - NVIDIA Jetson Nano (Tanna TechBiz Eagle 101 carrier).

GENERATED from EchoPath_Jetson_Pipeline.ipynb (cells marked LIFTABLE).
Do not hand-edit: change the notebook and re-run tools/extract_device.py, so the
workstation and the device keep executing identical fusion and alert logic. That
equivalence is what makes the workstation-vs-device fidelity comparison meaningful -
any difference measured is then attributable to FP16 and input resolution, not to a
porting mistake.

TARGET  JetPack 4.6.x / L4T 32.7 / Ubuntu 18.04 / Python 3.6 / CUDA 10.2 / TensorRT 8.2
        Maxwell SM 5.3 => FP16 only. INT8 needs DP4A (compute capability 6.1+).

PYTHON 3.6 ONLY - no dataclasses (3.7), no walrus (3.8), no f"{x=}" (3.8),
no builtin generics (3.9).

Setup on the device:
    python3 -m venv --system-site-packages ~/echopath-env
    source ~/echopath-env/bin/activate
    pip install pycuda            # tensorrt, cv2 and numpy come from JetPack
    # NEVER pip install opencv-python - it shadows JetPack's GStreamer+CUDA build
    #                                    and silently kills the CSI camera path

Build the engines on the device (they are not portable across TRT version or GPU):
    /usr/src/tensorrt/bin/trtexec --onnx=onnx/yolov8n_416.onnx --fp16 --verbose \
        --workspace=512 --saveEngine=engines/yolo_416_fp16_trt8201.plan
    /usr/src/tensorrt/bin/trtexec --onnx=onnx/midas_small_256.onnx --fp16 \
        --workspace=512 --saveEngine=engines/midas_256_fp16_trt8201.plan

Run:
    python3 echopath_device.py --source csi --threaded
    python3 echopath_device.py --source file --input Video_Data/Classroom.MP4
"""

import os, sys, platform, json, math, time, threading, queue, subprocess
from collections import defaultdict, deque

import numpy as np
import cv2

IS_JETSON = True          # this file only ever runs on the Jetson
BACKEND   = 'tensorrt'


if True:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    print('tensorrt : %s' % trt.__version__)
    DEVICE = 'cuda'


# ---------------------------------------------------------------- paths
ROOT         = r'E:\EchoPath' if not IS_JETSON else '/home/echopath'
YOLO_WEIGHTS = os.path.join(ROOT, 'runs', 'detect', 'runs', 'echopath', 'v3', 'weights', 'best.pt')
MIDAS_LOCAL  = os.path.join(ROOT, 'models', 'midas_small')      # vendored, not torch.hub
VIDEO_DIR    = os.path.join(ROOT, 'Video_Data')
OUTPUT_DIR   = os.path.join(ROOT, 'out_jetson')
ENGINE_DIR   = os.path.join(ROOT, 'engines')
ONNX_DIR     = os.path.join(ROOT, 'onnx')
VOCAB_DIR    = os.path.join(ROOT, 'assets', 'audio')
for _d in (OUTPUT_DIR, ENGINE_DIR, ONNX_DIR, VOCAB_DIR):
    if not os.path.isdir(_d):
        os.makedirs(_d)

YOLO_ENGINE  = os.path.join(ENGINE_DIR, 'yolo_416_fp16_trt8201.plan')
DEPTH_ENGINE = os.path.join(ENGINE_DIR, 'midas_256_fp16_trt8201.plan')

# ---------------------------------------------------------------- source
SOURCE          = 'files'          # 'files' | 'csi'
SUPPORTED_EXTS  = ('.mp4', '.avi', '.mov', '.mkv')   # matched case-insensitively
CSI_SENSOR_ID   = 0
CSI_W, CSI_H    = 1280, 720
CSI_FPS         = 30
CSI_FLIP        = 0                # nvvidconv flip-method
USE_OPENCV_AUTO_ROTATE = True      # file input only

# ---------------------------------------------------------------- detection
IMGSZ           = 640              # 640, NOT 416. The mAP ablation on the 2780
                                   # validation IMAGES said 416 was free, but on the
                                   # 1080x1920 portrait VIDEOS letterboxing to 416
                                   # leaves ~234 px of content and loses up to 52% of
                                   # detections. Re-measure on real IMX219 frames
                                   # before trading resolution for Jetson latency.
CONF_GLOBAL     = 0.40             # paper operating point
CONF_FLOOR      = 0.15             # predict at this, then filter per class
CONF_OVERRIDES  = {                # measured: chair@0.25 gave 2.3x detections
    'chair': 0.25, 'stairs': 0.30, 'wet_floor_sign': 0.30,
}
IOU_NMS         = 0.45

# ---------------------------------------------------------------- depth
DEPTH_SIZE      = 256              # MiDaS Small native; do NOT upsample
CLOSE_PCTL      = 75
MEDIUM_PCTL     = 50
THRESH_EMA      = 0.15             # smooth the thresholds, not the map
ZONE_HYST       = 0.03             # Schmitt-trigger delta on zone boundaries
ZONE_VOTE       = 3                # majority vote window
INNER_CROP      = 0.50             # preserve source-notebook semantics

DEPTH_MODE      = 'midas'          # 'midas' | 'geometric' | 'fused'

# ---------------------------------------------------------------- runtime
PROCESS_EVERY_N = 1                # detection cadence
DEPTH_EVERY_N   = 3                # depth cadence - decoupled from detection
DEBUG_VIEW      = False            # annotation costs real ARM CPU; off on device
QUEUE_DEPTH     = 1                # drop-stale, never buffer

# ---------------------------------------------------------------- alerts
ALERT_COOLDOWN_S = 2.5             # paper III-B.3 - the single source of truth now
# Rendered phrases are 1.29-3.18 s long (median 2.35 s). Scheduling alerts closer than
# that guarantees overlapping speech in the baked video and stale queued audio live, so
# the floor must exceed a typical phrase.
MIN_GAP_S        = 2.5             # floor between any two spoken alerts
ESCALATE_MIN_S   = 2.5             # a closing close-ahead hazard may insert ONE extra
                                   # alert this long after the last normal one
MAX_PHRASE_S     = 3.2             # longest rendered phrase; used to serialise audio
HYST_K, HYST_M   = 2, 3            # must appear in K of last M frames
SILENT_CLASSES   = ('whiteboard', 'monitor', 'notice_board', 'window',
                    'backpack', 'handbag')      # last two: zero training instances
ALERT_ZONES      = ('close', 'medium')

# ---------------------------------------------------------------- audio
ALSA_DEVICE      = 'default'       # USB dongle - Eagle 101 has no analog codec
SPEECH_RATE      = 170
SPEECH_VOLUME    = 1.0
USE_VOCABULARY   = True            # pre-rendered WAVs; pyttsx3 is the fallback

# ---------------------------------------------------------------- geometry
F_PX             = None            # measured focal length in px; None -> estimate below
F_PX_RATIO       = 0.80            # fallback f_px = 0.80 * max(W,H) => ~64 deg FOV.
                                   # Test footage is iPhone 17 portrait 1080x1920, whose
                                   # video FOV puts f_px in 1280-1600; the fallback gives
                                   # 1536, so the HEIGHT cue is good to roughly +/-7%.
                                   # Use calibrate_f_px() to remove even that.
Y_HORIZON        = None            # image row of the horizon. None => assume a LEVEL
                                   # camera and use frame_h/2. Using 0 (the old default)
                                   # overstates nearness by 2-25x on the ground cue.
CLOSE_M          = 1.5             # metre thresholds for DEPTH_MODE='geometric'
MEDIUM_M         = 3.0
CAM_HEIGHT_M     = 1.35            # chest mount
EDGE_TOL         = 4               # px from frame edge => truncated

REAL_HEIGHTS = {
    'person': 1.70, 'door': 2.00, 'chair': 0.90, 'table': 0.75,
    'water_dispenser': 1.30, 'waste_bin': 0.70, 'wet_floor_sign': 0.60,
    'fire_extinguisher': 0.60, 'potted_plant': 0.80, 'elevator': 2.00,
    'slide': 1.20, 'monitor': 0.45,
    # no reliable height prior -> ground-plane cue only
    'column': None, 'stairs': None, 'whiteboard': None,
    'notice_board': None, 'window': None, 'backpack': None, 'handbag': None,
}

# MUST match the exported engine's output channels exactly. The v3 weights were
# trained on data.yaml with nc=19: 'backpack' and 'handbag' are present with ZERO
# instances. Dropping them here (to match the paper's 17) shifts every index above
# 10, so the device would announce fire_extinguisher as 'table', wet_floor_sign as
# 'monitor' (which is silenced => wet floor hazard produces NO alert), and crash on
# indices 17-18. Keep 19 until the model is retrained at nc=17; silence the two
# empty classes instead.
CLASS_NAMES = ['person', 'chair', 'door', 'stairs', 'slide', 'whiteboard',
               'notice_board', 'elevator', 'column', 'window', 'water_dispenser',
               'backpack', 'handbag', 'potted_plant', 'waste_bin',
               'fire_extinguisher', 'wet_floor_sign', 'table', 'monitor']   # nc=19
PAPER_CLASSES = 17                 # what the report tabulates (excludes the two empties)


class Detector(object):
    """bbox in ORIGINAL frame pixel coords, xyxy."""
    names = CLASS_NAMES
    def infer(self, frame_bgr):
        raise NotImplementedError


class DepthEstimator(object):
    """Returns a DEPTH_SIZE x DEPTH_SIZE float32 map normalised to [0,1].
    Higher value = closer, matching MiDaS inverse-depth convention."""
    def infer(self, frame_bgr):
        raise NotImplementedError


# ------------------------------------------------------------------ torch side


# --------------------------------------------------------------- tensorrt side
class TRTEngine(object):
    """Minimal TensorRT 8.2 wrapper - static shapes, batch 1, pinned host buffers."""
    def __init__(self, plan_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(plan_path, 'rb') as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.dev, self.shapes = [], {}, {}, {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            h = cuda.pagelocked_empty(size, dtype)
            d = cuda.mem_alloc(h.nbytes)
            self.bindings.append(int(d))
            self.host[name], self.dev[name], self.shapes[name] = h, d, shape
            if self.engine.binding_is_input(i):
                self.in_name = name
            else:
                self.out_name = name

    def __call__(self, x):
        np.copyto(self.host[self.in_name], x.ravel())
        self.cuda.memcpy_htod_async(self.dev[self.in_name],
                                    self.host[self.in_name], self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.host[self.out_name],
                                    self.dev[self.out_name], self.stream)
        self.stream.synchronize()
        return self.host[self.out_name].reshape(self.shapes[self.out_name])


class TRTDetector(Detector):
    def __init__(self, plan=YOLO_ENGINE, imgsz=IMGSZ, names=CLASS_NAMES):
        self.eng = TRTEngine(plan)
        self.imgsz = imgsz
        self.names = names
        ch = self.eng.shapes[self.eng.out_name][1]      # (1, 4+nc, anchors)
        if ch - 4 != len(names):
            raise RuntimeError(
                'engine has %d classes but CLASS_NAMES has %d - indices would be '
                'silently remapped. Re-export or fix CLASS_NAMES.' % (ch - 4, len(names)))

    def infer(self, frame_bgr):
        blob, r, dw, dh = letterbox(frame_bgr, self.imgsz)
        raw = self.eng(blob)
        return decode_yolo(raw, self.names, r, dw, dh,
                           frame_bgr.shape[1], frame_bgr.shape[0])


class MidasTRT(DepthEstimator):
    def __init__(self, plan=DEPTH_ENGINE, size=DEPTH_SIZE):
        self.eng = TRTEngine(plan)
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], np.float32)

    def infer(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        d = self.eng(np.ascontiguousarray(img.transpose(2, 0, 1)[None])).squeeze()
        lo, hi = float(d.min()), float(d.max())
        return (d - lo) / (hi - lo + 1e-6)


def build_backends():
    if DEPTH_MODE == 'geometric':
        return TRTDetector(), None        # no depth network is loaded at all
    return TRTDetector(), MidasTRT()


def clamp_bbox(bbox, w, h):
    """Clip to frame with a degenerate-box guard. Unchanged from the source notebook."""
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(w - 1, int(bbox[2])); y2 = min(h - 1, int(bbox[3]))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def direction_of(bbox, frame_w):
    cx = (bbox[0] + bbox[2]) / 2.0
    third = frame_w / 3.0
    if cx < third:
        return 'left'
    if cx < 2 * third:
        return 'ahead'
    return 'right'


class ZoneClassifier(object):
    """Percentile zoning with EMA-smoothed thresholds, Schmitt hysteresis and a
    majority vote. Same decision rule as the source notebook, stabilised."""

    def __init__(self, close_pctl=CLOSE_PCTL, medium_pctl=MEDIUM_PCTL,
                 alpha=THRESH_EMA, hyst=ZONE_HYST, vote=ZONE_VOTE):
        self.cp, self.mp = close_pctl, medium_pctl
        self.alpha, self.hyst = alpha, hyst
        self.t_close = self.t_medium = None
        self.vote_n = vote
        self._votes = defaultdict(lambda: deque(maxlen=vote))
        self._last = {}

    def update_frame(self, depth_map):
        """Once per DEPTH frame - not once per detection."""
        c, m = np.percentile(depth_map, [self.cp, self.mp])
        if self.t_close is None:
            self.t_close, self.t_medium = float(c), float(m)
        else:
            a = self.alpha
            self.t_close  = a * float(c) + (1 - a) * self.t_close
            self.t_medium = a * float(m) + (1 - a) * self.t_medium
        return self.t_close, self.t_medium

    def classify(self, value, key):
        """Schmitt trigger: entering a zone needs +hyst, leaving needs -hyst."""
        prev = self._last.get(key)
        tc, tm = self.t_close, self.t_medium
        if prev == 'close':
            z = 'close' if value > tc - self.hyst else (
                'medium' if value > tm - self.hyst else 'far')
        elif prev == 'medium':
            z = 'close' if value > tc + self.hyst else (
                'medium' if value > tm - self.hyst else 'far')
        else:
            z = 'close' if value > tc + self.hyst else (
                'medium' if value > tm + self.hyst else 'far')
        self._votes[key].append(z)
        v = self._votes[key]
        z = max(set(v), key=list(v).count) if len(v) == v.maxlen else z
        self._last[key] = z
        return z


def depth_in_bbox(depth_map, bbox, frame_w, frame_h, inner=INNER_CROP):
    """Median depth over the inner crop, with the bbox scaled DOWN into depth space."""
    dh, dw = depth_map.shape[:2]
    sx, sy = float(dw) / frame_w, float(dh) / frame_h
    x1, y1, x2, y2 = clamp_bbox((bbox[0] * sx, bbox[1] * sy,
                                 bbox[2] * sx, bbox[3] * sy), dw, dh)
    pad = (1.0 - inner) / 2.0
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    crop = depth_map[y1 + py:y2 - py, x1 + px:x2 - px]
    if crop.size == 0:
        crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return float(np.median(crop))


def fuse(detections, depth_map, frame_w, frame_h, zoner):
    """Attach depth, zone and direction to each detection."""
    if depth_map is None and DEPTH_MODE in ('midas', 'fused') and detections:
        # Without this guard every object silently becomes 'far', the run yields zero
        # alerts, and it looks like "nothing was detected" rather than a fault.
        raise RuntimeError(
            "DEPTH_MODE=%r needs a depth map but none was supplied. Did build_backends() "
            "return depther=None because DEPTH_MODE was 'geometric' at the time?"
            % DEPTH_MODE)
    if depth_map is not None:
        zoner.update_frame(depth_map)
    out = []
    for i, d in enumerate(detections):
        rec = dict(d)
        rec['direction'] = direction_of(d['bbox'], frame_w)
        if depth_map is None:
            rec['depth'], rec['proximity'] = None, 'far'
        else:
            v = depth_in_bbox(depth_map, d['bbox'], frame_w, frame_h)
            key = '%s|%s' % (d['label'], rec['direction'])
            rec['depth'] = v
            rec['proximity'] = 'far' if v is None else zoner.classify(v, key)
        out.append(rec)
    return out


def generate_alerts(dets):
    """close/medium only, close first. Same rule as the source notebook."""
    a = [d for d in dets if d['proximity'] in ALERT_ZONES]
    a.sort(key=lambda d: 0 if d['proximity'] == 'close' else 1)
    return a


def effective_horizon(frame_h):
    """Image row of the horizon. Calibrate Y_HORIZON for the real mount; until then
    assume the camera is level, which puts the horizon at the vertical centre. The old
    default of 0 treated the top of the frame as infinitely distant and made every
    floor-standing object read far too near."""
    if Y_HORIZON is not None:
        return float(Y_HORIZON)
    return frame_h * 0.5


def calibrate_f_px(known_distance_m, real_height_m, bbox_height_px):
    """f_px from one measured shot: put an object of known height at a measured
    distance, detect it, pass its bbox height.  f = Z * h_px / H_real"""
    return float(known_distance_m) * float(bbox_height_px) / float(real_height_m)


def effective_f_px(frame_w, frame_h):
    """Calibrated focal length in pixels if known, else estimated from frame size.
    0.80 * max(W,H) corresponds to roughly a 64 deg FOV - typical of both the test
    phone footage and the IMX219. UNCALIBRATED: absolute distances are +/- ~50%
    until measured with the deployment camera, so treat zones, not metres, as the
    output."""
    if F_PX:
        return float(F_PX)
    return F_PX_RATIO * float(max(frame_w, frame_h))


def zone_from_metres(z_m):
    if z_m is None:
        return 'far'          # no usable cue -> never alert
    if z_m <= CLOSE_M:
        return 'close'
    if z_m <= MEDIUM_M:
        return 'medium'
    return 'far'


def geometric_distance(bbox, label, frame_h, frame_w):
    """Return (Z_metres, cue, truncated). Approximate until F_PX is calibrated."""
    x1, y1, x2, y2 = bbox
    h_px = max(1.0, y2 - y1)
    truncated = (y1 <= EDGE_TOL) or (y2 >= frame_h - EDGE_TOL)
    f = effective_f_px(frame_w, frame_h)
    y_hor = effective_horizon(frame_h)

    Hr = REAL_HEIGHTS.get(label)
    Z_h = (Hr * f / h_px) if (Hr and not truncated) else None          # height cue
    below = y2 - y_hor
    # above the horizon there is no floor contact point - the cue does not apply
    Z_g = (CAM_HEIGHT_M * f / below) if below > 1.0 else None           # ground cue

    if Z_h is None and Z_g is None:
        return None, 'none', truncated
    if Z_h is None:
        return Z_g, 'ground', truncated
    if Z_g is None:
        return Z_h, 'height', truncated
    return min(Z_h, Z_g), 'fused', truncated       # conservative: nearer estimate wins


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker(object):
    """Greedy same-class IoU association. Gives objects identity across frames, which is
    what enables smoothing, closing speed and time-to-contact."""

    def __init__(self, iou_thresh=0.30, ema=0.35):
        self.thresh, self.ema = iou_thresh, ema
        self.live, self.next_id = {}, 0

    def update(self, dets):
        used, out = set(), []
        for tid in list(self.live.keys()):
            t = self.live[tid]
            best, bi = self.thresh, None
            for i, d in enumerate(dets):
                if i in used or d['label'] != t['label']:
                    continue
                s = _iou(t['bbox'], d['bbox'])
                if s > best:
                    best, bi = s, i
            if bi is None:
                t['misses'] += 1
                if t['misses'] > 2:
                    self.live.pop(tid)
                continue
            used.add(bi)
            d = dets[bi]
            z = d.get('Z')
            if z is not None:
                t['Z'] = z if t['Z'] is None else self.ema * z + (1 - self.ema) * t['Z']
                t['dZ'] = None if t['Z_prev'] is None else t['Z'] - t['Z_prev']
                t['Z_prev'] = t['Z']
            t['bbox'], t['misses'], t['age'] = d['bbox'], 0, t['age'] + 1
            d['track_id'], d['Z_smooth'], d['dZ'] = tid, t['Z'], t['dZ']
            d['ttc'] = (-t['Z'] / t['dZ']) if (t['dZ'] and t['dZ'] < 0) else None
            out.append(d)
        for i, d in enumerate(dets):
            if i in used:
                continue
            tid = self.next_id; self.next_id += 1
            self.live[tid] = dict(bbox=d['bbox'], label=d['label'], misses=0, age=1,
                                  Z=d.get('Z'), Z_prev=d.get('Z'), dZ=None)
            d['track_id'], d['Z_smooth'], d['dZ'], d['ttc'] = tid, d.get('Z'), None, None
            out.append(d)
        return out


def attach_geometric(dets, frame_h, frame_w):
    for d in dets:
        Z, cue, trunc = geometric_distance(d['bbox'], d['label'], frame_h, frame_w)
        d['Z'], d['cue'], d['truncated'] = Z, cue, trunc
    return dets


PRIORITY = {'close': 0, 'medium': 1, 'far': 2}


class AlertPolicy(object):
    """Decides WHAT to say. Knows nothing about audio."""

    def __init__(self, cooldown=ALERT_COOLDOWN_S, min_gap=MIN_GAP_S,
                 k=HYST_K, m=HYST_M, silent=SILENT_CLASSES):
        self.cooldown, self.min_gap = cooldown, min_gap
        self.k, self.m = k, m
        self.silent = set(silent)
        self._seen = defaultdict(lambda: deque(maxlen=m))
        self._last_key = {}
        self._esc_used = {}
        self._last_any = -1e9

    def step(self, dets, now):
        """Return the single alert to speak, or None."""
        keys_this_frame = set()
        for d in dets:
            if d['label'] in self.silent or d['proximity'] not in ALERT_ZONES:
                continue
            k = (d['label'], d['direction'])
            keys_this_frame.add(k)
            self._seen[k].append(1)
        for k in list(self._seen.keys()):          # absent this frame
            if k not in keys_this_frame:
                self._seen[k].append(0)

        eligible = []
        for d in dets:
            if d['label'] in self.silent or d['proximity'] not in ALERT_ZONES:
                continue
            k = (d['label'], d['direction'])
            if sum(self._seen[k]) < self.k:        # entry gate
                continue
            # A closing obstacle dead ahead may pre-empt its own cooldown - but only
            # ONCE per cooldown window, or it machine-guns the user.
            # ONE extra alert per cooldown window, and only after ESCALATE_MIN_S.
            # Gating on _esc_used alone was the bug: a NORMAL alert never armed the
            # limiter, so escalation fired again at the MIN_GAP floor (1.03 s).
            last_k = self._last_key.get(k, -1e9)
            escalating = (d.get('dZ') is not None and d['dZ'] < 0
                          and d['proximity'] == 'close' and d['direction'] == 'ahead'
                          and (now - last_k) >= ESCALATE_MIN_S
                          and self._esc_used.get(k, -1e9) < last_k)
            if (now - self._last_key.get(k, -1e9)) < self.cooldown and not escalating:
                continue
            d['_escalated'] = escalating
            eligible.append(d)

        if not eligible:
            return None
        # Hard floor between ANY two spoken alerts. 'close' does NOT get an exemption
        # here - stacking that with the escalation bypass produced three alerts in
        # 0.6 s against a 2.5 s cooldown.
        if (now - self._last_any) < self.min_gap:
            return None
        eligible.sort(key=lambda d: (PRIORITY[d['proximity']], d.get('Z_smooth') or 0.0))
        best = eligible[0]
        bk = (best['label'], best['direction'])
        if best.get('_escalated'):
            self._esc_used[bk] = now       # do NOT push out the normal 2.5 s cadence
        else:
            self._last_key[bk] = now
        self._last_any = now
        return best


def phrase_for(d):
    """Natural phrasing, carried over from the source notebook's make_spoken_text."""
    lab = d['label'].replace('_', ' ')
    if d['direction'] == 'ahead':
        return '%s %s ahead' % (lab, d['proximity'])
    return '%s %s on your %s' % (lab, d['proximity'], d['direction'])


def vocab_key(label, proximity, direction):
    return '%s_%s_%s' % (label, proximity, direction)


class AudioOut(object):
    """Speaks. Pre-rendered WAVs by default - zero synthesis cost on the Nano.
    Priority pre-emption: a close alert cancels a playing medium one."""

    def __init__(self, vocab_dir=VOCAB_DIR, device=ALSA_DEVICE,
                 use_vocab=USE_VOCABULARY):
        self.vocab_dir, self.device, self.use_vocab = vocab_dir, device, use_vocab
        self.q = queue.Queue(maxsize=1)
        self.current_priority = 99
        self._proc = None
        self._stop = threading.Event()
        self._tts = None
        self.t = threading.Thread(target=self._worker)
        self.t.daemon = True
        self.t.start()

    def _speak_fallback(self, text):
        if self._tts is None:
            import pyttsx3
            self._tts = pyttsx3.init()
            self._tts.setProperty('rate', SPEECH_RATE)
            self._tts.setProperty('volume', SPEECH_VOLUME)
        self._tts.say(text); self._tts.runAndWait()

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.15)
            except queue.Empty:
                continue
            if item is None:
                break
            key, text, prio = item
            wav = os.path.join(self.vocab_dir, key + '.wav')
            try:
                if self.use_vocab and os.path.exists(wav):
                    if os.name == 'nt':
                        import winsound
                        winsound.PlaySound(wav, winsound.SND_FILENAME)
                    else:
                        self._proc = subprocess.Popen(
                            ['aplay', '-q', '-D', self.device, wav])
                        self._proc.wait(); self._proc = None
                else:
                    self._speak_fallback(text)
            except Exception as e:
                print('[audio] %s' % e)
            self.current_priority = 99

    def say(self, d):
        prio = PRIORITY[d['proximity']]
        if prio < self.current_priority and self._proc is not None:
            try:
                self._proc.terminate()      # pre-empt the lower-priority alert
            except Exception:
                pass
        if self.q.full():
            if prio >= self.current_priority:
                return False                # drop - never buffer a stale alert
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.current_priority = prio
        self.q.put((vocab_key(d['label'], d['proximity'], d['direction']),
                    phrase_for(d), prio))
        return True

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except Exception:
            pass


def csi_pipeline(sensor_id=CSI_SENSOR_ID, w=CSI_W, h=CSI_H, fps=CSI_FPS, flip=CSI_FLIP):
    return ('nvarguscamerasrc sensor-id=%d ! '
            'video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1 ! '
            'nvvidconv flip-method=%d ! video/x-raw,format=BGRx ! '
            'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1'
            % (sensor_id, w, h, fps, flip))


def iter_csi():
    cap = cv2.VideoCapture(csi_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError('CSI camera did not open. Check the vendor BSP device tree.')
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
    finally:
        cap.release()


def iter_video_file(path, auto_rotate=USE_OPENCV_AUTO_ROTATE):
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1 if auto_rotate else 0)
    except Exception:
        pass
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
    finally:
        cap.release()


def run_pipeline(frames, detector, depther, audio=None, policy=None,
                 zoner=None, tracker=None, on_frame=None, fps_hint=30.0,
                 max_frames=None, verbose=False):
    """Single-threaded reference implementation. Same decisions as the threaded
    version - use this to validate, ThreadedPipeline to hit frame rate."""
    policy  = policy  or AlertPolicy()
    zoner   = zoner   or ZoneClassifier()
    tracker = tracker or Tracker()
    log, depth_map, idx, t0 = [], None, 0, time.time()

    for frame in frames:
        if max_frames and idx >= max_frames:
            break
        h, w = frame.shape[:2]
        if idx % PROCESS_EVERY_N == 0:
            dets = detector.infer(frame)
            if DEPTH_MODE in ('midas', 'fused') and idx % DEPTH_EVERY_N == 0:
                depth_map = depther.infer(frame)
            if DEPTH_MODE == 'geometric':
                dets = attach_geometric(dets, h, w)
                for d in dets:
                    d['direction'] = direction_of(d['bbox'], w)
                    d['depth'] = d['Z']
                    d['proximity'] = zone_from_metres(d['Z'])
            else:
                dets = fuse(dets, depth_map, w, h, zoner)
                dets = attach_geometric(dets, h, w)
            dets = tracker.update(dets)

            now = idx / float(fps_hint)
            chosen = policy.step(generate_alerts(dets), now)
            if chosen is not None:
                log.append(dict(frame=idx, t=round(now, 3), label=chosen['label'],
                                proximity=chosen['proximity'],
                                direction=chosen['direction'],
                                phrase=phrase_for(chosen)))
                if audio is not None:
                    audio.say(chosen)
                if verbose:
                    print('  [%6.2fs] %s' % (now, phrase_for(chosen)))
            if on_frame is not None:
                on_frame(frame, dets, idx)
        idx += 1

    return dict(alerts=log, frames=idx, seconds=time.time() - t0)


class ThreadedPipeline(object):
    """Capture -> detect -> depth(every Nth) -> fuse+alert -> audio.
    Depth-1 drop-stale queues throughout: for a walking user a 600 ms-old alert
    describes a world that no longer exists, so dropping beats delivering late."""

    def __init__(self, detector, depther, audio=None):
        self.det, self.dep, self.audio = detector, depther, audio
        self.q_cap = queue.Queue(maxsize=QUEUE_DEPTH)
        self.q_det = queue.Queue(maxsize=QUEUE_DEPTH)
        self.depth_map = None
        self.stop_flag = threading.Event()
        self.stats = defaultdict(list)

    @staticmethod
    def _put_drop(q, item):
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _capture(self, frames):
        for i, f in enumerate(frames):
            if self.stop_flag.is_set():
                break
            self._put_drop(self.q_cap, (i, f))
        self.stop_flag.set()

    def _depth(self):
        while not self.stop_flag.is_set():
            try:
                i, f = self.q_cap.queue[0]
            except IndexError:
                time.sleep(0.005); continue
            if i % DEPTH_EVERY_N == 0:
                t = time.time()
                self.depth_map = self.dep.infer(f)
                self.stats['depth'].append(time.time() - t)
            time.sleep(0.001)

    def _detect(self):
        while not self.stop_flag.is_set():
            try:
                i, f = self.q_cap.get(timeout=0.1)
            except queue.Empty:
                continue
            t = time.time()
            dets = self.det.infer(f)
            self.stats['detect'].append(time.time() - t)
            self._put_drop(self.q_det, (i, f, dets, self.depth_map))

    def run(self, frames, fps_hint=30.0):
        policy, zoner, tracker = AlertPolicy(), ZoneClassifier(), Tracker()
        log = []
        threads = [threading.Thread(target=self._capture, args=(frames,)),
                   threading.Thread(target=self._detect),
                   threading.Thread(target=self._depth)]
        for t in threads:
            t.daemon = True; t.start()
        t_start = time.time(); n = 0
        while not (self.stop_flag.is_set() and self.q_det.empty()):
            try:
                i, f, dets, dm = self.q_det.get(timeout=0.1)
            except queue.Empty:
                continue
            h, w = f.shape[:2]
            t = time.time()
            dets = fuse(dets, dm, w, h, zoner)
            dets = tracker.update(attach_geometric(dets, h, w))
            chosen = policy.step(generate_alerts(dets), i / float(fps_hint))
            self.stats['fuse'].append(time.time() - t)
            if chosen is not None:
                log.append(dict(frame=i, label=chosen['label'],
                                proximity=chosen['proximity'],
                                direction=chosen['direction']))
                if self.audio is not None:
                    self.audio.say(chosen)
            n += 1
        for t in threads:
            t.join(timeout=1.0)
        el = time.time() - t_start
        return dict(alerts=log, frames=n, seconds=el, fps=n / el if el else 0,
                    stats={k: (float(np.mean(v)), float(np.percentile(v, 95)))
                           for k, v in self.stats.items() if v})


# ================================================================ entry point
def main():
    import argparse
    ap = argparse.ArgumentParser(description='EchoPath device runtime')
    ap.add_argument('--source', choices=['csi', 'file'], default='csi')
    ap.add_argument('--input', help='video path, required when --source file')
    ap.add_argument('--tag', default='nano', help='prefix for the alert log')
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--threaded', action='store_true',
                    help='staged pipeline - use this for the live camera')
    args = ap.parse_args()

    detector, depther = build_backends()
    audio = None if args.no_audio else AudioOut()

    if args.source == 'csi':
        frames = iter_csi()
        fps = float(CSI_FPS)
        name = 'csi'
    else:
        if not args.input:
            raise SystemExit('--input is required with --source file')
        cap = cv2.VideoCapture(args.input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        frames = iter_video_file(args.input)
        name = os.path.basename(args.input)

    if audio is not None and os.path.exists(os.path.join(VOCAB_DIR, 'system_ready.wav')):
        audio.q.put(('system_ready', 'echo path ready', 0))

    res = {'alerts': [], 'frames': 0}
    try:
        if args.threaded:
            res = ThreadedPipeline(detector, depther, audio).run(frames, fps_hint=fps)
            print('%d frames, %.1f fps' % (res['frames'], res.get('fps', 0.0)))
        else:
            res = run_pipeline(frames, detector, depther, audio=audio, fps_hint=fps)
            print('%d frames in %.1fs' % (res['frames'], res['seconds']))
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        if audio is not None:
            audio.stop()

    out = os.path.join(OUTPUT_DIR, '%s__%s.jsonl' % (args.tag, name))
    fh = open(out, 'w')
    try:
        for a in res['alerts']:
            a['video'] = name
            fh.write(json.dumps(a) + '\n')
    finally:
        fh.close()
    print('%d alerts -> %s' % (len(res['alerts']), out))


if __name__ == '__main__':
    main()
