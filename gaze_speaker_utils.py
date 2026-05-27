# gaze_speaker_utils.py
# -*- coding: utf-8 -*-

# Standard library
import json
import math
import os
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import List

# Third-party
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

# Optional third-party (with fallbacks)
try:
    import insightface
    _INSIGHT_OK = True
except Exception:
    _INSIGHT_OK = False

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------
# Utilities
# -------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def iou_xyxy(a, b) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    areaB = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    denom = areaA + areaB - inter
    return float(inter / denom) if denom > 0 else 0.0


def clamp_bbox(b, w, h):
    x1 = max(0, min(int(b[0]), w-1))
    y1 = max(0, min(int(b[1]), h-1))
    x2 = max(0, min(int(b[2]), w-1))
    y2 = max(0, min(int(b[3]), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return [x1, y1, x2, y2]


def crop_face(gray_image, bbox, padding=0.775, size=112):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
    r = max(x2 - x1, y2 - y1) * padding
    p1x = int(cx - r); p1y = int(cy - r)
    p2x = int(cx + r); p2y = int(cy + r)
    h, w = gray_image.shape[:2]
    p1x = max(0, p1x); p1y = max(0, p1y)
    p2x = min(w, p2x); p2y = min(h, p2y)
    crop = gray_image[p1y:p2y, p1x:p2x]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# -------------------------
# SORT-style tracker (Kalman + Hungarian)
# -------------------------
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _tlbr_to_cxcywh(b):
    x1,y1,x2,y2 = b
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w; cy = y1 + 0.5 * h
    return cx, cy, w, h

def _cxcywh_to_tlbr(cx, cy, w, h):
    x1 = cx - 0.5 * w; y1 = cy - 0.5 * h
    return [x1, y1, x1 + w, y1 + h]

class Track:
    __slots__ = (
        "tid", "bbox", "conf", "misses", "age", "last_ts", "last_update_ms",
        "crops", "ema", "state", "gaze_scores", "gaze_ema", "gaze_state", "last_gaze_ms",
        "kf", "confirmed", "hits", "pid", "lip_landmarks"
    )
    def __init__(self, tid, bbox, conf, ts_ms, max_frames, kf):
        self.tid = tid
        self.pid = tid
        self.bbox = bbox[:]       # tlbr
        self.conf = float(conf)
        self.misses = 0
        self.age = 0
        self.last_ts = ts_ms
        self.last_update_ms = ts_ms
        # speaker / gaze buffers (unchanged)
        self.crops = deque(maxlen=max_frames)
        self.ema = 0.0            # speaker EMA
        self.state = 0            # speaker hysteresis state
        self.gaze_scores = deque(maxlen=180)
        self.gaze_ema = 0.0
        self.gaze_state = 0
        self.last_gaze_ms = ts_ms
        self.lip_landmarks = deque(maxlen=max_frames)
        # SORT
        self.kf = kf
        self.confirmed = False
        self.hits = 0

    # keep same helpers
    def update_speaking(self, score, alpha: float):
        self.ema = (1.0 - alpha) * self.ema + alpha * float(score)

    def update_gaze(self, score, alpha: float):
        self.gaze_ema = (1.0 - alpha) * self.gaze_ema + alpha * float(score)


class SORTTracker:
    """
    Lightweight SORT: per-track Kalman filter (cx,cy,w,h, vx,vy,vw,vh) + Hungarian association.
    - Predict every frame.
    - Associate and correct only on detection frames.
    - IDs persist through fast motion.
    """
    def __init__(self, iou_th=0.2, max_age=30, min_hits=2, W=0, H=0):
        self.iou_th = float(iou_th)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.tracks: List[Track] = []
        self.next_id = 1
        self.last_ts_ms = 0
        self.W, self.H = int(W), int(H)

    @staticmethod
    def _new_kf(init_cx, init_cy, init_w, init_h, dt=1/15.0):
        kf = cv2.KalmanFilter(8, 4, type=cv2.CV_32F)
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        F = np.eye(8, dtype=np.float32)
        F[0,4] = F[1,5] = F[2,6] = F[3,7] = float(dt)
        kf.transitionMatrix = F
        Hm = np.zeros((4,8), np.float32)
        Hm[0,0] = Hm[1,1] = Hm[2,2] = Hm[3,3] = 1.0
        kf.measurementMatrix = Hm
        # Noises (tuned for 2K faces; adjust if needed)
        kf.measurementNoiseCov = np.diag([5e-2, 5e-2, 5e-2, 5e-2]).astype(np.float32)
        #kf.measurementNoiseCov = np.diag([1e-1, 1e-1, 1e-1, 1e-1]).astype(np.float32)
        kf.errorCovPost = np.diag([1,1,1,1, 10,10,10,10]).astype(np.float32)
        kf.statePost = np.array([[init_cx],[init_cy],[init_w],[init_h],[0],[0],[0],[0]], np.float32)
        return kf

    @staticmethod
    def _set_dt(kf, dt):
        F = kf.transitionMatrix.copy()
        F[0,4] = F[1,5] = F[2,6] = F[3,7] = float(dt)
        kf.transitionMatrix = F

    def _predict_all(self, dt_s):
        for tr in self.tracks:
            self._set_dt(tr.kf, dt_s)
            pred = tr.kf.predict()
            cx, cy, w, h = [float(pred[i,0]) for i in range(4)]
            tr.bbox = clamp_bbox(_cxcywh_to_tlbr(cx, cy, max(2.0,w), max(2.0,h)), self.W, self.H)

    def _associate(self, detections):
        """
        detections: [(tlbr, conf), ...]
        Returns: matches [(ti, di)], unmatched_tidx, unmatched_didx
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        # cost = 1 - IoU (only consider IoU >= iou_th)
        cost = np.ones((len(self.tracks), len(detections)), dtype=np.float32)
        for ti, tr in enumerate(self.tracks):
            for di, (db, _) in enumerate(detections):
                iou = iou_xyxy(tr.bbox, db)
                if iou >= self.iou_th:
                    cost[ti, di] = 1.0 - float(iou)
                else:
                    cost[ti, di] = 1.0  # effectively impossible

        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
        else:
            # greedy fallback
            rows, cols = [], []
            tmp = cost.copy()
            for _ in range(min(cost.shape)):
                r, c = np.unravel_index(np.argmin(tmp), tmp.shape)
                if tmp[r, c] >= 1.0: break
                rows.append(int(r)); cols.append(int(c))
                tmp[r,:] = 1.0; tmp[:,c] = 1.0

        matches = []
        used_tr, used_det = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1.0:  # below threshold
                continue
            matches.append((r, c))
            used_tr.add(r); used_det.add(c)

        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_tr]
        unmatched_d = [i for i in range(len(detections)) if i not in used_det]
        return matches, unmatched_t, unmatched_d

    def update(self, detections, ts_ms, is_detection_frame: bool):
        # dt
        dt_s = 0.0
        if self.last_ts_ms > 0:
            dt_s = max(0.0, (ts_ms - self.last_ts_ms) / 1000.0)
        self.last_ts_ms = ts_ms

        # 1) predict all
        self._predict_all(dt_s)

        # 2) associate & correct only on detection frames
        if is_detection_frame:
            matches, unmatched_t, unmatched_d = self._associate(detections)

            # correct matched
            for ti, di in matches:
                tr = self.tracks[ti]
                db, dc = detections[di]
                cx, cy, w, h = _tlbr_to_cxcywh(db)
                meas = np.array([[cx],[cy],[w],[h]], np.float32)
                tr.kf.correct(meas)
                # update bbox from posterior
                post = tr.kf.statePost
                cx, cy, w, h = [float(post[i,0]) for i in range(4)]
                tr.bbox = clamp_bbox(_cxcywh_to_tlbr(cx, cy, max(2.0,w), max(2.0,h)), self.W, self.H)
                tr.conf = float(dc)
                tr.misses = 0
                tr.hits += 1
                if not tr.confirmed and tr.hits >= self.min_hits:
                    tr.confirmed = True
                tr.age += 1
                tr.last_update_ms = ts_ms

            # age unmatched (count a miss only on detection frames)
            for ti in unmatched_t:
                tr = self.tracks[ti]
                tr.misses += 1
                tr.age += 1
                tr.last_update_ms = ts_ms

            # spawn new tracks for unmatched detections (with SPAWN GUARD)
            for di in unmatched_d:
                db, dc = detections[di]

                # --- SPAWN GUARD: if this det overlaps a confirmed track a lot, skip spawning ---
                overlap_with_confirmed = False
                for t in self.tracks:
                    if t.confirmed and iou_xyxy(t.bbox, db) >= 0.6:   # threshold to taste
                        overlap_with_confirmed = True
                        break
                if overlap_with_confirmed:
                    continue

                cx, cy, w, h = _tlbr_to_cxcywh(db)
                kf = self._new_kf(cx, cy, w, h, dt=max(1e-3, dt_s if dt_s > 0 else 1/15.0))
                self.tracks.append(Track(self.next_id, clamp_bbox(db, self.W, self.H), float(dc), ts_ms, 500, kf))
                self.next_id += 1


            # prune
            self.tracks = [t for t in self.tracks if (t.misses <= self.max_age)]

        else:
            # non-detection frames: keep predictions, do not increment misses
            for tr in self.tracks:
                tr.age += 1
                tr.last_update_ms = ts_ms

        return self.tracks

    def cull_duplicates(self, iou_th=0.7):
        """Remove duplicate tracks that heavily overlap (keep the 'stronger' one)."""
        if len(self.tracks) <= 1:
            return
        keep = [True] * len(self.tracks)

        def strength(t):
            # priority: confirmed > hits > fewer misses
            return (1 if t.confirmed else 0, t.hits, -t.misses)

        for i in range(len(self.tracks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.tracks)):
                if not keep[j]:
                    continue
                iou = iou_xyxy(self.tracks[i].bbox, self.tracks[j].bbox)
                if iou >= iou_th:
                    a, b = self.tracks[i], self.tracks[j]
                    # keep the stronger one
                    winner, loser_idx = (a, j) if strength(a) >= strength(b) else (b, i)
                    keep[loser_idx] = False

        self.tracks = [t for t, k in zip(self.tracks, keep) if k]


# ---------------------------
# Tier-2: Lamp rejection + Appearance embedding + Stable ID
# ---------------------------
class FaceVerifier:
    """Fast “is-this-a-real-face” gate using MediaPipe + simple lamp/ceiling rejection."""
    def __init__(self, min_det_conf=0.5, min_trk_conf=0.5, refine=True):
        self._mp = mp.solutions.face_mesh
        self._fm = self._mp.FaceMesh(
            static_image_mode=False,
            refine_landmarks=refine,
            max_num_faces=5,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf,
        )

    def close(self):
        try: self._fm.close()
        except Exception: pass

    def verify(self, frame_bgr, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame_bgr.shape[:2]
        x1 = max(0, min(x1, W-2)); x2 = max(1, min(x2, W-1))
        y1 = max(0, min(y1, H-2)); y2 = max(1, min(y2, H-1))
        if x2 <= x1 or y2 <= y1:
            return False

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return False

        # lamp/ceiling quick reject: very bright & low saturation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if float(np.mean(hsv[...,2])) > 230 and float(np.mean(hsv[...,1])) < 25:
            return False

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        return bool(res.multi_face_landmarks)


class FaceEmbedder:
    """Appearance descriptor. Uses InsightFace if available, else HSV hist fallback."""
    def __init__(self):
        self.ok = False
        self._app = None
        if _INSIGHT_OK:
            try:
                self._app = insightface.app.FaceAnalysis(
                    name="antelopev2", allowed_modules=['detection', 'recognition']
                )
                self._app.prepare(ctx_id=0, det_size=(320, 320))
                self.ok = True
            except Exception:
                self._app = None
                self.ok = False

    def _embed_insight(self, frame_bgr, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return None
        faces = self._app.get(crop)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = getattr(f, "normed_embedding", None)
        return emb.astype(np.float32) if emb is not None else None

    def _embed_hsv_hist(self, frame_bgr, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (96,96))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        vecs = []
        for ch in range(3):
            h = cv2.calcHist([hsv],[ch],None,[32],[0,256]).flatten().astype(np.float32)
            h = h / (np.linalg.norm(h) + 1e-6)
            vecs.append(h)
        return np.concatenate(vecs, axis=0)  # (96,)

    def embed(self, frame_bgr, bbox):
        if self.ok:
            e = self._embed_insight(frame_bgr, bbox)
            if e is not None:
                return e
        return self._embed_hsv_hist(frame_bgr, bbox)


def _cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return 1.0 - float(np.dot(a, b))


class IDStabilizer:
    def __init__(self, use_insight=True,
                 th_insight=0.35, th_hist=0.22,
                 ema=0.85, max_people=5, max_stale_ms=10000):
        self.use_insight = bool(use_insight)
        self.th = float(th_insight if use_insight else th_hist)
        self.ema = float(ema)
        self.max_people = int(max_people)
        self.max_stale_ms = int(max_stale_ms)
        self.bank = {}     # pid -> {'emb': np.ndarray, 'last': ms, 'seen': int, 'bbox': [x1,y1,x2,y2]}
        self.next_pid = 1

    def get_pids(self):
        return list(self.bank.keys())

    def _match_one(self, emb, pid):
        rec = self.bank[pid]
        d = _cosine_distance(rec['emb'], emb)
        return d

    def observe_update(self, pid, emb, bbox, ts_ms):
        rec = self.bank[pid]
        rec['emb']  = (self.ema * rec['emb'] + (1.0 - self.ema) * emb).astype(np.float32)
        rec['bbox'] = [int(b) for b in bbox]
        rec['last'] = ts_ms
        rec['seen'] = rec.get('seen', 0) + 1

    def create_new(self, emb, bbox, ts_ms):
        pid = self.next_pid
        self.bank[pid] = {
            'emb': emb.astype(np.float32),
            'last': ts_ms,
            'seen': 1,
            'bbox': [int(b) for b in bbox],
        }
        self.next_pid += 1
        # keep bank small
        if len(self.bank) > self.max_people:
            oldest = min(self.bank.items(), key=lambda kv: kv[1]['last'])[0]
            if oldest != pid:
                self.bank.pop(oldest, None)
        return pid

    def prune(self, ts_ms):
        stale = [pid for pid, rec in self.bank.items()
                 if (ts_ms - rec['last']) > self.max_stale_ms]
        for pid in stale:
            self.bank.pop(pid, None)

class ReIDManager:
    """
    Per detection frame:
      - verify face, get embedding for each track
      - build a cost matrix track x pid
      - Hungarian assignment (one PID per frame)
      - update bank (EMA) for matched pairs
      - unmatched tracks => new PIDs
    Non-detection frames: carry over last pid mapping.
    """
    def __init__(self,
                max_people=5,
                lambda_iou=0.4,
                iou_gate_min=0.05,
                th_insight=0.35,
                th_hist=0.22):
        self.verifier  = FaceVerifier()
        self.embedder  = FaceEmbedder()
        self.stab      = IDStabilizer(
            use_insight=self.embedder.ok,
            th_insight=th_insight,
            th_hist=th_hist,
            ema=0.85,
            max_people=max_people,
            max_stale_ms=10000
        )
        self.tid2pid   = {}
        self.lambda_iou   = float(lambda_iou)
        self.iou_gate_min = float(iou_gate_min)


    def _build_cost(self, cand_embs, cand_bboxes, pids):
        """
        Returns cost matrix C (ncand x npid) and a mask of valid matches.
        Cost = distance + lambda*(1 - IoU) ; invalid when distance>th or IoU<gate.
        """
        nc = len(cand_embs)
        npid = len(pids)
        if npid == 0 or nc == 0:
            return None, None

        C = np.full((nc, npid), 1e6, dtype=np.float32)  # big default
        valid = np.zeros((nc, npid), dtype=bool)

        for i, (emb, bbox) in enumerate(zip(cand_embs, cand_bboxes)):
            for j, pid in enumerate(pids):
                rec = self.stab.bank[pid]
                d = _cosine_distance(rec['emb'], emb)
                if d > self.stab.th:
                    continue
                # IoU with last known pid bbox (if available)
                iou = 0.0
                if rec.get('bbox', None) is not None:
                    iou = iou_xyxy(bbox, rec['bbox'])
                if iou < self.iou_gate_min:
                    continue
                C[i, j] = d + self.lambda_iou * (1.0 - iou)
                valid[i, j] = True
        return C, valid

    def process(self, frame_bgr, tracks, ts_ms, is_detection_frame, W, H):
        if is_detection_frame:
            # 1) collect candidates (confirmed tracks only, verified faces)
            cand_trs, cand_embs, cand_bboxes = [], [], []
            for tr in tracks:
                if not getattr(tr, "confirmed", False):
                    continue
                bbox = clamp_bbox(tr.bbox, W, H)
                if not self.verifier.verify(frame_bgr, bbox):
                    continue
                emb = self.embedder.embed(frame_bgr, bbox)
                if emb is None:
                    continue
                cand_trs.append(tr)
                cand_embs.append(emb)
                cand_bboxes.append(bbox)

            pids = self.stab.get_pids()

            assigned_cand = set()
            assigned_pid  = set()
            # 2) Assign existing PIDs first via Hungarian (one-to-one)
            if len(pids) > 0 and len(cand_trs) > 0:
                C, valid = self._build_cost(cand_embs, cand_bboxes, pids)
                if C is not None:
                    if _HAS_SCIPY:
                        rows, cols = linear_sum_assignment(C)
                    else:
                        # greedy fallback: repeatedly pick the smallest valid cost
                        rows, cols = [], []
                        tmp = C.copy()
                        used_r, used_c = set(), set()
                        while True:
                            idx = np.argmin(tmp)
                            if not np.isfinite(tmp.flat[idx]):
                                break
                            r, c = np.unravel_index(idx, tmp.shape)
                            if tmp[r, c] >= 1e6:
                                break
                            rows.append(int(r)); cols.append(int(c))
                            used_r.add(int(r)); used_c.add(int(c))
                            tmp[r, :] = 1e6
                            tmp[:, c] = 1e6

                    for r, c in zip(rows, cols):
                        if C[r, c] >= 1e6 or not valid[r, c]:
                            continue
                        tr  = cand_trs[r]
                        pid = pids[c]
                        # assign pid to this track (unique per frame)
                        tr.pid = int(pid)
                        self.tid2pid[tr.tid] = int(pid)
                        self.stab.observe_update(pid, cand_embs[r], cand_bboxes[r], ts_ms)
                        assigned_cand.add(r)
                        assigned_pid.add(pid)

            # 3) Unmatched candidates -> new PID
            for r, tr in enumerate(cand_trs):
                if r in assigned_cand:
                    continue
                pid_new = self.stab.create_new(cand_embs[r], cand_bboxes[r], ts_ms)
                tr.pid = int(pid_new)
                self.tid2pid[tr.tid] = int(pid_new)

            # 4) prune stale PIDs
            self.stab.prune(ts_ms)

        else:
            # Non-detection frames: just carry over mapping
            for tr in tracks:
                tr.pid = self.tid2pid.get(tr.tid, getattr(tr, 'pid', None))

    def close(self):
        self.verifier.close()

# -------------------------
# Audio ring + capture
# -------------------------
class AudioRing:
    def __init__(self, samplerate=16000, max_seconds=12.0):
        self.sr = int(samplerate)
        self.max_sec = float(max_seconds)
        self.blocks = deque()
        self._dur = 0.0

    def push(self, block, ts_end_ms):
        self.blocks.append((int(ts_end_ms), np.asarray(block, dtype=np.float32)))
        # trim from left if over limit
        while True:
            dur = sum(len(b)/self.sr for _, b in self.blocks)
            if dur <= self.max_sec + 0.5:
                break
            self.blocks.popleft()

    def slice_last(self, window_sec, ts_now_ms):
        needed = int(self.sr * window_sec)
        if not self.blocks:
            return np.zeros(needed, dtype=np.float32)
        out = []
        total = 0
        for _, b in reversed(self.blocks):
            out.append(b); total += len(b)
            if total >= needed:
                break
        audio = np.concatenate(list(reversed(out)), axis=0) if out else np.zeros(0, dtype=np.float32)
        if len(audio) < needed:
            pad = np.zeros(needed - len(audio), dtype=np.float32)
            audio = np.concatenate([pad, audio], axis=0)
        elif len(audio) > needed:
            audio = audio[-needed:]
        return audio


class AudioCapture:
    def __init__(self, samplerate=16000, blocksize=1024):
        self.sr = samplerate
        self.block = blocksize
        self.ring = AudioRing(samplerate=self.sr, max_seconds=12.0)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        ts_end_ms = now_ms()
        mono = indata[:, 0].copy()
        self.ring.push(mono, ts_end_ms)

    def start(self):
        self.stream = sd.InputStream(
            channels=1, samplerate=self.sr, blocksize=self.block,
            dtype='float32', callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# -------------------------
# UVC (OpenCV) helpers for ZED
# -------------------------
RESOLUTION_MAP = {
    # (SBS_width, SBS_height, per_eye_width, per_eye_height)
    "HD2K":   (4416, 1242, 2208, 1242),
    "HD1080": (3840, 1080, 1920, 1080),
    "HD720":  (2560,  720, 1280,  720),
}

def open_zed_uvc(device_index: int, sbs_w: int, sbs_h: int, fps: int, use_mjpeg: bool = True) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video device index {device_index}")

    if use_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  sbs_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sbs_h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # warm-up
    for _ in range(5):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.01)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Failed to grab a frame from the UVC device.")
    return cap


def crop_eye_from_sbs(sbs: np.ndarray, which: str) -> np.ndarray:
    """Crop LEFT or RIGHT image from a side-by-side stereo frame."""
    h, w = sbs.shape[:2]
    mid = w // 2
    return sbs[:, :mid] if which == "LEFT" else sbs[:, mid:]


def nms_detections(dets, iou_th=0.45):
    if not dets: return dets
    boxes = np.array([d[0] for d in dets], np.float32)
    scores = np.array([d[1] for d in dets], np.float32)
    order = scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / np.maximum(1e-6, area_i + area_j - inter)
        order = order[np.where(iou <= iou_th)[0] + 1]
    return [dets[k] for k in keep]

# ---- Windows ----
def resize_for_display(img, scale: float):
    if img is None:
        return None
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def stable_pid(t):
    """Return persistent person id if present, else fall back to the track id."""
    return int(getattr(t, "pid", getattr(t, "tid", -1)))

class SessionLogger:
    """
    Sessionized logs:
      base_dir/
        session_meta.json
        hero_changes.jsonl
        tracks/track_<pid>.jsonl        # one file per person (stable PID)
    """
    def __init__(self, base_dir: str, meta: dict = None):
        self.base_dir = base_dir
        self.tracks_dir = os.path.join(base_dir, "tracks")
        os.makedirs(self.tracks_dir, exist_ok=True)
        self.hero_f = open(os.path.join(base_dir, "hero_changes.jsonl"), "a", buffering=1)

        if meta:
            with open(os.path.join(base_dir, "session_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        self.prev_hero_pid = None
        self.prev_hero_ts  = None

    def log_tracks(self, tracks, ts_ms: int, W: int, H: int):
        """Append one record per CONFIRMED track (on detection frames)."""
        for tr in tracks:
            if not getattr(tr, "confirmed", False):
                continue
            x1, y1, x2, y2 = clamp_bbox(tr.bbox, W, H)
            rec = {
                "ts_ms": int(ts_ms),
                "pid": stable_pid(tr),
                "tid": int(tr.tid),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "speak_ema": float(getattr(tr, "ema", 0.0)),
                "speak_state": int(getattr(tr, "state", 0)),
                "gaze_ema": float(getattr(tr, "gaze_ema", 0.0)),
                "gaze_state": int(getattr(tr, "gaze_state", 0)),
            }
            # write to per-person (PID) file
            path = os.path.join(self.tracks_dir, f"track_{rec['pid']}.jsonl")
            with open(path, "a", buffering=1) as tf:
                tf.write(json.dumps(rec) + "\n")

    def log_hero_change(self, hero_track, ts_ms: int, rule: str, speak_w: float, gaze_w: float):
        """
        Log only when hero PID changes. Include both PID (stable) and TID (current track),
        plus rule used at decision time.
        """
        hero_pid = int(stable_pid(hero_track)) if hero_track else None
        hero_tid = int(hero_track.tid) if hero_track else None

        if hero_pid != self.prev_hero_pid:
            evt = {
                "ts_ms": int(ts_ms),
                "hero_pid": hero_pid,
                "hero_tid": hero_tid,
                "prev_hero_pid": self.prev_hero_pid,
                "prev_duration_ms": int(ts_ms - self.prev_hero_ts) if self.prev_hero_ts is not None else None,
                "rule": str(rule),          # "both" or "weighted"
                "speak_w": float(speak_w),  # e.g., 0.7
                "gaze_w": float(gaze_w),    # e.g., 0.3
            }
            self.hero_f.write(json.dumps(evt) + "\n")
            self.prev_hero_pid = hero_pid
            self.prev_hero_ts  = ts_ms

    def close(self):
        try: self.hero_f.close()
        except: pass

# =========================
# Lip landmarks for LASER
# =========================

LASER_EXPECTED_K = 82  # LASER expects K=82 lip points (x,y in [0,1]) per frame

def _make_default_lip_indices_82():
    """
    Fallback: derive a lips vertex set from MediaPipe's FACEMESH_LIPS connections
    and coerce to length 82. Prefer passing your exact training indices if you have them.
    """
    try:
        lips_edges = list(mp.solutions.face_mesh.FACEMESH_LIPS)
        idxs = sorted(set([u for (u, v) in lips_edges] + [v for (u, v) in lips_edges]))
    except Exception:
        idxs = list(range(LASER_EXPECTED_K))  # last-resort dummy (82 sequential)
    if len(idxs) >= LASER_EXPECTED_K:
        return idxs[:LASER_EXPECTED_K]
    return idxs + [idxs[-1]] * (LASER_EXPECTED_K - len(idxs))

class LipLandmarks:
    """
    Compute 82 lip landmarks per confirmed track, normalized to the 112x112 face
    crop space used by LASER. Each frame we append either a (82,2) array in [0,1]
    or None (on failure/skip) to Track.lip_landmarks.
    """
    def __init__(self,
                 indices_82=None,
                 size=112,
                 padding=0.775,
                 every=2,
                 refine=False,
                 topk=3):
        self.size = int(size)
        self.padding = float(padding)
        self.every = max(1, int(every))
        self.topk = max(1, int(topk))
        self.frame_ctr = 0

        self.idx = list(indices_82) if indices_82 is not None else _make_default_lip_indices_82()
        if len(self.idx) != LASER_EXPECTED_K:
            raise ValueError(f"LipLandmarks: got {len(self.idx)} indices; expected {LASER_EXPECTED_K}.")

        # Lean FaceMesh instance (no iris) for performance
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=bool(refine),
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def close(self):
        try: self._fm.close()
        except Exception: pass

    def _crop_face_bgr(self, frame_bgr, bbox):
        # Same geometry as crop_face(...), but BGR and returns 112x112
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        r = max(x2 - x1, y2 - y1) * self.padding
        p1x = int(max(0, cx - r)); p1y = int(max(0, cy - r))
        p2x = int(min(frame_bgr.shape[1], cx + r)); p2y = int(min(frame_bgr.shape[0], cy + r))
        crop = frame_bgr[p1y:p2y, p1x:p2x]
        if crop.size == 0:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

    def _extract_on_crop(self, crop_bgr_112):
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(crop_bgr_112, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        # Coordinates already normalized to the crop; clamp to [0,1]
        pts = np.zeros((LASER_EXPECTED_K, 2), dtype=np.float32)
        for i, j in enumerate(self.idx):
            x = float(np.clip(lm[j].x, 0.0, 1.0))
            y = float(np.clip(lm[j].y, 0.0, 1.0))
            pts[i] = [x, y]
        return pts

    def update_tracks(self, frame_bgr, tracks, W, H, confirmed_only=True):
        """
        Append one entry per track this frame:
          - (82,2) lip points in [0,1] if computed,
          - None if skipped/failed.
        We throttle work: only the largest 'topk' confirmed tracks every `every` frames.
        """
        self.frame_ctr += 1
        do_compute = (self.frame_ctr % self.every) == 0

        # ensure per-track buffer exists
        for tr in tracks:
            if not hasattr(tr, "lip_landmarks") or tr.lip_landmarks is None:
                tr.lip_landmarks = deque(maxlen=getattr(tr, "crops", deque()).maxlen or 500)

        if not do_compute:
            # keep time alignment: append None for all tracks
            for tr in tracks:
                tr.lip_landmarks.append(None)
            return

        # choose top-K largest confirmed tracks to process
        cands = []
        for tr in tracks:
            if confirmed_only and not getattr(tr, "confirmed", False):
                continue
            x1, y1, x2, y2 = clamp_bbox(tr.bbox, W, H)
            area = max(1.0, float(x2 - x1) * float(y2 - y1))
            cands.append((area, tr, (x1, y1, x2, y2)))
        cands.sort(key=lambda t: -t[0])
        cands = cands[:self.topk]

        processed = set()
        for _, tr, bb in cands:
            crop_bgr = self._crop_face_bgr(frame_bgr, bb)
            pts = self._extract_on_crop(crop_bgr)  # (82,2) or None
            tr.lip_landmarks.append(pts)               # may be None; handled later
            processed.add(id(tr))

        # others: append None to keep alignment
        for tr in tracks:
            if id(tr) not in processed:
                tr.lip_landmarks.append(None)


def build_laser_landmarks(landmarks_deque, window_t: int, expected_k: int = LASER_EXPECTED_K):
    """
    Convert a track's lip_landmarks history into a LASER-ready array:
      shape = (1, 3, T, K, 2), dtype=float32, normalized to [0,1] (crop space).
    Missing entries are **zero-filled**.
    """
    T = int(window_t)
    if landmarks_deque is None:
        lm_win = [None] * T
    else:
        hist = list(landmarks_deque)
        # left-pad with None to reach T, then take last T samples
        lm_win = ([None] * max(0, T - len(hist)) + hist)[-T:]

    arr = np.zeros((T, expected_k, 2), dtype=np.float32)  # ZERO by default
    for t, pts in enumerate(lm_win):
        if isinstance(pts, np.ndarray) and pts.ndim == 2:
            k = min(expected_k, pts.shape[0])
            arr[t, :k, :] = pts[:k, :]  # if fewer points, rest stay zero

    # match LASER's expected (1, 3, T, K, 2)
    out = np.tile(arr[None, None, ...], (1, 3, 1, 1, 1))  # duplicate across 3 visual channels
    return out.astype(np.float32)

class SpeechTranscriber:
    """
    On-demand speech-to-text using speech_recognition + WhisperX.
    - Opens the mic ONLY during transcribe_once() and then releases it.
    - Lazy-loads WhisperX model on first use to avoid slow startup elsewhere.
    - Does NOT keep any PortAudio/PyAudio stream open between calls.
    """
    _model_lock = threading.Lock()
    _model = None

    def __init__(self,
                 whisper_model_name: str = "large-v3",
                 device: str = "cuda",
                 compute_type: str = "float32",
                 download_root: str = None,
                 language: str = "de", #do not change creates some issues
                 mic_sample_rate: int = 16000,
                 energy_threshold: int = 150,
                 pause_threshold: float = 0.8,
                 dynamic_energy: bool = False,
                 device_index: int | None = None):
        """
        device_index: pass an explicit microphone index to avoid default-device conflicts.
        """
        # Defer imports so we don't impose them on modules that don't need STT
        try:
            import speech_recognition as sr  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"SpeechTranscriber dependencies missing: {e}")
        # --- backend selection ---
        try:
            import whisperx
            # HARD disable diarization dependency
            self.backend = "whisperx"
            self._whisperx_ok = True
            print("[ASR] Using WhisperX")
        except Exception as e:
            self.backend = "faster-whisper"
            self._whisperx_ok = False
            print(f"[ASR] WhisperX unusable → fallback ({e})")

        self._sr = None          # set on first call
        self._whisperx = None    # set on first call
        self._fw_model = None    # for faster-whisper
        self._mic_rate = int(mic_sample_rate)
        self._energy = int(energy_threshold)
        self._pause = float(pause_threshold)
        self._dyn_energy = bool(dynamic_energy)
        self._dev_index = device_index

        self._w_name = whisper_model_name
        self._w_device = device
        self._w_compute = compute_type
        self._w_dlroot = download_root
        self._w_lang = language

    def _ensure_faster_whisper_loaded(self):
        if self._fw_model is None:
            from faster_whisper import WhisperModel

            self._fw_model = WhisperModel(
                self._w_name,
                device=self._w_device,
                compute_type="float16" if self._w_device == "cuda" else "int8"
            )

    def _ensure_loaded(self):
        if self._sr is None:
            import speech_recognition as sr
            self._sr = sr

        # --- backend-specific loading ---
        if self.backend == "whisperx":
            try:
                if self._whisperx is None:
                    import whisperx
                    self._whisperx = whisperx

                with SpeechTranscriber._model_lock:
                    if SpeechTranscriber._model is None:
                        SpeechTranscriber._model = self._whisperx.load_model(
                            self._w_name,
                            device=self._w_device,
                            compute_type=self._w_compute,
                            download_root=self._w_dlroot,
                            language=self._w_lang
                        )

            except Exception as e:
                print(f"[ASR] WhisperX failed at runtime → fallback ({e})")
                self.backend = "faster-whisper"
                SpeechTranscriber._model = None
                self._ensure_faster_whisper_loaded()

        else:  # faster-whisper fallback
            self._ensure_faster_whisper_loaded()

    def transcribe_once(self, timeout: float | None = None) -> str | None:
        """
        Record a short utterance and return the recognized text (or None if silence).
        This function blocks while listening once.
        """
        self._ensure_loaded()
        sr = self._sr
        r = sr.Recognizer()
        r.energy_threshold = self._energy
        r.pause_threshold = self._pause
        r.dynamic_energy_threshold = self._dyn_energy

        # Open and close the mic within this call to avoid long-held device handles
        with sr.Microphone(sample_rate=self._mic_rate,
                   device_index=self._dev_index) as source:

            try:
                audio = r.listen(source, timeout=timeout)
            except sr.WaitTimeoutError:
                return None

        # Convert to float32 mono [-1, 1]
        raw = audio.get_raw_data()
        import numpy as np
        sig = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

        # Transcribe
        if self.backend == "whisperx":
            try:
                result = SpeechTranscriber._model.transcribe(
                    sig,
                    batch_size=16
                )
            except IndexError:
                return None
            segs = result.get("segments") or []
            if not segs:
                return None
            out = "".join(seg.get("text", "") or "" for seg in segs).strip()
            return out or None

        else:  # faster-whisper
            segments, _ = self._fw_model.transcribe(sig, language=self._w_lang)
            out = "".join(seg.text for seg in segments).strip()
            return out or None

class SessionMedia:
    """
    - Audio: writes 16 kHz mono WAV continuously during the session.
    - Video: lazy-open CV VideoWriter at first frame; you call write_frame() whenever you have a frame.
    - On close: stop audio & video, then try to mux WAV + MP4 into a single MP4 via ffmpeg.
    """
    def __init__(self, base_dir: str, fps: int = 15, audio_sr: int = 16000, audio_dev_index: int | None = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # paths
        self.wav_path = self.base_dir / "session.wav"
        self.vid_path = self.base_dir / "session_video.mp4"
        self.out_path = self.base_dir / "session_muxed.mp4"

        # audio
        self._audio_sr = int(audio_sr)
        self._audio_dev = audio_dev_index
        self._audio_stream = None
        self._wav = None

        # video
        self._fps = int(fps)
        self._vw = None
        self._vw_size = None  # (W, H)

        # start audio immediately
        self._start_audio()

    def _start_audio(self):
        # create WAV file
        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)  # int16
        self._wav.setframerate(self._audio_sr)

        # open sounddevice stream that writes straight to WAV
        def _cb(indata, frames, time_info, status):
            # indata is float32 [-1,1]; convert to int16
            pcm = np.clip(indata[:, 0], -1.0, 1.0)
            pcm_i16 = (pcm * 32767.0).astype(np.int16).tobytes()
            try:
                self._wav.writeframes(pcm_i16)
            except Exception:
                pass

        self._audio_stream = sd.InputStream(
            channels=1, samplerate=self._audio_sr, dtype='float32',
            callback=_cb, blocksize=1024, device=self._audio_dev
        )
        self._audio_stream.start()

    def _stop_audio(self):
        try:
            if self._audio_stream:
                self._audio_stream.stop()
                self._audio_stream.close()
        finally:
            self._audio_stream = None
            if self._wav:
                try: self._wav.close()
                except Exception: pass
                self._wav = None

    def _ensure_video(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        size = (w, h)
        if self._vw is not None and self._vw_size == size:
            return
        # (Re)open writer
        if self._vw is not None:
            try: self._vw.release()
            except Exception: pass
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
        self._vw = cv2.VideoWriter(str(self.vid_path), fourcc, self._fps, size, True)
        if not self._vw.isOpened():
            raise RuntimeError("Failed to open VideoWriter for session_video.mp4")
        self._vw_size = size

    def write_frame(self, frame_bgr):
        """
        Safe to call only when you actually have a camera frame (e.g., in vision mode).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return
        self._ensure_video(frame_bgr)
        self._vw.write(frame_bgr)

    def _stop_video(self):
        if self._vw is not None:
            try: self._vw.release()
            except Exception: pass
            self._vw = None

    def _ffmpeg_exists(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True
        except Exception:
            return False
    
    def _probe_duration(self, path: Path) -> float:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, check=True
            )
            return float(out.stdout.strip())
        except Exception:
            return 0.0


    def _mux_ffmpeg(self):
        if not (self.vid_path.exists() and self.wav_path.exists()):
            return False
        if not self._ffmpeg_exists():
            return False

        # -y overwrite; align durations; AAC audio @128k; copy video
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(self.vid_path),
            "-i", str(self.wav_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(self.out_path)
        ]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception:
            return False

    def close_and_mux(self):
        """
        Stop audio+video and try to produce `session_muxed.mp4`.
        If mux fails (no ffmpeg or other issue), you'll still have WAV and MP4 separately.
        """
        self._stop_video()
        self._stop_audio()
        ok = self._mux_ffmpeg()
        return str(self.out_path if ok else self.vid_path), ok

class LiveAVRecorder:
    """
    Start/stop a single MP4 recording with audio+video muxed in real time.
    - start(width, height): spawns ffmpeg, opens a sounddevice InputStream, begins piping.
    - write_frame(frame_bgr): push BGR24 frames (size must match width x height).
    - stop(): closes audio stream, closes pipes, finalizes MP4.
    """
    def __init__(self, out_path: str, fps: int = 15, audio_sr: int = 16000, audio_dev_index: int | None = None):
        self.out_path = Path(out_path)
        self.fps = int(fps)
        self.audio_sr = int(audio_sr)
        self.audio_dev_index = audio_dev_index

        self.proc = None
        self.width = None
        self.height = None
        self._audio_fd_r = None
        self._audio_fd_w = None
        self._audio_stream = None
        self._running = False

    def start(self, width: int, height: int):
        if self._running:
            return
        self.width, self.height = int(width), int(height)

        # Create a dedicated pipe (fd 3) for raw audio (s16le)
        self._audio_fd_r, self._audio_fd_w = os.pipe()

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-f", "s16le",
            "-ar", str(self.audio_sr),
            "-ac", "1",
            "-thread_queue_size", "512",
            "-i", "pipe:3",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(self.out_path)
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(self._audio_fd_r,)
        )
        os.close(self._audio_fd_r)
        self._audio_fd_r = None

        # ---- audio stream here ----
        def _audio_cb(indata, frames, time_info, status):
            try:
                os.write(self._audio_fd_w, indata.tobytes())
            except Exception:
                pass

        try:
            self._audio_stream = sd.InputStream(
                channels=1,
                samplerate=self.audio_sr,
                dtype="int16",
                callback=_audio_cb,
                blocksize=1024,
                device=self.audio_dev_index
            )
            self._audio_stream.start()
        except Exception as e:
            print(f"[WARN] Audio init failed with sr={self.audio_sr}, retrying at 48000Hz ({e})")
            self.audio_sr = 48000
            self._audio_stream = sd.InputStream(
                channels=1,
                samplerate=self.audio_sr,
                dtype="int16",
                callback=_audio_cb,
                blocksize=1024,
                device=self.audio_dev_index
            )
            self._audio_stream.start()

        self._running = True


    def write_frame(self, frame_bgr: np.ndarray):
        if not self._running or self.proc is None or self.proc.stdin is None:
            return
        if frame_bgr is None or frame_bgr.size == 0:
            return
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(f"Frame size {w}x{h} != recorder size {self.width}x{self.height}")
        try:
            buf = np.ascontiguousarray(frame_bgr)  # <-- ensure contiguous
            self.proc.stdin.write(buf.tobytes())
        except Exception:
            pass

    def stop(self):
        if not self._running:
            return
        # Stop audio
        try:
            if self._audio_stream:
                self._audio_stream.stop()
                self._audio_stream.close()
        except Exception:
            pass
        self._audio_stream = None

        # Close audio pipe write end
        try:
            if self._audio_fd_w is not None:
                os.close(self._audio_fd_w)
        except Exception:
            pass
        self._audio_fd_w = None

        # Close video stdin so ffmpeg can finalize
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass

        # Wait a moment for ffmpeg to write trailer
        try:
            if self.proc:
                self.proc.wait(timeout=3.0)
        except Exception:
            pass

        self.proc = None
        self._running = False

__all__ = [
    # utils
    "now_ms", "iou_xyxy", "clamp_bbox", "crop_face", "nms_detections",
    # tracker
    "Track", "SORTTracker", "ReIDManager",
    # audio
    "AudioRing", "AudioCapture",
    # zed helpers
    "RESOLUTION_MAP", "open_zed_uvc", "crop_eye_from_sbs",
    # extra
    "resize_for_display", "stable_pid", "SessionLogger",
    # lip landmarks
    "LipLandmarks", "build_laser_landmarks"
]

__all__.extend([
    "SpeechTranscriber",
    "SessionMedia",
    "LiveAVRecorder",
])
