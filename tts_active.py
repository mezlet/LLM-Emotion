# tts_active.py
import asyncio
import json
import time
import sys
import requests
import websockets

# ================== CONFIG ==================
AUTH_TOKEN = "ZWNFuNQVIPyztWCfPPM5VLPslpj8rR"

AUDIO_DEVICES_URL = "http://emah/tritium/audio/devices"
LEVELS_WS_URL     = "ws://emah/tritium/ws/audio/levels"  # change if different

TARGET_NAME_SUBSTR = "acapela"    # device name filter (case-insensitive)

# Smoothed level = EMA of raw levels:
EMA_ALPHA        = 0.35           # 0..1; higher = more responsive, lower = smoother

# Hysteresis thresholds:
ON_THRESH        = 0.08           # go ACTIVE when smoothed >= this
OFF_THRESH       = 0.02           # go INACTIVE only when smoothed <= this
QUIET_HOLD_SECS  = 0.15           # must stay <= OFF_THRESH this long to flip to INACTIVE

# Optional: require a brief minimum active time before reporting True (prevents 1-2 blips)
MIN_ACTIVE_SECS  = 0.10
# ============================================

HEADERS    = {"X-Tritium-Auth-Token": AUTH_TOKEN}
WS_HEADERS = {"X-Tritium-Auth-Token": AUTH_TOKEN, "Accept": "application/json"}

# Public state you can read from other modules
tts_active = False       # debounced, hysteretic active flag
tts_level  = 0.0         # current raw level for the tracked device
tts_ema    = 0.0         # smoothed (EMA) level we actually threshold on

# Internal timers
_last_above_on = 0.0     # when EMA last crossed/was above ON_THRESH
_quiet_since   = None    # when EMA last went/stayed below OFF_THRESH


def find_target_device():
    """Return (device_id, device_name, volume_scale) for the target sink-input."""
    try:
        resp = requests.get(AUDIO_DEVICES_URL, headers=HEADERS, timeout=5)
        resp.raise_for_status()
        devices = resp.json()
    except Exception as e:
        print(f"[tts_active] ERROR fetching devices: {e}", file=sys.stderr)
        return None, None, None

    for d in devices:
        name = str(d.get("name", ""))
        if TARGET_NAME_SUBSTR.lower() in name.lower():
            try:
                dev_id = int(d["id"])
                scale  = float(d.get("volume", 0.0))
                return dev_id, name, scale
            except Exception:
                continue
    return None, None, None


def get_device_scale(dev_id: int) -> float:
    """Lookup the device's current volume (scale)."""
    try:
        resp = requests.get(AUDIO_DEVICES_URL, headers=HEADERS, timeout=5)
        resp.raise_for_status()
        for d in resp.json():
            if int(d.get("id", -1)) == dev_id:
                return float(d.get("volume", 0.0))
    except Exception as e:
        print(f"[tts_active] WARN refresh scale failed: {e}", file=sys.stderr)
    return 1.0


def _update_state_from_level(raw_level: float):
    """
    Apply EMA smoothing + hysteresis + hold times to update (tts_active, tts_ema).
    """
    global tts_active, tts_level, tts_ema, _last_above_on, _quiet_since
    now = time.time()
    tts_level = max(0.0, min(1.0, float(raw_level)))

    # EMA smoothing
    if tts_ema == 0.0:
        tts_ema = tts_level
    else:
        tts_ema = EMA_ALPHA * tts_level + (1.0 - EMA_ALPHA) * tts_ema

    # Hysteresis with hold:
    if tts_active:
        # We're active; look for sustained quiet to turn off
        if tts_ema <= OFF_THRESH:
            if _quiet_since is None:
                _quiet_since = now
            elif (now - _quiet_since) >= QUIET_HOLD_SECS:
                tts_active = False
                _quiet_since = None
        else:
            _quiet_since = None
    else:
        # We're inactive; look for level high enough to turn on
        if tts_ema >= ON_THRESH:
            if _last_above_on == 0.0:
                _last_above_on = now
            elif (now - _last_above_on) >= MIN_ACTIVE_SECS:
                tts_active = True
        else:
            _last_above_on = 0.0


async def listen_levels_for_device(dev_id: int, dev_name: str, init_scale: float):
    """
    Subscribe to the levels WS and maintain tts_active/tts_ema globals.
    """
    scale = init_scale
    last_scale_refresh = 0.0

    while True:
        try:
            async with websockets.connect(LEVELS_WS_URL, additional_headers=WS_HEADERS) as ws:
                print(f"[tts_active] Connected to levels WS: {LEVELS_WS_URL}")
                print(f"[tts_active] Tracking {dev_id} ({dev_name}) | initial scale={scale:.2f}")
                async for msg in ws:
                    # Periodically refresh the output volume (scale) (not used in state, but handy to print)
                    now = time.time()
                    if now - last_scale_refresh > 5.0:
                        scale = get_device_scale(dev_id)
                        last_scale_refresh = now

                    try:
                        data = json.loads(msg)  # e.g., {"8": 0.73, "11": 0.02, ...}
                    except Exception:
                        continue

                    key = str(dev_id)
                    if key in data:
                        _update_state_from_level(float(data[key]))
                        # Optional console view, comment if too chatty
                        bar = "█" * int(40 * max(0.0, min(1.0, tts_ema)))  # visualize EMA
                        # print(f"{dev_name} | {bar:<40} | level={tts_level:.2f} ema={tts_ema:.2f} scale={scale:.2f} TTS={tts_active}")
        except Exception as e:
            print(f"[tts_active] WS error: {e} — reconnecting in 1s...")
            await asyncio.sleep(1.0)


def is_tts_active() -> bool:
    """Public getter: debounced, hysteretic TTS speaking state."""
    return tts_active


def current_level() -> float:
    """Raw instantaneous level (0..1) for the tracked device."""
    return tts_level


def current_ema() -> float:
    """Smoothed (EMA) level (0..1) used for state transitions."""
    return tts_ema


# Optional: standalone run for quick testing
if __name__ == "__main__":
    dev_id, name, scale = find_target_device()
    if dev_id is None:
        print(f"[tts_active] Device containing '{TARGET_NAME_SUBSTR}' not found.")
        try:
            r = requests.get(AUDIO_DEVICES_URL, headers=HEADERS, timeout=5)
            if r.ok:
                print("Available devices:")
                for d in r.json():
                    print(f"  {d.get('id')}: {d.get('name')}")
        except Exception:
            pass
        sys.exit(1)
    try:
        asyncio.run(listen_levels_for_device(dev_id, name, scale))
    except KeyboardInterrupt:
        pass
