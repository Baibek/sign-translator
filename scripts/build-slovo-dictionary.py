#!/usr/bin/env python3
"""Build the built-in Russian Sign Language word dictionary from the open Slovo dataset.

Slovo (SberDevices, https://github.com/ai-forever/slovo) contains 20,400 videos of 1,000 RSL signs and
MediaPipe hand landmarks for every frame. Licence: Slovo public licence (CC BY-SA 4.0 variant) —
attribution required, derived data shared under the same terms. See public/signs/ATTRIBUTION.md.

For each sign this script picks the clearest sample (both/one hand tracked in most frames, smooth motion),
smooths and resamples it, and writes compact files the web app animates:
  public/signs/ru/index.json      {"version":1, "words": {"привет": "0001", ...}, ...}
  public/signs/ru/<id>.json       {"text":..., "aspect": w/h, "frames": [[hand, hand], ...]}
        hand = [slot, x0,y0,z0, x1,y1,z1, ...]  (integers, coords ×1000 in video space)

Runs in GitHub Actions (needs internet); the landmark file is ~1.2 GB and is streamed with ijson.
Usage: python scripts/build-slovo-dictionary.py [--landmarks FILE] [--annotations FILE] [--limit N]
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict

BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/"
LANDMARKS_URL = BASE + "slovo_mediapipe.json"
ZIP_URLS = [BASE + "slovo.zip", BASE + "slovo_full360.zip"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "signs", "ru")
N_FRAMES = 28
SKIP = {"no_event", "no event", ""}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (enough for zipfile to read one member)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_annotations():
    """annotations.csv lives inside the 16 GB zip; read just that member via HTTP ranges."""
    last = None
    for url in ZIP_URLS:
        try:
            with zipfile.ZipFile(HttpRangeFile(url)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                print("CSV files in", url, ":", names, file=sys.stderr)
                name = next((n for n in names if "annotation" in n.lower()), names[0])
                return z.read(name).decode("utf-8-sig")
        except Exception as e:  # try the next archive
            last = e
            print("annotations from", url, "failed:", e, file=sys.stderr)
    raise RuntimeError(f"could not read annotations.csv: {last}")


def parse_annotations(text):
    dialect = csv.Sniffer().sniff(text[:4096], delimiters="\t,;")
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    out = {}
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        vid = r.get("attachment_id")
        label = r.get("text", "")
        if not vid or label.lower() in SKIP:
            continue
        try:
            w, h = float(r.get("width") or 0), float(r.get("height") or 0)
        except ValueError:
            w = h = 0
        out[vid] = {"text": label, "aspect": (w / h) if w and h else 9 / 16, "user": r.get("user_id", "")}
    return out


def iter_landmarks(path_or_url):
    import ijson  # streaming JSON parser (pip install ijson)

    if os.path.exists(path_or_url):
        f = open(path_or_url, "rb")
    else:
        f = urllib.request.urlopen(path_or_url, timeout=600)
    with f:
        yield from ijson.kvitems(f, "", use_float=True)


def frame_hands(frame):
    """frame: {hand_key: [{x,y,z}×21]} → list of 21×3 point lists."""
    hands = []
    if isinstance(frame, dict):
        items = frame.values()
    elif isinstance(frame, list):
        items = frame
    else:
        return hands
    for pts in items:
        if isinstance(pts, list) and len(pts) >= 21 and isinstance(pts[0], dict):
            hands.append([[float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))] for p in pts[:21]])
    return hands[:2]


def assign_slots(frames):
    """Give hands stable slots across frames: 0 = left side of the image, 1 = right side (viewer's view)."""
    out = []
    for hands in frames:
        if len(hands) == 2:
            a, b = sorted(hands, key=lambda h: h[0][0])
            out.append({0: a, 1: b})
        elif len(hands) == 1:
            out.append({None: hands[0]})
        else:
            out.append({})
    # single hands: put them in the slot whose last known wrist is nearest (default: dominant = right side)
    last = {0: None, 1: None}
    for fr in out:
        if None in fr:
            h = fr.pop(None)
            x = h[0][0]
            d = {s: abs(x - last[s][0]) if last[s] else None for s in (0, 1)}
            if d[0] is not None and (d[1] is None or d[0] < d[1]):
                fr[0] = h
            else:
                fr[1] = h
        for s in (0, 1):
            if s in fr:
                last[s] = fr[s][0]
    return out


def quality(frames):
    if len(frames) < 8:
        return -1
    tracked = sum(1 for f in frames if f) / len(frames)
    jitter = 0.0
    n = 0
    for a, b in zip(frames, frames[1:]):
        for s in a.keys() & b.keys():
            jitter += math.dist(a[s][0][:2], b[s][0][:2])
            n += 1
    jitter = jitter / n if n else 1
    return tracked - min(jitter * 4, 0.5)


def smooth_resample(frames, n=N_FRAMES):
    """Fill short gaps, smooth with a 3-frame window, resample to n frames."""
    for s in (0, 1):
        idx = [i for i, f in enumerate(frames) if s in f]
        if len(idx) < max(3, len(frames) * 0.25):  # hand barely seen: drop it
            for f in frames:
                f.pop(s, None)
            continue
        for i in range(idx[0], idx[-1] + 1):  # linear fill inside the visible span
            if s not in frames[i]:
                prev = max(j for j in idx if j < i)
                nxt = min(j for j in idx if j > i)
                t = (i - prev) / (nxt - prev)
                frames[i][s] = [[a + (b - a) * t for a, b in zip(p, q)] for p, q in zip(frames[prev][s], frames[nxt][s])]
    sm = []
    for i, f in enumerate(frames):
        g = {}
        for s, pts in f.items():
            nb = [frames[j][s] for j in (i - 1, i, i + 1) if 0 <= j < len(frames) and s in frames[j]]
            g[s] = [[sum(h[k][c] for h in nb) / len(nb) for c in range(3)] for k in range(21)]
        sm.append(g)
    out = []
    for k in range(n):
        pos = k * (len(sm) - 1) / (n - 1)
        i = int(pos)
        t = pos - i
        a, b = sm[i], sm[min(i + 1, len(sm) - 1)]
        g = {}
        for s in a.keys() | b.keys():
            if s in a and s in b:
                g[s] = [[p[c] + (q[c] - p[c]) * t for c in range(3)] for p, q in zip(a[s], b[s])]
            else:
                g[s] = (a.get(s) if t < 0.5 else b.get(s)) or a.get(s) or b.get(s)
        out.append(g)
    return out


def encode(frames):
    enc = []
    for f in frames:
        hands = []
        for s in sorted(f):
            flat = [s]
            for x, y, z in f[s]:
                flat += [round(x * 1000), round(y * 1000), round(z * 1000)]
            hands.append(flat)
        enc.append(hands)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default=LANDMARKS_URL)
    ap.add_argument("--annotations", default=None, help="local annotations.csv (default: read from the dataset zip)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (testing)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    text = open(args.annotations, encoding="utf-8-sig").read() if args.annotations else fetch_annotations()
    ann = parse_annotations(text)
    print(f"annotations: {len(ann)} videos, {len({a['text'] for a in ann.values()})} signs", file=sys.stderr)

    best = {}  # sign text → (score, vid, frames, aspect)
    seen = 0
    for vid, frames in iter_landmarks(args.landmarks):
        seen += 1
        if args.limit and seen > args.limit:
            break
        a = ann.get(vid)
        if not a:
            continue
        fr = assign_slots([frame_hands(f) for f in frames])
        q = quality(fr)
        if q > best.get(a["text"], (-9,))[0]:
            best[a["text"]] = (q, vid, fr, a["aspect"])
        if seen % 2000 == 0:
            print(f"… {seen} videos, {len(best)} signs", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    words = {}
    for n, (label, (q, vid, fr, aspect)) in enumerate(sorted(best.items())):
        if q < 0.4:
            continue
        frames = smooth_resample(fr)
        if not any(frames):
            continue
        fid = f"{n:04d}"
        with open(os.path.join(args.out, fid + ".json"), "w", encoding="utf-8") as f:
            json.dump({"text": label, "aspect": round(aspect, 4), "source": vid, "frames": encode(frames)}, f,
                      ensure_ascii=False, separators=(",", ":"))
        words[label] = fid
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "language": "rsl", "dataset": "Slovo (SberDevices)",
                   "license": "Slovo public licence (CC BY-SA 4.0 variant)", "frames": N_FRAMES,
                   "words": dict(sorted(words.items()))}, f, ensure_ascii=False, indent=0)
    print(f"wrote {len(words)} signs to {args.out}", file=sys.stderr)
    if len(words) < 100 and not args.limit:
        sys.exit("too few signs — check the dataset format")


if __name__ == "__main__":
    main()
