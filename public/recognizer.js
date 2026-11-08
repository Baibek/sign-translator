// recognizer.js — pure gesture-recognition helpers (no DOM).
// Works in the browser and in Node (unit tests import it directly).
//
// Per-frame feature layout (FEATURE_SIZE = 132):
//   [ Left slot (66) | Right slot (66) ]
//   slot = 21 landmarks × (x,y,z) relative to wrist, scaled by palm size (63)
//        + wrist position relative to frame center (2)
//        + presence flag (1)
// A missing hand is all zeros (presence = 0).
//
// Handedness note: MediaPipe reports handedness assuming a mirrored (selfie)
// image. We use its label as-is; what matters is consistency between
// training and recognition, which is guaranteed because both use this code.

export const LANDMARKS = 21;
export const SLOT_SIZE = LANDMARKS * 3 + 2 + 1; // 66
export const FEATURE_SIZE = SLOT_SIZE * 2; // 132
export const DEFAULT_RESAMPLE = 24;

const WRIST = 0;
const MIDDLE_MCP = 9;
const WRIST_POS_WEIGHT = 2; // wrist-in-frame coords are in [-0.5,0.5]; boost a bit

/** Read handedness label ("Left"/"Right") for hand i from a HandLandmarker result. */
function handLabel(result, i) {
  const h = (result.handedness || result.handednesses || [])[i];
  const cat = Array.isArray(h) ? h[0] : h;
  const name = cat && (cat.categoryName || cat.displayName || cat.label);
  return name === "Left" || name === "Right" ? name : null;
}

/**
 * Convert a HandLandmarker result ({landmarks, handedness}) to a feature vector.
 * @returns {Float32Array} length FEATURE_SIZE
 */
export function frameFeatures(result) {
  const out = new Float32Array(FEATURE_SIZE);
  const hands = (result && result.landmarks) || [];
  const used = [false, false];

  for (let i = 0; i < hands.length && i < 2; i++) {
    const lm = hands[i];
    if (!lm || lm.length < LANDMARKS) continue;
    const label = handLabel(result, i);
    let slot = label === "Left" ? 0 : label === "Right" ? 1 : i;
    if (used[slot]) slot = 1 - slot; // both hands labelled the same: use the free slot
    if (used[slot]) continue;
    used[slot] = true;

    const w = lm[WRIST];
    const m = lm[MIDDLE_MCP];
    const palm = Math.hypot(m.x - w.x, m.y - w.y, (m.z || 0) - (w.z || 0)) || 1e-6;
    const base = slot * SLOT_SIZE;
    for (let j = 0; j < LANDMARKS; j++) {
      const p = lm[j];
      out[base + j * 3] = (p.x - w.x) / palm;
      out[base + j * 3 + 1] = (p.y - w.y) / palm;
      out[base + j * 3 + 2] = ((p.z || 0) - (w.z || 0)) / palm;
    }
    out[base + 63] = (w.x - 0.5) * WRIST_POS_WEIGHT;
    out[base + 64] = (w.y - 0.5) * WRIST_POS_WEIGHT;
    out[base + 65] = 1;
  }
  return out;
}

/** Number of hands present in a feature vector. */
export function handsInFeatures(f) {
  return (f[SLOT_SIZE - 1] > 0 ? 1 : 0) + (f[FEATURE_SIZE - 1] > 0 ? 1 : 0);
}

/** Euclidean distance between two frames. */
export function frameDistance(a, b) {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Linearly resample a sequence of frames to exactly n frames. */
export function resample(seq, n = DEFAULT_RESAMPLE) {
  if (!seq || seq.length === 0) return [];
  const dim = seq[0].length;
  const out = [];
  if (seq.length === 1) {
    for (let i = 0; i < n; i++) out.push(Float32Array.from(seq[0]));
    return out;
  }
  for (let i = 0; i < n; i++) {
    const pos = n === 1 ? 0 : (i * (seq.length - 1)) / (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, seq.length - 1);
    const t = pos - lo;
    const a = seq[lo];
    const b = seq[hi];
    const f = new Float32Array(dim);
    for (let d = 0; d < dim; d++) f[d] = a[d] + (b[d] - a[d]) * t;
    out.push(f);
  }
  return out;
}

/**
 * Dynamic Time Warping distance with a Sakoe-Chiba band.
 * Returned value is normalised by path length (≈ average per-frame distance),
 * so it is comparable across sequence lengths.
 * @param {ArrayLike<number>[]} a
 * @param {ArrayLike<number>[]} b
 * @param {{band?: number}} [opts] band as fraction of the longer length (default 0.2)
 */
export function dtwDistance(a, b, { band = 0.2 } = {}) {
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return Infinity;
  const w = Math.max(Math.abs(n - m), Math.ceil(band * Math.max(n, m)), 1);
  // cost[i][j] and path length steps[i][j], rolling rows
  let prevC = new Float64Array(m + 1).fill(Infinity);
  let prevS = new Float64Array(m + 1);
  let curC = new Float64Array(m + 1);
  let curS = new Float64Array(m + 1);
  prevC[0] = 0;
  for (let i = 1; i <= n; i++) {
    curC.fill(Infinity);
    curS.fill(0);
    const jFrom = Math.max(1, i - w);
    const jTo = Math.min(m, i + w);
    for (let j = jFrom; j <= jTo; j++) {
      const d = frameDistance(a[i - 1], b[j - 1]);
      // pick the cheapest predecessor: match, insertion, deletion
      let c = prevC[j - 1], s = prevS[j - 1];
      if (prevC[j] < c) { c = prevC[j]; s = prevS[j]; }
      if (curC[j - 1] < c) { c = curC[j - 1]; s = curS[j - 1]; }
      curC[j] = c + d;
      curS[j] = s + 1;
    }
    [prevC, curC] = [curC, prevC];
    [prevS, curS] = [curS, prevS];
  }
  const total = prevC[m];
  return Number.isFinite(total) ? total / prevS[m] : Infinity;
}

/**
 * k-nearest-neighbour classifier over DTW distance.
 * Examples are stored resampled to `n` frames.
 */
export class GestureClassifier {
  /** @param {{n?: number, threshold?: number, band?: number}} [opts] */
  constructor({ n = DEFAULT_RESAMPLE, threshold = 2.5, band = 0.2 } = {}) {
    this.n = n;
    this.threshold = threshold; // reject if best distance is above this
    this.band = band;
    /** @type {Map<string, Float32Array[][]>} */
    this.examples = new Map();
  }

  /** Add one example sequence (array of feature frames) for a label. */
  add(label, seq) {
    label = String(label).trim();
    if (!label || !seq || seq.length === 0) return false;
    const list = this.examples.get(label) || [];
    list.push(resample(seq, this.n));
    this.examples.set(label, list);
    return true;
  }

  /** Remove a label (all examples) or a single example by index. */
  remove(label, index) {
    if (index === undefined) return this.examples.delete(label);
    const list = this.examples.get(label);
    if (!list || index < 0 || index >= list.length) return false;
    list.splice(index, 1);
    if (list.length === 0) this.examples.delete(label);
    return true;
  }

  clear() {
    this.examples.clear();
  }

  labels() {
    return [...this.examples.keys()];
  }

  count(label) {
    return (this.examples.get(label) || []).length;
  }

  get size() {
    let s = 0;
    for (const l of this.examples.values()) s += l.length;
    return s;
  }

  /**
   * Classify a sequence.
   * @returns {{label: string, distance: number, confidence: number, accepted: boolean} | null}
   *   null when there are no examples or the best match is above the threshold.
   */
  classify(seq, { k = 3, threshold = this.threshold } = {}) {
    if (!seq || seq.length === 0 || this.size === 0) return null;
    const q = resample(seq, this.n);
    const scored = [];
    for (const [label, list] of this.examples) {
      for (const ex of list) scored.push({ label, d: dtwDistance(q, ex, { band: this.band }) });
    }
    scored.sort((x, y) => x.d - y.d);

    // Distance-weighted vote among the k nearest examples.
    const votes = new Map();
    for (const { label, d } of scored.slice(0, Math.max(1, k))) {
      votes.set(label, (votes.get(label) || 0) + 1 / (d + 1e-6));
    }
    let label = null, best = -1;
    for (const [l, v] of votes) if (v > best) { best = v; label = l; }

    const distance = scored.find((s) => s.label === label).d;
    if (!(distance <= threshold)) return null;

    // Confidence: closeness to the examples + margin to the nearest other label.
    const closeness = Math.max(0, 1 - distance / threshold);
    const other = scored.find((s) => s.label !== label);
    const margin = other ? Math.max(0, Math.min(1, 1 - distance / (other.d + 1e-6))) : closeness;
    const confidence = Math.max(0, Math.min(1, 0.5 * closeness + 0.5 * margin));
    return { label, distance, confidence, accepted: true };
  }

  /** Serializable form (numbers rounded to keep localStorage small). */
  toJSON() {
    const examples = {};
    for (const [label, list] of this.examples) {
      examples[label] = list.map((seq) => seq.map((f) => Array.from(f, (v) => Math.round(v * 1000) / 1000)));
    }
    return { version: 1, n: this.n, threshold: this.threshold, band: this.band, examples };
  }

  static fromJSON(data) {
    const obj = typeof data === "string" ? JSON.parse(data) : data;
    if (!obj || typeof obj !== "object" || typeof obj.examples !== "object") {
      throw new Error("Invalid gesture data");
    }
    const c = new GestureClassifier({
      n: obj.n || DEFAULT_RESAMPLE,
      threshold: typeof obj.threshold === "number" ? obj.threshold : undefined,
      band: typeof obj.band === "number" ? obj.band : undefined,
    });
    for (const [label, list] of Object.entries(obj.examples)) {
      if (!Array.isArray(list)) continue;
      for (const seq of list) {
        if (Array.isArray(seq) && seq.length) c.add(label, seq.map((f) => Float32Array.from(f)));
      }
    }
    return c;
  }
}

/**
 * Splits a continuous stream of frames into gesture segments.
 * Timestamp-driven: call push(features, timestampMs, handsPresent) per frame;
 * it returns a segment object when a gesture has just ended, else null.
 *
 * A gesture starts when a hand appears or motion exceeds the threshold, and
 * ends after `stillMs` of stillness or when hands disappear for `absentMs`.
 * Segments shorter than `minMs` are dropped; at `maxMs` a segment is forced out.
 */
export class Segmenter {
  constructor({
    motionThreshold = 0.5, // smoothed per-frame (≈30 fps) feature-space speed
    stillMs = 500,
    minMs = 300,
    maxMs = 4000,
    absentMs = 200,
    minFrames = 4,
  } = {}) {
    Object.assign(this, { motionThreshold, stillMs, minMs, maxMs, absentMs, minFrames });
    this.reset();
  }

  reset() {
    this.active = false;
    this.frames = [];
    this.times = [];
    this.startMs = 0;
    this.lastMotionMs = 0;
    this.lastHandsMs = 0;
    this.prev = null;
    this.prevMs = 0;
    this.prevHands = false;
    this.frameIdx = 0;
    this.startIdx = 0;
    this.energy = 0;
  }

  /** Motion between consecutive frames, normalised to a 33 ms frame interval. */
  _motion(f, t) {
    if (!this.prev) return 0;
    const dt = Math.max(1, t - this.prevMs);
    return (frameDistance(f, this.prev) * 33.3) / dt;
  }

  push(features, timestampMs, handsPresent) {
    const t = timestampMs;
    const hands = !!handsPresent;
    // Motion is only meaningful when hands are visible in both frames.
    const motion = hands && this.prevHands ? this._motion(features, t) : 0;
    // Smoothed motion energy (EMA) — suppresses single-frame landmark jitter.
    this.energy = hands ? 0.5 * this.energy + 0.5 * motion : 0;
    const moving = this.energy > this.motionThreshold;
    const appeared = hands && !this.prevHands;
    let segment = null;
    const idx = this.frameIdx++;

    if (hands) this.lastHandsMs = t;

    if (!this.active) {
      if (hands && (appeared || moving)) {
        this.active = true;
        this.startMs = t;
        this.startIdx = idx;
        this.lastMotionMs = t;
        this.frames = [features];
        this.times = [t];
      }
    } else {
      const tooLong = t - this.startMs > this.maxMs;
      if (hands && !tooLong) {
        this.frames.push(features);
        this.times.push(t);
        if (moving) this.lastMotionMs = t;
      }
      const absent = !hands && t - this.lastHandsMs >= this.absentMs;
      const still = t - this.lastMotionMs >= this.stillMs;
      if (absent || still || tooLong) segment = this._finish(t, tooLong);
    }

    this.prev = features;
    this.prevMs = t;
    this.prevHands = hands;
    return segment;
  }

  /** Force the current gesture (if any) to end now. */
  flush(timestampMs = this.prevMs) {
    return this.active ? this._finish(timestampMs, true) : null;
  }

  _finish(t, forced) {
    this.active = false;
    let frames = this.frames;
    let times = this.times;
    this.frames = [];
    this.times = [];
    if (frames.length === 0) return null;

    // Trim the trailing still part (keep ~100 ms) unless that makes it too short
    // (static handshapes have no motion at all — keep them whole).
    if (!forced) {
      const cut = this.lastMotionMs + 100;
      let k = times.length;
      while (k > 0 && times[k - 1] > cut) k--;
      if (k >= this.minFrames && times[k - 1] - times[0] >= this.minMs) {
        frames = frames.slice(0, k);
        times = times.slice(0, k);
      }
    }
    const startMs = times[0];
    const endMs = times[times.length - 1];
    if (endMs - startMs < this.minMs || frames.length < this.minFrames) return null;
    return { frames, times, startMs, endMs, frameIdx: this.startIdx };
  }
}
