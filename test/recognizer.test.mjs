import { test } from "node:test";
import assert from "node:assert/strict";
import {
  FEATURE_SIZE,
  SLOT_SIZE,
  frameFeatures,
  handsInFeatures,
  resample,
  dtwDistance,
  GestureClassifier,
  Segmenter,
} from "../public/recognizer.js";

// ---- synthetic data helpers -------------------------------------------------

// Deterministic PRNG so tests are stable.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

// Hand shape: 21 points relative to wrist (palm size ≈ 0.1 in image units).
function handShape(kind) {
  const pts = [{ x: 0, y: 0, z: 0 }];
  for (let finger = 0; finger < 5; finger++) {
    const angle = -0.9 + finger * 0.45;
    for (let joint = 1; joint <= 4; joint++) {
      let r = 0.04 * joint + 0.06;
      if (kind === "fist" && joint > 1) r = 0.07 - 0.01 * joint; // curled fingers
      if (kind === "point" && finger !== 1 && joint > 1) r = 0.06;
      pts.push({ x: Math.sin(angle) * r, y: -Math.cos(angle) * r, z: -0.01 * joint });
    }
  }
  // make wrist→middle MCP (index 9) ≈ 0.1
  return pts;
}

function hand(shape, wx, wy, rand, noise = 0.003) {
  return shape.map((p) => ({
    x: wx + p.x + (rand() - 0.5) * noise,
    y: wy + p.y + (rand() - 0.5) * noise,
    z: p.z,
  }));
}

function result(hands) {
  return {
    landmarks: hands.map((h) => h.lm),
    handedness: hands.map((h) => [{ categoryName: h.side, score: 0.99 }]),
  };
}

// A gesture is a function of t∈[0,1] → list of hands.
const gestures = {
  // open right hand sweeping left→right
  wave: (t) => [{ side: "Right", shape: "open", x: 0.3 + 0.4 * t, y: 0.5 }],
  // right fist moving top→bottom
  knock: (t) => [{ side: "Right", shape: "fist", x: 0.5, y: 0.3 + 0.4 * t }],
  // both hands pointing, circling (unrelated to the others)
  circle: (t) => [
    { side: "Left", shape: "point", x: 0.3 + 0.1 * Math.cos(6.28 * t), y: 0.5 + 0.1 * Math.sin(6.28 * t) },
    { side: "Right", shape: "point", x: 0.7 - 0.1 * Math.cos(6.28 * t), y: 0.5 + 0.1 * Math.sin(6.28 * t) },
  ],
};

function sequence(name, { frames = 30, seed = 1, speedJitter = 0 } = {}) {
  const rand = rng(seed);
  const warp = 1 + (rand() - 0.5) * speedJitter; // time warp exponent
  const seq = [];
  for (let i = 0; i < frames; i++) {
    const t = Math.pow(i / (frames - 1), warp);
    const hs = gestures[name](t).map((h) => ({
      side: h.side,
      lm: hand(handShape(h.shape), h.x, h.y, rand),
    }));
    seq.push(frameFeatures(result(hs)));
  }
  return seq;
}

// ---- tests ------------------------------------------------------------------

test("frameFeatures: layout, slots and normalisation", () => {
  const rand = rng(7);
  const lm = hand(handShape("open"), 0.6, 0.4, rand, 0);
  const f = frameFeatures(result([{ side: "Right", lm }]));
  assert.equal(f.length, FEATURE_SIZE);
  // Left slot empty, Right slot present
  assert.equal(f[SLOT_SIZE - 1], 0);
  assert.equal(f[FEATURE_SIZE - 1], 1);
  assert.equal(handsInFeatures(f), 1);
  // wrist at origin, middle MCP at distance 1 after palm scaling
  const b = SLOT_SIZE;
  assert.equal(f[b], 0);
  assert.equal(f[b + 1], 0);
  const mx = f[b + 27], my = f[b + 28], mz = f[b + 29];
  assert.ok(Math.abs(Math.hypot(mx, my, mz) - 1) < 1e-5);
  // empty result → zeros
  assert.ok(frameFeatures({ landmarks: [], handedness: [] }).every((v) => v === 0));
  assert.equal(handsInFeatures(frameFeatures(null)), 0);
});

test("resample produces n frames and preserves endpoints", () => {
  const seq = [[0, 0], [1, 2], [2, 4]];
  const r = resample(seq, 5);
  assert.equal(r.length, 5);
  assert.deepEqual(Array.from(r[0]), [0, 0]);
  assert.deepEqual(Array.from(r[4]), [2, 4]);
  assert.deepEqual(Array.from(r[2]), [1, 2]);
  assert.equal(resample([[3]], 4).length, 4);
  assert.deepEqual(resample([], 4), []);
});

test("dtwDistance: zero for identical, tolerant to time warping", () => {
  const a = sequence("wave", { frames: 24, seed: 2 });
  assert.equal(dtwDistance(a, a), 0);
  const slow = resample(a, 40);
  const other = sequence("knock", { frames: 24, seed: 2 });
  assert.ok(dtwDistance(a, slow) < dtwDistance(a, other) / 5);
  assert.equal(dtwDistance([], a), Infinity);
});

test("classifier distinguishes gestures and rejects unrelated ones", () => {
  const c = new GestureClassifier();
  for (let s = 0; s < 3; s++) {
    c.add("привет", sequence("wave", { frames: 25 + s * 5, seed: 10 + s, speedJitter: 0.4 }));
    c.add("стучать", sequence("knock", { frames: 25 + s * 5, seed: 20 + s, speedJitter: 0.4 }));
  }
  assert.deepEqual(c.labels().sort(), ["привет", "стучать"]);
  assert.equal(c.count("привет"), 3);

  const w = c.classify(sequence("wave", { frames: 33, seed: 99, speedJitter: 0.4 }));
  assert.ok(w, "wave should be recognised");
  assert.equal(w.label, "привет");
  assert.ok(w.confidence > 0.5, `confidence ${w.confidence}`);

  const k = c.classify(sequence("knock", { frames: 21, seed: 98, speedJitter: 0.4 }));
  assert.ok(k);
  assert.equal(k.label, "стучать");

  const u = c.classify(sequence("circle", { frames: 30, seed: 97 }));
  assert.equal(u, null, `unrelated gesture should be rejected, got ${JSON.stringify(u)}`);

  // remove
  assert.ok(c.remove("стучать"));
  assert.deepEqual(c.labels(), ["привет"]);
  assert.equal(new GestureClassifier().classify(sequence("wave")), null);
});

test("classifier JSON round-trip", () => {
  const c = new GestureClassifier({ threshold: 3 });
  c.add("да", sequence("wave", { seed: 1 }));
  c.add("да", sequence("wave", { seed: 2 }));
  c.add("нет", sequence("knock", { seed: 3 }));
  const json = JSON.stringify(c);
  const d = GestureClassifier.fromJSON(json);
  assert.deepEqual(d.labels(), c.labels());
  assert.equal(d.count("да"), 2);
  assert.equal(d.threshold, 3);
  const q = sequence("knock", { seed: 4 });
  assert.equal(d.classify(q).label, c.classify(q).label);
  assert.ok(Math.abs(d.classify(q).distance - c.classify(q).distance) < 0.01);
  assert.throws(() => GestureClassifier.fromJSON({ foo: 1 }));
});

test("segmenter emits one segment for a movement burst", () => {
  const seg = new Segmenter();
  const rand = rng(5);
  const empty = frameFeatures({ landmarks: [] });
  const segments = [];
  let t = 0;
  const feed = (f, hands) => {
    const s = seg.push(f, t, hands);
    if (s) segments.push(s);
    t += 33;
  };
  const still = (x) => frameFeatures(result([{ side: "Right", lm: hand(handShape("open"), x, 0.5, rand, 0.001) }]));

  for (let i = 0; i < 15; i++) feed(empty, false); // nothing
  // hand appears and stays still (static pose) — then moves after a short while
  for (let i = 0; i < 5; i++) feed(still(0.3), true);
  for (let i = 0; i < 25; i++) {
    const shape = i % 2 ? "open" : "fist"; // strong motion: shape alternates + translation
    const lm = hand(handShape(shape), 0.3 + 0.015 * i, 0.5, rand);
    feed(frameFeatures(result([{ side: "Right", lm }])), true);
  }
  const burstEnd = t - 33;
  for (let i = 0; i < 30; i++) feed(still(0.66), true); // 1 s of stillness
  assert.equal(segments.length, 1, `expected 1 segment, got ${segments.length}`);
  const s = segments[0];
  assert.ok(s.startMs >= 15 * 33 && s.startMs <= 20 * 33, `start ${s.startMs}`);
  assert.ok(s.endMs >= burstEnd - 100 && s.endMs <= burstEnd + 300, `end ${s.endMs} vs ${burstEnd}`);
  assert.ok(s.endMs - s.startMs >= 300);
  assert.equal(s.frames.length, s.times.length);
  assert.equal(s.frameIdx, 15);

  // hands disappear → no new segment from silence
  for (let i = 0; i < 30; i++) feed(empty, false);
  assert.equal(segments.length, 1);
});

test("segmenter: short blips are dropped, long gestures are capped", () => {
  const seg = new Segmenter();
  const rand = rng(6);
  let t = 0;
  const out = [];
  const moving = (i) => frameFeatures(result([{ side: "Left", lm: hand(handShape(i % 2 ? "open" : "fist"), 0.5, 0.5, rand) }]));
  // 150 ms blip then hands vanish
  for (let i = 0; i < 5; i++) { const s = seg.push(moving(i), t, true); if (s) out.push(s); t += 33; }
  for (let i = 0; i < 20; i++) { const s = seg.push(frameFeatures(null), t, false); if (s) out.push(s); t += 33; }
  assert.equal(out.length, 0);
  // 6 s of continuous motion → capped at 4 s
  for (let i = 0; i < 182; i++) { const s = seg.push(moving(i), t, true); if (s) out.push(s); t += 33; }
  assert.ok(out.length >= 1);
  assert.ok(out[0].endMs - out[0].startMs <= 4000);
  assert.ok(out[0].endMs - out[0].startMs >= 3900);
});
