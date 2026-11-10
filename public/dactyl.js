// Built-in fingerspelling (дактиль) for text → sign.
// A procedural 3D hand: each letter is a handshape spec → 21 MediaPipe-style landmarks.
// IMPORTANT: shapes are APPROXIMATE, compiled from public descriptions of the Russian dactyl alphabet,
// which disagree in places. They must be checked by a sign-language expert (e.g. against «е-Ымдау»).

// Finger flexion presets: [MCP, PIP, DIP] in degrees (toward the palm)
const FLEX = {
  ext: [0, 0, 0],      // straight
  curl: [88, 100, 70], // fist
  bend: [88, 0, 0],    // bent 90° at the knuckle, finger straight
  hook: [10, 95, 75],  // hooked at the middle joints
  half: [35, 45, 30],  // rounded (С / О)
};

// Hand geometry in palm units (wrist → middle knuckle ≈ 1). x: toward thumb, y: down (fingers point to −y), z: toward viewer
const MCP = { i: [0.3, -0.95, 0], m: [0.08, -1.0, 0], r: [-0.14, -0.93, 0], l: [-0.33, -0.8, 0] };
const SEG = { i: [0.42, 0.26, 0.2], m: [0.47, 0.29, 0.22], r: [0.44, 0.27, 0.21], l: [0.34, 0.21, 0.18] };
const SPREAD = { i: 9, m: 0, r: -8, l: -17 }; // degrees when fingers are spread
const CMC = [0.28, -0.18, 0.05];
const ORDER = ["i", "m", "r", "l"];
const BASE_IDX = { i: 5, m: 9, r: 13, l: 17 };

const rad = (d) => (d * Math.PI) / 180;
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/**
 * Letter specs. f: finger states for index/middle/ring/little (default curl), t: thumb
 * ('side' along the index, 'over' across the fingers, 'out' sideways, 'half' rounded, or a list of fingers its tip touches),
 * spread: fingers apart, o: orientation ('up','down','side','tiltL','tiltR'), mv: movement ('shake','down','twist','draw','cross').
 */
const SPEC = {
  "а": { t: "side" },
  "б": { f: { i: "ext", m: "hook" }, t: "over" },
  "в": { f: { i: "ext", m: "ext", r: "ext", l: "ext" }, t: "over" },
  "г": { f: { i: "ext" }, t: "out", o: "down" },
  "д": { f: { i: "ext", m: "ext" }, t: "over", mv: "draw" },
  "е": { f: { i: "hook", m: "hook", r: "hook", l: "hook" }, t: ["i", "m"] },
  "ё": { f: { i: "hook", m: "hook", r: "hook", l: "hook" }, t: ["i", "m"], mv: "shake" },
  "ж": { f: { i: "bend", m: "bend", r: "bend", l: "bend" }, t: ["i", "m", "r"] },
  "з": { f: { i: "ext" }, t: "over", mv: "draw" },
  "и": { f: { r: "ext", l: "ext" }, t: "over" },
  "й": { f: { r: "ext", l: "ext" }, t: "over", mv: "twist" },
  "к": { f: { i: "ext", m: "ext" }, t: "out", spread: true, mv: "down" },
  "л": { f: { i: "ext", m: "ext" }, t: "over", spread: true, o: "down" },
  "м": { f: { i: "ext", m: "ext", r: "ext" }, t: "over", spread: true, o: "down" },
  "н": { f: { i: "ext", m: "ext", r: "half", l: "ext" }, t: ["r"] },
  "о": { f: { i: "half", m: "half", r: "half", l: "half" }, t: ["i", "m"] },
  "п": { f: { i: "ext", m: "ext" }, t: "over", o: "down" },
  "р": { f: { i: "ext", m: "half", r: "ext", l: "ext" }, t: ["m"] },
  "с": { f: { i: "half", m: "half", r: "half", l: "half" }, t: "half" },
  "т": { f: { i: "ext", m: "ext", r: "ext" }, t: "over", o: "down" },
  "у": { f: { l: "ext" }, t: "out" },
  "ф": { f: { i: "half", m: "ext", r: "ext", l: "ext" }, t: ["i"], spread: true },
  "х": { f: { i: "hook" }, t: "over" },
  "ц": { f: { i: "ext", m: "ext" }, t: "over", mv: "down" },
  "ч": { f: { i: "ext", m: "half", r: "half" }, t: ["m", "r"] },
  "ш": { f: { i: "ext", m: "ext", r: "ext" }, t: "over", spread: true },
  "щ": { f: { i: "ext", m: "ext", r: "ext" }, t: "over", spread: true, mv: "down" },
  "ъ": { f: { i: "ext" }, t: "out", o: "tiltL", mv: "shake" },
  "ы": { f: { i: "ext", l: "ext" }, t: "over" },
  "ь": { f: { i: "ext" }, t: "out", o: "tiltR" },
  "э": { f: { i: "half" }, t: "half" },
  "ю": { f: { i: "half", l: "ext" }, t: ["i"] },
  "я": { f: { i: "ext", m: "ext" }, t: "over", mv: "cross" },
};

// Kazakh letters without a verified handshape here: shown with the closest Russian base letter
export const KAZAKH_FALLBACK = { "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и" };

/** Normalized letter key that has a built-in shape, or null. */
export function dactylKey(ch) {
  const c = String(ch).toLocaleLowerCase("ru");
  if (SPEC[c]) return c;
  if (KAZAKH_FALLBACK[c]) return KAZAKH_FALLBACK[c];
  return null;
}
export const hasDactyl = (ch) => dactylKey(ch) !== null;

function fingerChain(name, state, spread) {
  const [a1, a2, a3] = FLEX[state] || FLEX.curl;
  const yaw = rad(spread ? SPREAD[name] : SPREAD[name] * 0.25);
  const pts = [MCP[name]];
  let p = MCP[name];
  let ang = 0;
  [a1, a2, a3].forEach((a, k) => {
    ang += rad(a);
    const dir = [Math.sin(yaw) * Math.cos(ang), -Math.cos(yaw) * Math.cos(ang), Math.sin(ang)];
    p = add(p, mul(dir, SEG[name][k]));
    pts.push(p);
  });
  return pts; // MCP, PIP, DIP, TIP
}

function thumbChain(t, fingerTips) {
  let tip;
  let bow = [0.12, 0, 0.12];
  if (Array.isArray(t)) {
    tip = t.map((n) => fingerTips[n]).reduce((s, p) => add(s, p), [0, 0, 0]);
    tip = add(mul(tip, 1 / t.length), [0.03, 0.03, 0.03]);
  } else if (t === "side") { tip = [0.45, -0.95, 0.02]; bow = [0.06, 0, 0]; }
  else if (t === "out") { tip = [0.95, -0.65, 0.05]; bow = [0.05, 0.05, 0]; }
  else if (t === "half") { tip = [0.42, -1.1, 0.5]; bow = [0.25, 0, 0.1]; }
  else /* over */ { tip = [-0.05, -0.62, 0.42]; bow = [0.1, 0.05, 0.15]; }
  const pts = [CMC];
  for (const k of [0.36, 0.7, 1]) {
    const bulge = Math.sin(k * Math.PI) ;
    pts.push(add(lerp3(CMC, tip, k), mul(bow, bulge)));
  }
  return pts; // CMC, MCP, IP, TIP
}

function orient(p, o, extraYaw) {
  let [x, y, z] = p;
  const rotZ = { down: 180, side: 90, tiltL: -28, tiltR: 28 }[o] || 0;
  if (rotZ) {
    const r = rad(rotZ), c = Math.cos(r), s = Math.sin(r);
    [x, y] = [x * c - y * s, x * s + y * c];
  }
  // slight 3D turn so curled fingers stay visible
  const r = rad(-18 + extraYaw), c = Math.cos(r), s = Math.sin(r);
  [x, z] = [x * c + z * s, -x * s + z * c];
  return [x, y, z];
}

/** 21 landmarks (palm units, wrist at origin) for a letter at animation phase t∈[0,1]. */
export function letterPose(ch, t = 0) {
  const key = dactylKey(ch);
  const spec = SPEC[key] || SPEC["а"];
  const f = { i: "curl", m: "curl", r: "curl", l: "curl", ...(spec.f || {}) };
  const chains = {};
  for (const n of ORDER) chains[n] = fingerChain(n, f[n], spec.spread);
  if (spec.mv === "cross") { // Я: middle finger crosses over the index
    chains.m = chains.m.map((p, k) => (k ? add(p, [0.12 * k, 0, 0.05]) : p));
  }
  const tips = Object.fromEntries(ORDER.map((n) => [n, chains[n][3]]));
  const thumb = thumbChain(spec.t || "over", tips);

  const pts = new Array(21);
  pts[0] = [0, 0, 0];
  thumb.forEach((p, k) => (pts[1 + k] = p));
  for (const n of ORDER) chains[n].forEach((p, k) => (pts[BASE_IDX[n] + k] = p));

  // movement
  let yaw = 0, dx = 0, dy = 0;
  const w = Math.sin(t * Math.PI * 2);
  if (spec.mv === "shake") dx = 0.12 * w;
  if (spec.mv === "twist") yaw = 35 * w;
  if (spec.mv === "down") dy = 0.35 * Math.sin(t * Math.PI / 2);
  if (spec.mv === "draw") { dx = 0.18 * Math.sin(t * Math.PI * 2); dy = 0.25 * t; }
  return pts.map((p) => add(orient(p, spec.o, yaw), [dx, dy, 0]));
}

/** Pose → canvas-normalized hand {slot, pts:[[x,y,z]…]} (right hand, slot 1). */
export function letterHand(ch, t = 0, { cx = 0.5, cy = 0.8, size = 0.3, aspect = 1.6 } = {}) {
  const pose = letterPose(ch, t);
  // fingers pointing down: move the wrist up so the hand stays on screen
  const spec = SPEC[dactylKey(ch)] || {};
  const wy = spec.o === "down" ? 0.22 : cy;
  return { slot: 1, pts: pose.map(([x, y, z]) => [cx + (x * size) / aspect, wy + y * size, z * size]) };
}

/** Frames for one letter clip: a hold with its movement. */
export function letterFrames(ch, n = 16, opts) {
  const out = [];
  for (let k = 0; k < n; k++) out.push([letterHand(ch, k / (n - 1), opts)]);
  return out;
}

export const DACTYL_LETTERS = Object.keys(SPEC);
