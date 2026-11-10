// Text → sign: turns text into a sequence of gestures and animates them as a drawn hand.
// Recorded gestures (from «Жаттығу») are used first; other words are fingerspelled with the
// built-in dactyl alphabet (dactyl.js). planSigns() is pure (testable in Node).

import { LANDMARKS, SLOT_SIZE } from "./recognizer.js";
import { hasDactyl, dactylKey, letterFrames } from "./dactyl.js";

// MediaPipe hand skeleton (pairs of landmark indices)
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const PALM = [0, 1, 5, 9, 13, 17];

const WRIST_POS_WEIGHT = 2; // must match recognizer.js

const norm = (s) => s.toLocaleLowerCase("ru").replace(/ё/g, "е").trim();

/** Split text into words (letters, digits, hyphen) in any Cyrillic/Latin script. */
export function tokenize(text) {
  return String(text).match(/[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*/gu) || [];
}

/**
 * Plan how to show `text`.
 * Recorded labels match first (multi-word labels greedily, longest first). Other words are fingerspelled:
 * a recorded letter if there is one, else the built-in dactyl shape. Characters with neither are "missing".
 * @returns {{type:"sign"|"letter"|"missing", source?:"recorded"|"builtin", label?:string, text:string, word?:string}[]}
 */
export function planSigns(text, labels, { builtin = true } = {}) {
  const byKey = new Map();
  let maxWords = 1;
  for (const label of labels) {
    const key = tokenize(label).map(norm).join(" ");
    if (!key) continue;
    if (!byKey.has(key)) byKey.set(key, label);
    maxWords = Math.max(maxWords, key.split(" ").length);
  }
  const words = tokenize(text);
  const keys = words.map(norm);
  const plan = [];
  for (let i = 0; i < words.length; ) {
    let matched = false;
    for (let n = Math.min(maxWords, words.length - i); n >= 1; n--) {
      const key = keys.slice(i, i + n).join(" ");
      if (byKey.has(key)) {
        plan.push({ type: "sign", source: "recorded", label: byKey.get(key), text: words.slice(i, i + n).join(" ") });
        i += n;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    const word = words[i];
    const letters = [...word.toLocaleLowerCase("ru")];
    const canSpell = letters.every((ch) => byKey.has(norm(ch)) || (builtin && hasDactyl(ch)));
    if (letters.length && canSpell) {
      for (const ch of letters) {
        if (byKey.has(norm(ch))) plan.push({ type: "letter", source: "recorded", label: byKey.get(norm(ch)), text: ch, word });
        else plan.push({ type: "letter", source: "builtin", text: ch, word });
      }
    } else {
      plan.push({ type: "missing", text: word });
    }
    i++;
  }
  return plan;
}

/** Rebuild drawable hands (0..1 canvas coords, with depth) from one recorded feature frame. */
export function framePoints(f, { scale = 0.14, mirror = true } = {}) {
  const hands = [];
  for (let slot = 0; slot < 2; slot++) {
    const base = slot * SLOT_SIZE;
    if (!f[base + 65]) continue;
    const wx = f[base + 63] / WRIST_POS_WEIGHT + 0.5;
    const wy = f[base + 64] / WRIST_POS_WEIGHT + 0.5;
    const pts = [];
    for (let j = 0; j < LANDMARKS; j++) {
      let x = wx + f[base + j * 3] * scale;
      const y = wy + f[base + j * 3 + 1] * scale;
      if (mirror) x = 1 - x; // match the mirrored camera preview
      pts.push([x, y, -f[base + j * 3 + 2] * scale]); // MediaPipe z: smaller = closer to camera
    }
    hands.push({ slot, pts });
  }
  return hands;
}

/** Build player clips from a plan. `getFrames(label)` returns a recorded feature sequence or null. */
export function buildClips(plan, getFrames, { aspect = 1.6 } = {}) {
  const clips = [];
  plan.forEach((p, i) => {
    if (p.type === "missing") return;
    const caption = p.type === "letter" ? p.text.toLocaleUpperCase("ru") : p.text;
    if (p.source === "builtin") {
      clips.push({ i, frames: letterFrames(dactylKey(p.text), 16, { aspect }), caption, ms: 650, builtin: true });
      return;
    }
    const seq = getFrames(p.label);
    if (!seq || !seq.length) return;
    clips.push({ i, frames: seq.map((f) => framePoints(f)), caption, ms: p.type === "letter" ? 700 : 1400 });
  });
  return clips;
}

function lerpHands(a, b, t) {
  return b.map((hb) => {
    const ha = a.find((h) => h.slot === hb.slot);
    if (!ha) return hb;
    return { slot: hb.slot, pts: hb.pts.map((p, j) => p.map((v, k) => ha.pts[j][k] + (v - ha.pts[j][k]) * t)) };
  });
}

/** Animates clips on a canvas, drawing a filled, shaded hand. */
export class SignPlayer {
  constructor(canvas, { colors } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.colors = colors || (() => ({ bg: "#eef0f4", skin: "#e8b48f", skinDark: "#b97d57", line: "#7a4a2c", text: "#16181d", muted: "#5d6573" }));
    this.raf = 0;
    this.playing = false;
    this.last = null;
  }

  get aspect() {
    return this.canvas.width / this.canvas.height;
  }

  /**
   * @param {{frames:{slot:number,pts:number[][]}[][], caption:string, ms:number}[]} clips
   * @param {{speed?:number, onClip?:(index:number)=>void}} opts  onClip(-1) when finished
   */
  play(clips, { speed = 1, onClip = () => {} } = {}) {
    this.stop();
    if (!clips.length) return;
    this.playing = true;
    const blend = 160 / speed; // smooth hand transition into each clip
    const gap = 120 / speed;
    let idx = 0;
    let start = performance.now();
    let from = this.last;
    onClip(0);
    const tick = (now) => {
      if (!this.playing) return;
      const clip = clips[idx];
      const dur = clip.ms / speed;
      const t = Math.max(0, now - start); // rAF time can be slightly before start
      if (t >= dur + gap) {
        from = clip.frames[clip.frames.length - 1];
        idx++;
        if (idx >= clips.length) {
          this.playing = false;
          this.drawHands(from, clip.caption, 0.6);
          onClip(-1);
          return;
        }
        start = now;
        onClip(idx);
      } else {
        const p = Math.min(1, t / dur) * (clip.frames.length - 1);
        const i = Math.floor(p);
        let hands = i >= clip.frames.length - 1 ? clip.frames[clip.frames.length - 1] : lerpHands(clip.frames[i], clip.frames[i + 1], p - i);
        if (from && t < blend) hands = lerpHands(from, hands, t / blend);
        this.drawHands(hands, clip.caption, 1);
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop() {
    this.playing = false;
    cancelAnimationFrame(this.raf);
  }

  clear(message = "") {
    const { ctx, canvas } = this;
    const c = this.colors();
    this.last = null;
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (message) this.caption(message, c.muted, 0.05);
  }

  caption(text, color, size = 0.08) {
    const { ctx, canvas } = this;
    ctx.fillStyle = color;
    ctx.font = `700 ${Math.round(canvas.height * size)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(text, canvas.width / 2, canvas.height - canvas.height * 0.04, canvas.width * 0.9);
  }

  drawHands(hands, caption = "", alpha = 1) {
    const { ctx, canvas } = this;
    const c = this.colors();
    const W = canvas.width, H = canvas.height;
    this.last = hands;
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = alpha;
    for (const hand of hands) this.drawHand(hand.pts.map(([x, y, z]) => [x * W, y * H, z * H]), c);
    ctx.globalAlpha = 1;
    if (caption) this.caption(caption, c.text);
  }

  drawHand(P, c) {
    const { ctx } = this;
    const palmLen = Math.hypot(P[9][0] - P[0][0], P[9][1] - P[0][1], P[9][2] - P[0][2]) || 40;
    const width = palmLen * 0.24;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // palm (outline then fill)
    const palm = () => {
      ctx.beginPath();
      PALM.forEach((k, n) => (n ? ctx.lineTo(P[k][0], P[k][1]) : ctx.moveTo(P[k][0], P[k][1])));
      ctx.closePath();
    };
    palm();
    ctx.lineWidth = width * 1.25;
    ctx.strokeStyle = c.line;
    ctx.stroke();
    palm();
    ctx.lineWidth = width;
    ctx.strokeStyle = c.skin;
    ctx.stroke();
    ctx.fillStyle = c.skin;
    ctx.fill();

    // fingers, far ones first (larger z = closer to the viewer)
    const order = FINGERS.map((f, n) => ({ f, n, z: f.reduce((s, k) => s + P[k][2], 0) / f.length }))
      .sort((a, b) => a.z - b.z);
    for (const { f, n } of order) {
      const chain = f;
      const w = n === 0 ? width * 1.05 : n === 4 ? width * 0.82 : width * 0.92;
      const path = () => {
        ctx.beginPath();
        chain.forEach((k, m) => (m ? ctx.lineTo(P[k][0], P[k][1]) : ctx.moveTo(P[k][0], P[k][1])));
      };
      path();
      ctx.lineWidth = w + Math.max(2, width * 0.18);
      ctx.strokeStyle = c.line;
      ctx.stroke();
      path();
      ctx.lineWidth = w;
      // fingers bent toward the viewer look a shade darker
      const zAvg = f.reduce((s, k) => s + P[k][2], 0) / f.length - P[0][2];
      ctx.strokeStyle = zAvg > palmLen * 0.25 ? c.skinDark : c.skin;
      ctx.stroke();
      // knuckle creases
      ctx.fillStyle = c.line;
      for (const k of chain.slice(1, -1)) {
        ctx.globalAlpha *= 0.35;
        ctx.beginPath();
        ctx.arc(P[k][0], P[k][1], w * 0.12, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha /= 0.35;
      }
    }
  }
}
