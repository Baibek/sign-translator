// Text → sign: turns text into a sequence of recorded gestures and animates them as a hand skeleton.
// planSigns() is pure (testable in Node); SignPlayer needs a <canvas>.

import { LANDMARKS, SLOT_SIZE } from "./recognizer.js";

// MediaPipe hand skeleton (pairs of landmark indices)
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // little
  [0, 17],
];

const WRIST_POS_WEIGHT = 2; // must match recognizer.js

const norm = (s) => s.toLocaleLowerCase("ru").replace(/ё/g, "е").trim();

/** Split text into words (letters, digits, hyphen) in any Cyrillic/Latin script. */
export function tokenize(text) {
  return (String(text).match(/[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*/gu) || []).map((w) => w);
}

/**
 * Plan how to show `text` with the recorded labels.
 * Multi-word labels ("как дела") match greedily, longest first.
 * A word without a sign is fingerspelled if every letter has a sign, otherwise reported as missing.
 * @returns {{type:"sign"|"letter"|"missing", label?:string, text:string, word?:string}[]}
 */
export function planSigns(text, labels) {
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
        plan.push({ type: "sign", label: byKey.get(key), text: words.slice(i, i + n).join(" ") });
        i += n;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    const letters = [...keys[i]];
    if (letters.length && letters.every((ch) => byKey.has(ch))) {
      for (const ch of letters) plan.push({ type: "letter", label: byKey.get(ch), text: ch, word: words[i] });
    } else {
      plan.push({ type: "missing", text: words[i] });
    }
    i++;
  }
  return plan;
}

/** Rebuild drawable hand points (0..1 canvas coords) from one feature frame. */
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
      pts.push([x, y]);
    }
    hands.push({ slot, pts });
  }
  return hands;
}

const lerp = (a, b, t) => a + (b - a) * t;
function lerpFrame(a, b, t) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = lerp(a[i], b[i], t);
  // presence flags: no fading in/out of a hand mid-frame
  for (let s = 0; s < 2; s++) out[s * SLOT_SIZE + 65] = t < 0.5 ? a[s * SLOT_SIZE + 65] : b[s * SLOT_SIZE + 65];
  return out;
}

/** Animates planned signs on a canvas. */
export class SignPlayer {
  constructor(canvas, { colors } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.colors = colors || (() => ({ bg: "#111", bone: "#fff", joint: "#fff", left: "#5aa0ff", right: "#ff8a3d", text: "#fff" }));
    this.raf = 0;
    this.playing = false;
  }

  /**
   * @param {{frames:ArrayLike<number>[], caption:string, ms:number}[]} clips
   * @param {(index:number)=>void} onClip  called when a clip starts (-1 when finished)
   */
  play(clips, { speed = 1, onClip = () => {} } = {}) {
    this.stop();
    if (!clips.length) return;
    this.playing = true;
    const gap = 250 / speed;
    let idx = 0;
    let start = performance.now();
    onClip(0);
    const tick = (now) => {
      if (!this.playing) return;
      const clip = clips[idx];
      const dur = clip.ms / speed;
      const t = now - start;
      if (t >= dur + gap) {
        idx++;
        if (idx >= clips.length) {
          this.playing = false;
          this.drawFrame(clip.frames[clip.frames.length - 1], clip.caption, 0.5);
          onClip(-1);
          return;
        }
        start = now;
        onClip(idx);
      } else {
        const p = Math.min(1, t / dur) * (clip.frames.length - 1);
        const i = Math.floor(p);
        const f = i >= clip.frames.length - 1 ? clip.frames[clip.frames.length - 1] : lerpFrame(clip.frames[i], clip.frames[i + 1], p - i);
        this.drawFrame(f, clip.caption, t > dur ? 0.5 : 1);
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
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (message) this.caption(message, c.muted || c.text, 0.045);
  }

  caption(text, color, size = 0.07) {
    const { ctx, canvas } = this;
    ctx.fillStyle = color;
    ctx.font = `600 ${Math.round(canvas.height * size)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(text, canvas.width / 2, canvas.height - canvas.height * 0.04, canvas.width * 0.9);
  }

  drawFrame(f, caption = "", alpha = 1) {
    const { ctx, canvas } = this;
    const c = this.colors();
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = alpha;
    for (const { slot, pts } of framePoints(f)) {
      const color = slot === 0 ? c.left : c.right;
      ctx.lineCap = "round";
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(4, W * 0.012);
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.moveTo(pts[a][0] * W, pts[a][1] * H);
        ctx.lineTo(pts[b][0] * W, pts[b][1] * H);
      }
      ctx.stroke();
      ctx.fillStyle = c.joint;
      for (const [x, y] of pts) {
        ctx.beginPath();
        ctx.arc(x * W, y * H, Math.max(3, W * 0.008), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    if (caption) this.caption(caption, c.text);
  }
}
