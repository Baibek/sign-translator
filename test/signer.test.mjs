import { test } from "node:test";
import assert from "node:assert/strict";
import { planSigns, tokenize, framePoints } from "../public/signer.js";
import { FEATURE_SIZE, SLOT_SIZE } from "../public/recognizer.js";

test("tokenize keeps Cyrillic incl. Kazakh letters and hyphens", () => {
  assert.deepEqual(tokenize("Сәлем, как-то дела? 2 раза"), ["Сәлем", "как-то", "дела", "2", "раза"]);
});

test("planSigns: words, multi-word labels, case and ё", () => {
  const plan = planSigns("Привет! Как дела, ёж?", ["привет", "как дела", "еж", "как"]);
  assert.deepEqual(plan.map((p) => [p.type, p.label]), [["sign", "привет"], ["sign", "как дела"], ["sign", "еж"]]);
});

test("planSigns: fingerspells unknown words when all letters exist, else missing", () => {
  const plan = planSigns("мама папа", ["м", "а"]);
  assert.deepEqual(plan.map((p) => p.type), ["letter", "letter", "letter", "letter", "missing"]);
  assert.equal(plan[0].word, "мама");
  assert.deepEqual(plan[4], { type: "missing", text: "папа" });
});

test("framePoints rebuilds a mirrored hand from features", () => {
  const f = new Float32Array(FEATURE_SIZE);
  const base = SLOT_SIZE; // right-hand slot
  f[base + 65] = 1;
  f[base + 63] = 0.2; // wrist x = 0.6 before mirroring
  f[base + 64] = 0;   // wrist y = 0.5
  f[base + 12 * 3 + 1] = -2; // middle fingertip 2 palms above wrist
  const hands = framePoints(f, { scale: 0.1 });
  assert.equal(hands.length, 1);
  assert.equal(hands[0].slot, 1);
  assert.ok(Math.abs(hands[0].pts[0][0] - 0.4) < 1e-6);
  assert.ok(Math.abs(hands[0].pts[12][1] - 0.3) < 1e-6);
});
