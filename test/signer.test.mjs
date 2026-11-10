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

test("planSigns: recorded letters first, then built-in dactyl; only unknown chars are missing", () => {
  const plan = planSigns("мама папа hi", ["м"]);
  assert.deepEqual(plan.map((p) => p.source || p.type), ["recorded", "builtin", "recorded", "builtin", "builtin", "builtin", "builtin", "builtin", "missing"]);
  assert.equal(plan[0].word, "мама");
  assert.deepEqual(plan.at(-1), { type: "missing", text: "hi" });
  // without built-ins, words with unrecorded letters are missing
  assert.equal(planSigns("папа", ["м"], { builtin: false })[0].type, "missing");
});

test("built-in dactyl covers the Russian alphabet and Kazakh letters", async () => {
  const { hasDactyl, letterPose, letterFrames } = await import("../public/dactyl.js");
  for (const ch of "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі") assert.ok(hasDactyl(ch), ch);
  const pose = letterPose("в");
  assert.equal(pose.length, 21);
  for (const p of pose) for (const v of p) assert.ok(Number.isFinite(v));
  // В: straight fingers -> fingertips well above the wrist; А: fist -> fingertips close to the palm
  assert.ok(letterPose("в")[12][1] < -1.4);
  assert.ok(letterPose("а")[12][1] > -1.2);
  const frames = letterFrames("щ", 8);
  assert.equal(frames.length, 8);
  assert.ok(frames[7][0].pts[0][1] > frames[0][0].pts[0][1], "Щ moves down");
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
  assert.equal(hands[0].pts[0].length, 3);
  assert.ok(Math.abs(hands[0].pts[12][1] - 0.3) < 1e-6);
});
