// Downloads the MediaPipe hand landmarker model to public/models/.
// Usage: node scripts/fetch-model.mjs [--optional]
//   --optional : never exit with a non-zero code (used by postinstall, so
//                `npm install` still succeeds offline).
import { mkdirSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dest = path.join(root, "public", "models", "hand_landmarker.task");
const optional = process.argv.includes("--optional");
const MIN_BYTES = 100_000; // the real model is ~7.5 MB

function isValid(file) {
  try {
    return statSync(file).size > MIN_BYTES;
  } catch {
    return false;
  }
}

async function viaFetch(tmp) {
  const res = await fetch(MODEL_URL, { signal: AbortSignal.timeout(120_000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  writeFileSync(tmp, Buffer.from(await res.arrayBuffer()));
}

function viaCurl(tmp) {
  // curl honours HTTPS_PROXY / system CA settings, useful behind proxies.
  execFileSync("curl", ["-fsSL", "--retry", "2", "-o", tmp, MODEL_URL], {
    stdio: ["ignore", "ignore", "inherit"],
    timeout: 180_000,
  });
}

async function main() {
  if (isValid(dest)) {
    console.log(`[setup] Model already present: ${path.relative(root, dest)}`);
    return;
  }
  mkdirSync(path.dirname(dest), { recursive: true });
  const tmp = `${dest}.download`;
  console.log(`[setup] Downloading hand landmarker model...\n        ${MODEL_URL}`);

  const errors = [];
  for (const [name, fn] of [["fetch", viaFetch], ["curl", viaCurl]]) {
    try {
      await fn(tmp);
      if (!isValid(tmp)) throw new Error("downloaded file is too small");
      renameSync(tmp, dest);
      console.log(`[setup] Saved ${path.relative(root, dest)} (${statSync(dest).size} bytes, via ${name})`);
      return;
    } catch (err) {
      errors.push(`${name}: ${err?.message ?? err}`);
      try { unlinkSync(tmp); } catch {}
    }
  }
  throw new Error(errors.join("; "));
}

main().catch((err) => {
  console.warn(
    `[setup] WARNING: could not download the model (${err.message}).\n` +
      `        Run "npm run setup" later, or download it manually to public/models/hand_landmarker.task:\n` +
      `        ${MODEL_URL}`
  );
  if (!optional) process.exitCode = 1;
});
