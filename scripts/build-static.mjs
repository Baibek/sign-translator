// Builds a static copy of the app into dist/ for GitHub Pages (no server, no AI mode).
import { cp, mkdir, rm, access, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const dist = path.join(root, "dist");
const vision = path.join(root, "node_modules/@mediapipe/tasks-vision");
const model = path.join(root, "public/models/hand_landmarker.task");

await access(model).catch(() => { throw new Error("Model missing: run `npm run setup` first"); });
await rm(dist, { recursive: true, force: true });
await cp(path.join(root, "public"), dist, { recursive: true, filter: (src) => !src.endsWith(".download") });
await mkdir(path.join(dist, "vendor/tasks-vision"), { recursive: true });
await cp(path.join(vision, "vision_bundle.mjs"), path.join(dist, "vendor/tasks-vision/vision_bundle.mjs"));
await cp(path.join(vision, "wasm"), path.join(dist, "vendor/tasks-vision/wasm"), { recursive: true });
await writeFile(path.join(dist, ".nojekyll"), ""); // serve files as-is on GitHub Pages
console.log("Static site built in dist/");
