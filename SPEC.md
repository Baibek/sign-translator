# Sign → Russian translator (жест → орыс тілі) — shared spec

Goal: a web app. A person signs (sign language / gestures) in front of a webcam; the app translates it into Russian text in real time and can speak it (ru-RU speech synthesis). UI language: Kazakh (short labels), output: Russian.

Stack: Node.js 22 (ESM), Express, @anthropic-ai/sdk (TypeScript/JS SDK), @mediapipe/tasks-vision (browser, vendored from node_modules — do NOT rely on cdn.jsdelivr at runtime). No build step: plain ES modules in public/.

## Layout
```
sign-translator/
  package.json          (type: module; scripts: start, setup, test)
  server.js             backend (agent B)
  scripts/fetch-model.mjs  downloads hand_landmarker.task to public/models/ (agent B)
  public/index.html     UI (agent A)
  public/app.js         main app (agent A)
  public/recognizer.js  landmark normalization + DTW/kNN classifier, pure functions, no DOM (agent A)
  public/styles.css     (agent A)
  test/recognizer.test.mjs  node:test unit tests for recognizer.js (agent A)
  README.md             Kazakh + Russian run instructions (agent B)
  .env.example          ANTHROPIC_API_KEY=... (agent B)
```

## Static routes served by server.js
- `/` → public/
- `/vendor/tasks-vision/` → node_modules/@mediapipe/tasks-vision/ (contains vision_bundle.mjs and wasm/)
- `/models/hand_landmarker.task` → public/models/hand_landmarker.task
  (downloaded by `npm run setup` from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

Frontend imports: `import { FilesetResolver, HandLandmarker } from "/vendor/tasks-vision/vision_bundle.mjs"`; `FilesetResolver.forVisionTasks("/vendor/tasks-vision/wasm")`; model path `/models/hand_landmarker.task`; runningMode "VIDEO", numHands 2.

## API contract
`GET /api/health` → `{ ok: true, ai: boolean }` (ai = API credentials configured)

`POST /api/translate`  (JSON, body limit 15mb)
Request:
```json
{
  "frames": ["data:image/jpeg;base64,...", "..."],   // 2..8 frames, chronological, ~480px wide JPEG
  "context": ["привет", "как"],                       // optional: previous recognized Russian words/phrases
  "signLanguage": "auto"                               // optional: "auto" | "rsl" (русский жестовый) | "ksl" (казахский жестовый)
}
```
Response 200:
```json
{ "text": "Привет", "confidence": 0.0-1.0, "gloss": "ПРИВЕТ", "note": "optional short explanation in Russian" }
```
If nothing meaningful is signed: `{ "text": "", "confidence": 0, ... }`.
Errors: `{ "error": "message" }` with 400 (bad input), 503 (no API key), 502 (upstream error), 429 (rate limited).

## Recognition logic (frontend)
- Per frame: HandLandmarker → up to 2 hands × 21 landmarks + handedness.
- Feature per frame: for each hand slot (Left, Right): landmarks translated to wrist origin, scaled by palm size (wrist→middle MCP distance), flattened (63 numbers) + wrist position relative to frame center (2 numbers); missing hand = zeros + presence flag. 
- Segmentation: a "gesture" starts when a hand appears / motion exceeds threshold and ends after ~500 ms of stillness or hands disappearing (min 300 ms, max 4 s).
- Mode 1 "Менің жестерім" (custom, offline): user records examples (label = Russian word/phrase, ≥3 examples recommended). Sequences resampled to 24 frames; classify with DTW distance, kNN (k=3); reject if best distance > threshold (show "?"). Stored in localStorage; export/import JSON.
- Mode 2 "AI аудармашы": on gesture end, send 4–8 evenly spaced frames (JPEG, 480px) to /api/translate with last ~5 words as context.
- Mode "Аралас" (hybrid): use custom match if confident, else AI.
- Output: running Russian sentence, history list, buttons: Тазалау (clear), Дауыстау (speak, ru-RU speechSynthesis), Көшіру (copy).
