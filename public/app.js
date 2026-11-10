// app.js — UI glue: camera, MediaPipe hand tracking, segmentation,
// custom (DTW) recognition, AI translation, training and output.
import { FilesetResolver, HandLandmarker, DrawingUtils } from "./vendor/tasks-vision/vision_bundle.mjs";
import { frameFeatures, handsInFeatures, GestureClassifier, Segmenter } from "./recognizer.js";
import { planSigns, buildClips, SignPlayer } from "./signer.js";

// ---------- constants ----------
// Relative URLs so the app works both from the Node server and as a static site (GitHub Pages subpath)
const WASM_PATH = new URL("./vendor/tasks-vision/wasm", import.meta.url).href;
const MODEL_PATH = new URL("./models/hand_landmarker.task", import.meta.url).href;
const STORE_GESTURES = "ym.gestures.v1";
const STORE_SETTINGS = "ym.settings.v1";
const SNAP_WIDTH = 480;
const SNAP_QUALITY = 0.7;
const SNAP_INTERVAL_MS = 1000 / 6; // ~6 fps
const SNAP_KEEP_MS = 8000; // ring buffer length
const FEAT_KEEP_MS = 5000;
const HYBRID_MIN_CONF = 0.6; // "Аралас": use custom match at/above this confidence
const CONTEXT_WORDS = 5;
const MANUAL_WINDOW_MS = 2000;

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const el = {
  video: $("video"), overlay: $("overlay"), placeholder: $("placeholder"), btnCamera: $("btn-camera"),
  countdown: $("countdown"), motionBar: $("motion-bar"), error: $("error"),
  modes: $("modes"), signLang: $("sign-lang"), btnTranslate: $("btn-translate"), aiHint: $("ai-hint"),
  trainForm: $("train-form"), labelInput: $("label-input"), btnRecord: $("btn-record"), labels: $("labels"),
  btnExport: $("btn-export"), importFile: $("import-file"), importLabel: $("import-label"),
  btnClearGestures: $("btn-clear-gestures"),
  sentence: $("sentence"), badge: $("badge"), btnClear: $("btn-clear"), btnSpeak: $("btn-speak"),
  btnCopy: $("btn-copy"), autoSpeak: $("auto-speak"), history: $("history"),
  st: { camera: $("st-camera"), model: $("st-model"), gesture: $("st-gesture"), ai: $("st-ai") },
};

// ---------- state ----------
const state = {
  landmarker: null,
  modelPromise: null,
  stream: null,
  running: false,
  lastVideoTime: -1,
  aiAvailable: false,
  mode: "custom",
  words: [], // current sentence
  history: [],
  snaps: [], // {t, url}
  lastSnapT: 0,
  feats: [], // {t, f, hands}
  rec: null, // training recording state
  ignoreSegmentsBefore: 0,
  ai: { inFlight: false, queued: null },
};
const segmenter = new Segmenter();
let classifier = new GestureClassifier();
let drawer = null;
const overlayCtx = el.overlay.getContext("2d");
const snapCanvas = document.createElement("canvas");
const snapCtx = snapCanvas.getContext("2d");

// ---------- small helpers ----------
function setStatus(name, s) {
  el.st[name].dataset.state = s; // off | on | busy | error
}
function showError(msg) {
  el.error.textContent = msg;
  el.error.hidden = !msg;
}
function loadJSON(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v ? JSON.parse(v) : fallback;
  } catch {
    return fallback;
  }
}
function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
function saveSettings() {
  saveJSON(STORE_SETTINGS, { mode: state.mode, autoSpeak: el.autoSpeak.checked, signLang: el.signLang.value });
}
function saveGestures() {
  if (!saveJSON(STORE_GESTURES, classifier.toJSON())) {
    showError("Жестерді браузерге сақтау мүмкін болмады (орын жетпеуі мүмкін). Экспорт жасап алыңыз.");
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtTime = (d) => d.toLocaleTimeString("kk-KZ", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const pct = (c) => `${Math.round((c || 0) * 100)}%`;

// ---------- model ----------
function isSoftwareWebGL() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2") || document.createElement("canvas").getContext("webgl");
    if (!gl) return true;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return /swiftshader|llvmpipe|softpipe|software/i.test(renderer);
  } catch {
    return false;
  }
}
async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  if (isSoftwareWebGL()) {
    // Software WebGL (SwiftShader/llvmpipe, e.g. GPU blocklisted or headless) is ~10× slower than WASM CPU.
    return await HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
  try {
    return await HandLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    console.warn("GPU delegate failed, falling back to CPU", e);
    return await HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}
function loadModel() {
  if (!state.modelPromise) {
    setStatus("model", "busy");
    state.modelPromise = createLandmarker()
      .then((lm) => {
        state.landmarker = lm;
        setStatus("model", "on");
        return lm;
      })
      .catch((e) => {
        console.error(e);
        setStatus("model", "error");
        state.modelPromise = null;
        showError("Қол модельі жүктелмеді. Серверде «npm run setup» орындалғанын тексеріңіз.");
        throw e;
      });
  }
  return state.modelPromise;
}

// ---------- camera ----------
function cameraErrorMessage(e) {
  switch (e && e.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Камераға рұқсат берілмеді. Браузердің мекенжай жолағындағы белгіше арқылы рұқсат беріңіз.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "Камера табылмады. Камераның қосылғанын тексеріңіз.";
    case "NotReadableError":
      return "Камера басқа бағдарламада қолданылып жатыр. Оны жауып, қайталаңыз.";
    default:
      return "Камераны қосу мүмкін болмады: " + (e && e.message ? e.message : e);
  }
}
async function startCamera() {
  showError("");
  if (!window.isSecureContext) {
    showError("Камера тек қауіпсіз байланыста жұмыс істейді (https немесе localhost арқылы ашыңыз).");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Бұл браузер камераны қолдамайды.");
    return;
  }
  el.btnCamera.disabled = true;
  setStatus("camera", "busy");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    setStatus("camera", "error");
    el.btnCamera.disabled = false;
    showError(cameraErrorMessage(e));
    return;
  }
  el.video.srcObject = state.stream;
  await el.video.play().catch(() => {});
  setStatus("camera", "on");
  el.placeholder.hidden = true;
  try {
    await loadModel();
  } catch {
    return;
  }
  state.running = true;
  updateButtons();
  scheduleFrame();
}

// ---------- frame loop ----------
function scheduleFrame() {
  if (!state.running) return;
  if (el.video.requestVideoFrameCallback) el.video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

function onFrame() {
  try {
    processFrame();
  } catch (e) {
    console.error(e);
  }
  scheduleFrame();
}

function processFrame() {
  const v = el.video;
  if (!state.landmarker || v.readyState < 2 || v.currentTime === state.lastVideoTime) return;
  state.lastVideoTime = v.currentTime;
  const now = performance.now();

  const result = state.landmarker.detectForVideo(v, now);
  drawLandmarks(result);
  captureSnapshot(now);

  const f = frameFeatures(result);
  const hands = handsInFeatures(f) > 0;
  state.feats.push({ t: now, f, hands });
  while (state.feats.length && now - state.feats[0].t > FEAT_KEEP_MS) state.feats.shift();

  const segment = segmenter.push(f, now, hands);
  el.motionBar.style.width = Math.min(100, (segmenter.energy / (segmenter.motionThreshold * 3)) * 100) + "%";

  if (state.rec) {
    stepRecording(f, now, hands);
  } else {
    setStatus("gesture", segmenter.active ? "busy" : "off");
    if (segment && segment.startMs >= state.ignoreSegmentsBefore) handleSegment(segment);
  }
}

function drawLandmarks(result) {
  const v = el.video;
  if (el.overlay.width !== v.videoWidth || el.overlay.height !== v.videoHeight) {
    el.overlay.width = v.videoWidth;
    el.overlay.height = v.videoHeight;
  }
  overlayCtx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  if (!drawer) drawer = new DrawingUtils(overlayCtx);
  (result.landmarks || []).forEach((lm, i) => {
    const side = result.handedness?.[i]?.[0]?.categoryName;
    const color = side === "Left" ? "#38bdf8" : "#f472b6";
    drawer.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 3 });
    drawer.drawLandmarks(lm, { color: "#ffffff", fillColor: color, lineWidth: 1, radius: 3 });
  });
}

// Small JPEG snapshots (unmirrored, as the camera sees the signer) for AI.
function captureSnapshot(now) {
  if (now - state.lastSnapT < SNAP_INTERVAL_MS) return;
  const v = el.video;
  if (!v.videoWidth) return;
  state.lastSnapT = now;
  const h = Math.round((SNAP_WIDTH * v.videoHeight) / v.videoWidth);
  if (snapCanvas.width !== SNAP_WIDTH || snapCanvas.height !== h) {
    snapCanvas.width = SNAP_WIDTH;
    snapCanvas.height = h;
  }
  snapCtx.drawImage(v, 0, 0, SNAP_WIDTH, h);
  state.snaps.push({ t: now, url: snapCanvas.toDataURL("image/jpeg", SNAP_QUALITY) });
  while (state.snaps.length && now - state.snaps[0].t > SNAP_KEEP_MS) state.snaps.shift();
}

/** Pick 4–8 evenly spaced snapshots inside [startMs, endMs] (widened if too few). */
function pickFrames(startMs, endMs) {
  const want = Math.max(4, Math.min(8, Math.round((endMs - startMs) / 250)));
  let pool = state.snaps.filter((s) => s.t >= startMs - 100 && s.t <= endMs + 100);
  if (pool.length < 4) {
    const mid = (startMs + endMs) / 2;
    pool = [...state.snaps].sort((a, b) => Math.abs(a.t - mid) - Math.abs(b.t - mid)).slice(0, 4);
    pool.sort((a, b) => a.t - b.t);
  }
  if (pool.length <= want) return pool.map((s) => s.url);
  const out = [];
  for (let i = 0; i < want; i++) out.push(pool[Math.round((i * (pool.length - 1)) / (want - 1))].url);
  return out;
}

// ---------- recognition ----------
function handleSegment(seg) {
  const mode = state.mode;
  if (mode === "custom" || mode === "hybrid") {
    const res = classifier.size ? classifier.classify(seg.frames, { k: 3 }) : null;
    if (res && (mode === "custom" || res.confidence >= HYBRID_MIN_CONF)) {
      addResult({ text: res.label, confidence: res.confidence, source: "жест" });
      return;
    }
    if (mode === "custom" || !state.aiAvailable) {
      showBadge(classifier.size ? "?" : "Алдымен жестерді жазыңыз", null, "жест", true);
      return;
    }
  }
  if (!state.aiAvailable) return;
  const frames = pickFrames(seg.startMs, seg.endMs);
  if (frames.length >= 2) requestAI(frames);
}

// Serialized AI requests: one in flight, at most one queued (latest wins).
function requestAI(frames) {
  if (state.ai.inFlight) {
    state.ai.queued = frames;
    return;
  }
  runAI(frames);
}
async function runAI(frames) {
  state.ai.inFlight = true;
  setStatus("ai", "busy");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 45000);
  try {
    const resp = await fetch(new URL("./api/translate", import.meta.url), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, context: state.words.slice(-CONTEXT_WORDS), signLanguage: el.signLang.value }),
      signal: ctrl.signal,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw Object.assign(new Error(data.error || resp.statusText), { status: resp.status });
    const text = String(data.text || "").trim();
    if (text) addResult({ text, confidence: Number(data.confidence) || 0, source: "AI", note: data.note, gloss: data.gloss });
    else showBadge("—", data.confidence, "AI", true);
    setStatus("ai", "off");
  } catch (e) {
    setStatus("ai", "error");
    const msg =
      e.name === "AbortError" ? "AI жауап бермеді (уақыт бітті)." :
      e.status === 429 ? "AI сұраныстары тым көп. Сәл күтіңіз." :
      e.status === 503 ? "AI қолжетімсіз: " + e.message :
      e.status === 400 ? "AI сұранысы қате: " + e.message :
      "AI қатесі: " + e.message;
    showError(msg);
  } finally {
    clearTimeout(timer);
    state.ai.inFlight = false;
    const next = state.ai.queued;
    state.ai.queued = null;
    if (next) runAI(next);
  }
}

function manualTranslate() {
  const now = performance.now();
  const frames = pickFrames(now - MANUAL_WINDOW_MS, now);
  if (frames.length < 2) {
    showError("Кадрлар жеткіліксіз — камера қосулы ма?");
    return;
  }
  showError("");
  requestAI(frames);
}

// ---------- output ----------
function addResult({ text, confidence, source, note, gloss }) {
  showError("");
  showBadge(text, confidence, source, false);
  const last = state.words[state.words.length - 1];
  if (last && last.toLowerCase() === text.toLowerCase()) return; // no duplicate consecutive words
  state.words.push(text);
  state.history.unshift({ time: new Date(), text, confidence, source, note: note || (gloss ? `глосса: ${gloss}` : "") });
  if (state.history.length > 200) state.history.pop();
  renderSentence();
  renderHistory();
  if (el.autoSpeak.checked) speak(text);
}

function showBadge(text, confidence, source, reject) {
  el.badge.hidden = false;
  el.badge.classList.toggle("reject", !!reject);
  el.badge.replaceChildren();
  const src = document.createElement("span");
  src.className = "src";
  src.textContent = source;
  const t = document.createElement("span");
  t.textContent = text;
  t.lang = "ru";
  el.badge.append(src, t);
  if (confidence != null) {
    const c = document.createElement("span");
    c.textContent = pct(confidence);
    el.badge.append(c);
  }
}

function sentenceText() {
  let s = state.words.join(" ").trim();
  if (s) s = s[0].toUpperCase() + s.slice(1);
  return s;
}
function renderSentence() {
  const s = sentenceText();
  if (!s) {
    el.sentence.innerHTML = '<span class="placeholder">Аударма осында шығады…</span>';
  } else {
    el.sentence.textContent = s;
  }
}
function renderHistory() {
  el.history.replaceChildren(
    ...state.history.map((h) => {
      const li = document.createElement("li");
      const time = document.createElement("time");
      time.dateTime = h.time.toISOString();
      time.textContent = fmtTime(h.time);
      const text = document.createElement("span");
      text.lang = "ru";
      text.textContent = h.text;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${h.source} · ${pct(h.confidence)}`;
      li.append(time, text, meta);
      if (h.note) {
        const n = document.createElement("span");
        n.className = "note";
        n.lang = "ru";
        n.textContent = h.note;
        li.append(n);
      }
      return li;
    })
  );
}

// Speech: ru-RU, prefer an installed Russian voice.
let ruVoice = null;
function pickVoice() {
  if (!("speechSynthesis" in window)) return;
  const voices = speechSynthesis.getVoices();
  ruVoice = voices.find((v) => v.lang === "ru-RU") || voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("ru")) || null;
}
function speak(text) {
  if (!("speechSynthesis" in window)) {
    showError("Бұл браузерде дауыстау қолжетімсіз.");
    return;
  }
  if (!text) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "ru-RU";
  if (ruVoice) u.voice = ruVoice;
  speechSynthesis.speak(u);
}

async function copyText() {
  const s = sentenceText();
  if (!s) return;
  try {
    await navigator.clipboard.writeText(s);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = s;
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  flash(el.btnCopy, "Көшірілді ✓");
}
function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1200);
}

// ---------- training ----------
async function startRecording(label) {
  if (!state.running || state.rec) return;
  showError("");
  state.rec = { label, phase: "countdown" };
  updateButtons();
  el.countdown.hidden = false;
  el.countdown.classList.remove("rec");
  for (const n of [3, 2, 1]) {
    el.countdown.textContent = n;
    await sleep(1000);
    if (!state.rec) return;
  }
  el.countdown.classList.add("rec");
  el.countdown.textContent = "Жазылуда…";
  setStatus("gesture", "busy");
  Object.assign(state.rec, { phase: "recording", startMs: performance.now(), frames: [], firstMotion: null, lastMotion: null });
}

// Records one gesture: from first motion until ~600 ms of stillness;
// a static handshape (no motion for 2 s) records the last second; max 4.5 s.
function stepRecording(f, t, hands) {
  const r = state.rec;
  if (r.phase !== "recording") return;
  if (hands) r.frames.push({ t, f });
  if (hands && segmenter.energy > segmenter.motionThreshold) {
    if (r.firstMotion == null) r.firstMotion = t;
    r.lastMotion = t;
  }
  let from = null, to = t;
  if (r.firstMotion != null && t - r.lastMotion >= 600) {
    from = r.firstMotion - 150;
    to = r.lastMotion + 100;
  } else if (r.firstMotion == null && t - r.startMs >= 2000) {
    from = t - 1000;
  } else if (t - r.startMs >= 4500) {
    from = (r.firstMotion ?? r.startMs) - 150;
  } else {
    return;
  }
  const frames = r.frames.filter((x) => x.t >= from && x.t <= to).map((x) => x.f);
  finishRecording(frames, t);
}

function finishRecording(frames, t) {
  const { label } = state.rec;
  state.rec = null;
  state.ignoreSegmentsBefore = t; // don't translate the training gesture itself
  el.countdown.hidden = true;
  setStatus("gesture", "off");
  if (frames.length < 6) {
    showError("Қол көрінбеді. Қолыңыз кадрда екенін тексеріп, қайталаңыз.");
  } else {
    classifier.add(label, frames);
    saveGestures();
    renderLabels();
    showBadge(`${label} (${classifier.count(label)})`, null, "жазылды", false);
  }
  updateButtons();
}

function renderLabels() {
  el.labels.replaceChildren(
    ...classifier.labels().sort((a, b) => a.localeCompare(b, "ru")).map((label) => {
      const li = document.createElement("li");
      const use = document.createElement("button");
      use.type = "button";
      use.className = "use";
      use.lang = "ru";
      use.textContent = label;
      use.title = "Тағы мысал жазу үшін таңдау";
      use.addEventListener("click", () => {
        el.labelInput.value = label;
        el.labelInput.focus();
      });
      const n = classifier.count(label);
      const count = document.createElement("span");
      count.className = "count" + (n < 3 ? " low" : "");
      count.textContent = `×${n}`;
      count.title = `${n} мысал`;
      const del = document.createElement("button");
      del.type = "button";
      del.className = "del";
      del.textContent = "×";
      del.setAttribute("aria-label", `«${label}» сөзін өшіру`);
      del.addEventListener("click", () => {
        if (!confirm(`«${label}» (${n} мысал) өшірілсін бе?`)) return;
        classifier.remove(label);
        saveGestures();
        renderLabels();
      });
      li.append(use, count, del);
      return li;
    })
  );
}

function exportGestures() {
  const blob = new Blob([JSON.stringify(classifier.toJSON())], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ym-gestures-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function importGestures(file) {
  try {
    const loaded = GestureClassifier.fromJSON(await file.text());
    // merge imported examples into the current set
    const data = loaded.toJSON();
    for (const [label, list] of Object.entries(data.examples)) for (const seq of list) classifier.add(label, seq);
    saveGestures();
    renderLabels();
    showError("");
    showBadge(`${loaded.size} мысал импортталды`, null, "импорт", false);
  } catch (e) {
    showError("Файлды оқу мүмкін болмады: JSON пішімі қате.");
  }
}

// ---------- modes / buttons ----------
function setMode(mode) {
  if (!state.aiAvailable && mode !== "custom") mode = "custom";
  state.mode = mode;
  const input = el.modes.querySelector(`input[value="${mode}"]`);
  if (input) input.checked = true;
  saveSettings();
}
function updateButtons() {
  el.btnRecord.disabled = !state.running || !!state.rec;
  el.btnTranslate.disabled = !state.running || !state.aiAvailable;
}

async function checkHealth() {
  try {
    const r = await fetch(new URL("./api/health", import.meta.url), { cache: "no-store" });
    const data = await r.json();
    state.aiAvailable = !!(data && data.ai);
    state.staticSite = false;
  } catch {
    // No backend (e.g. GitHub Pages): only the offline "Менің жестерім" mode works
    state.aiAvailable = false;
    state.staticSite = true;
  }
  if (state.staticSite) {
    el.aiHint.textContent = "Бұл сайт нұсқасы: «Менің жестерім» режимі толық жұмыс істейді. AI режимі тек сервермен іске қосылған толық нұсқада бар (README қараңыз).";
  }
  for (const input of el.modes.querySelectorAll("input")) {
    if (input.value !== "custom") input.disabled = !state.aiAvailable;
  }
  el.aiHint.hidden = state.aiAvailable;
  setStatus("ai", state.aiAvailable ? "off" : "error");
  el.st.ai.title = state.aiAvailable ? "" : (state.staticSite ? "AI тек толық нұсқада" : "AI өшірулі (ANTHROPIC_API_KEY жоқ)");
  updateButtons();
}

// ---------- text → sign ----------
let signPlayer = null;
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function t2sColors() {
  return { bg: cssVar("--surface-2"), skin: cssVar("--skin"), skinDark: cssVar("--skin-dark"), line: cssVar("--skin-line"),
    text: cssVar("--text"), muted: cssVar("--muted") };
}
function renderPlan(plan, current = -1) {
  const box = $("t2s-words");
  box.replaceChildren(...plan.map((p, i) => {
    const s = document.createElement("span");
    s.textContent = p.type === "letter" ? p.text.toLocaleUpperCase("ru") : p.text;
    if (p.type === "missing") { s.className = "missing"; s.title = "Бұл сөзді көрсету мүмкін емес"; }
    if (p.source === "builtin") s.classList.add("builtin");
    if (i === current) s.classList.add("now");
    return s;
  }));
}
function playText() {
  const text = $("t2s-input").value.trim();
  if (!text) { signPlayer.clear("Мәтін жазыңыз"); renderPlan([]); return; }
  const plan = planSigns(text, classifier.labels());
  renderPlan(plan);
  const playable = buildClips(plan, (label) => classifier.examples.get(label)?.[0], { aspect: signPlayer.aspect });
  const missing = plan.filter((p) => p.type === "missing").map((p) => p.text);
  const spelled = [...new Set(plan.filter((p) => p.source === "builtin").map((p) => p.word))];
  const notes = [];
  if (spelled.length) notes.push(`Дактильмен (әріп-әріп) көрсетілді: ${spelled.join(", ")}. Кірістірілген әріп пішіндері жуық, маманмен тексеру керек.`);
  if (plan.some((p) => p.source === "builtin" && "әғқңөұүһі".includes(p.text.toLocaleLowerCase("kk"))))
    notes.push("Қазақтың ә, ғ, қ, ң, ө, ұ, ү, һ, і әріптері әзірге ең жақын орыс әрпінің пішінімен көрсетіледі.");
  if (missing.length) notes.push(`Көрсету мүмкін емес: ${missing.join(", ")}.`);
  if (!notes.length) notes.push("Барлық сөз сіз жазған жестермен көрсетілді.");
  $("t2s-hint").textContent = notes.join(" ");
  if (!playable.length) { signPlayer.clear("Жазылған жест табылмады"); return; }
  signPlayer.play(playable, {
    speed: Number($("t2s-speed").value) || 1,
    onClip: (k) => renderPlan(plan, k >= 0 ? playable[k].i : -1),
  });
}
function initTextToSign() {
  signPlayer = new SignPlayer($("t2s-canvas"), { colors: t2sColors });
  signPlayer.clear("Мәтін жазып, «Көрсету» басыңыз");
  $("t2s-form").addEventListener("submit", (e) => { e.preventDefault(); playText(); });
  $("t2s-sentence").addEventListener("click", () => {
    const t = sentenceText();
    if (!t) { $("t2s-hint").textContent = "Жоғарыда әзірге аударма жоқ."; return; }
    $("t2s-input").value = t;
    playText();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { if (!signPlayer.playing) signPlayer.clear(); });
}

// ---------- init ----------
function init() {
  // restore gestures + settings
  const stored = loadJSON(STORE_GESTURES, null);
  if (stored) {
    try {
      classifier = GestureClassifier.fromJSON(stored);
    } catch {
      console.warn("Stored gestures are invalid; starting fresh");
    }
  }
  const settings = loadJSON(STORE_SETTINGS, {});
  el.autoSpeak.checked = !!settings.autoSpeak;
  if (settings.signLang) el.signLang.value = settings.signLang;
  renderLabels();
  renderSentence();
  initTextToSign();

  // events
  el.btnCamera.addEventListener("click", startCamera);
  el.modes.addEventListener("change", (e) => setMode(e.target.value));
  el.signLang.addEventListener("change", saveSettings);
  el.autoSpeak.addEventListener("change", saveSettings);
  el.btnTranslate.addEventListener("click", manualTranslate);
  el.trainForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const label = el.labelInput.value.trim();
    if (label) startRecording(label);
  });
  el.btnExport.addEventListener("click", exportGestures);
  el.importFile.addEventListener("change", () => {
    const f = el.importFile.files[0];
    if (f) importGestures(f);
    el.importFile.value = "";
  });
  el.importLabel.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.importFile.click();
    }
  });
  el.btnClearGestures.addEventListener("click", () => {
    if (!classifier.size || !confirm("Барлық жазылған жестер өшірілсін бе?")) return;
    classifier.clear();
    saveGestures();
    renderLabels();
  });
  el.btnClear.addEventListener("click", () => {
    state.words = [];
    el.badge.hidden = true;
    renderSentence();
  });
  el.btnSpeak.addEventListener("click", () => speak(sentenceText()));
  el.btnCopy.addEventListener("click", copyText);
  if ("speechSynthesis" in window) {
    pickVoice();
    speechSynthesis.addEventListener?.("voiceschanged", pickVoice);
  }
  if (!window.isSecureContext) {
    showError("Камера тек қауіпсіз байланыста жұмыс істейді (https немесе localhost арқылы ашыңыз).");
  }

  checkHealth().then(() => setMode(settings.mode || "custom"));
  setMode("custom");
  loadModel().catch(() => {}); // preload in background
}

init();
