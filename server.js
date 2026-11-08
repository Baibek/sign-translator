// Sign → Russian translator backend.
// Serves the static frontend, the vendored MediaPipe tasks-vision bundle, the
// hand landmarker model, and proxies AI translation requests to Claude.
import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import express from "express";
import Anthropic from "@anthropic-ai/sdk";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(ROOT, ".env"), quiet: true });

const PORT = Number(process.env.PORT) || 3000;
const MODEL = "claude-opus-5-5";
const PUBLIC_DIR = path.join(ROOT, "public");
const VISION_DIR = path.join(ROOT, "node_modules", "@mediapipe", "tasks-vision");
const MODEL_FILE = path.join(PUBLIC_DIR, "models", "hand_landmarker.task");

const MIN_FRAMES = 2;
const MAX_FRAMES = 8;
const MAX_FRAME_CHARS = 2_000_000; // per data URL (~1.5 MB image); 480px JPEGs are ~30-80 KB
const MAX_CONTEXT_ITEMS = 10;
const MAX_CONTEXT_CHARS = 200;
const SIGN_LANGUAGES = new Set(["auto", "rsl", "ksl"]);
const DATA_URL_RE = /^data:(image\/(?:jpeg|png));base64,([A-Za-z0-9+/]+={0,2})$/;

const RATE_LIMIT = 30; // requests per window per IP
const RATE_WINDOW_MS = 60_000;

const hasCredentials = () =>
  Boolean(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN);

let client = null;
function getClient() {
  // Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN from the environment.
  if (!client) client = new Anthropic({ timeout: 60_000, maxRetries: 2 });
  return client;
}

// ---------------------------------------------------------------- prompt ---

const SYSTEM_PROMPT = `Ты — опытный сурдопереводчик. Ты владеешь русским жестовым языком (РЖЯ) и казахским жестовым языком (они очень близки и во многом совпадают), а также распространёнными международными и бытовыми жестами и дактилем (ручной азбукой, русской и казахской).

Тебе присылают несколько кадров с веб-камеры. Это кадры ОДНОГО жеста в хронологическом порядке (первый кадр — начало движения, последний — конец). Анализируй форму кисти, ориентацию ладони, место исполнения (у лица, у груди, в нейтральном пространстве), направление и характер движения между кадрами, а также мимику.

Задача: определить наиболее вероятное значение этого жеста и вернуть его по-русски естественным текстом.
- Если передан контекст (ранее распознанные слова), используй его, чтобы выбрать подходящее значение и грамматическую форму, чтобы фраза звучала связно. Но выводи только значение ЭТОГО жеста (слово или короткую фразу), не повторяй контекст.
- Если это буква дактиля — верни эту букву.
- Никогда не придумывай длинных предложений и не домысливай то, чего нет в кадрах.
- Если жест неясен, руки не видны, или человек просто сидит/опускает руки — верни пустой text и confidence 0 (или очень низкую).
- confidence — число от 0 до 1, твоя честная оценка уверенности.
- gloss — глосса жеста ЗАГЛАВНЫМИ буквами (например, ПРИВЕТ), либо пустая строка.
- note — очень короткое пояснение на русском (что видно: форма руки, движение), либо пустая строка.`;

const OUTPUT_SCHEMA = {
  type: "object",
  properties: {
    text: { type: "string", description: "Meaning of the gesture in natural Russian, or empty string" },
    confidence: { type: "number", description: "0..1" },
    gloss: { type: "string", description: "Gloss in UPPERCASE, or empty string" },
    note: { type: "string", description: "Short explanation in Russian, or empty string" },
  },
  required: ["text", "confidence", "gloss", "note"],
  additionalProperties: false,
};

const LANGUAGE_HINT = {
  auto: "не указан (определи сам: РЖЯ, казахский жестовый язык или международный/бытовой жест)",
  rsl: "русский жестовый язык (РЖЯ)",
  ksl: "казахский жестовый язык",
};

// ------------------------------------------------------------ validation ---

class InputError extends Error {}

function parseTranslateBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new InputError("Тело запроса должно быть JSON-объектом.");
  }
  const { frames, context = [], signLanguage = "auto" } = body;

  if (!Array.isArray(frames) || frames.length < MIN_FRAMES || frames.length > MAX_FRAMES) {
    throw new InputError(`"frames" должен быть массивом из ${MIN_FRAMES}–${MAX_FRAMES} изображений.`);
  }
  const images = frames.map((frame, i) => {
    if (typeof frame !== "string" || frame.length > MAX_FRAME_CHARS) {
      throw new InputError(`Кадр ${i + 1}: ожидается data URL не длиннее ${MAX_FRAME_CHARS} символов.`);
    }
    const m = DATA_URL_RE.exec(frame);
    if (!m || m[2].length < 100) {
      throw new InputError(`Кадр ${i + 1}: ожидается data:image/jpeg;base64,... или data:image/png;base64,...`);
    }
    return { mediaType: m[1], data: m[2] };
  });

  if (!Array.isArray(context) || !context.every((c) => typeof c === "string")) {
    throw new InputError('"context" должен быть массивом строк.');
  }
  const ctx = context
    .slice(-MAX_CONTEXT_ITEMS)
    .map((c) => c.trim().slice(0, MAX_CONTEXT_CHARS))
    .filter(Boolean);

  if (typeof signLanguage !== "string" || !SIGN_LANGUAGES.has(signLanguage)) {
    throw new InputError('"signLanguage" должен быть "auto", "rsl" или "ksl".');
  }
  return { images, context: ctx, signLanguage };
}

// -------------------------------------------------------------- Claude ----

async function translateGesture({ images, context, signLanguage }) {
  const content = images.map((img) => ({
    type: "image",
    source: { type: "base64", media_type: img.mediaType, data: img.data },
  }));
  content.push({
    type: "text",
    text: [
      `Кадров: ${images.length} (один жест, в хронологическом порядке).`,
      `Контекст (предыдущие распознанные слова): ${context.length ? context.join(" ") : "нет"}.`,
      `Жестовый язык: ${LANGUAGE_HINT[signLanguage]}.`,
      "Определи значение этого жеста.",
    ].join("\n"),
  });

  // Server-side refusal fallback: if the model declines, the API re-runs the
  // request on Anthropic's recommended fallback model within the same call.
  const response = await getClient().beta.messages.create({
    model: MODEL,
    max_tokens: 4000,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    system: SYSTEM_PROMPT,
    output_config: {
      effort: "low",
      format: { type: "json_schema", schema: OUTPUT_SCHEMA },
    },
    messages: [{ role: "user", content }],
  });

  if (response.stop_reason === "refusal") {
    return { text: "", confidence: 0, gloss: "", note: "Модель отказалась интерпретировать эти кадры." };
  }

  const textBlock = response.content.find((b) => b.type === "text");
  if (!textBlock) throw new Error(`No text block in response (stop_reason=${response.stop_reason})`);
  let parsed;
  try {
    parsed = JSON.parse(textBlock.text);
  } catch {
    throw new Error(`Model returned non-JSON output (stop_reason=${response.stop_reason})`);
  }

  const confidence = Number(parsed.confidence);
  const text = typeof parsed.text === "string" ? parsed.text.trim() : "";
  return {
    text,
    confidence: text && Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : 0,
    gloss: typeof parsed.gloss === "string" ? parsed.gloss.trim() : "",
    note: typeof parsed.note === "string" ? parsed.note.trim() : "",
  };
}

// ------------------------------------------------------------ rate limit ---

const hits = new Map(); // ip -> { count, start }
function rateLimit(req, res, next) {
  const now = Date.now();
  const ip = req.ip || req.socket.remoteAddress || "unknown";
  let entry = hits.get(ip);
  if (!entry || now - entry.start >= RATE_WINDOW_MS) {
    entry = { count: 0, start: now };
    hits.set(ip, entry);
  }
  entry.count += 1;
  if (entry.count > RATE_LIMIT) {
    res.set("Retry-After", String(Math.ceil((entry.start + RATE_WINDOW_MS - now) / 1000)));
    return res.status(429).json({ error: "Слишком много запросов. Подождите минуту." });
  }
  next();
}
setInterval(() => {
  const now = Date.now();
  for (const [ip, e] of hits) if (now - e.start >= RATE_WINDOW_MS) hits.delete(ip);
}, RATE_WINDOW_MS).unref();

// ------------------------------------------------------------------ app ---

const app = express();
app.disable("x-powered-by");

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, ai: hasCredentials() });
});

app.post("/api/translate", rateLimit, express.json({ limit: "15mb" }), async (req, res) => {
  let input;
  try {
    input = parseTranslateBody(req.body);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
  if (!hasCredentials()) {
    return res.status(503).json({
      error: "API-кілт бапталмаған / API-ключ не настроен (ANTHROPIC_API_KEY в .env).",
    });
  }

  try {
    res.json(await translateGesture(input));
  } catch (err) {
    if (err instanceof Anthropic.RateLimitError) {
      return res.status(429).json({ error: "Лимит запросов к Claude API исчерпан. Попробуйте позже." });
    }
    if (err instanceof Anthropic.AuthenticationError || err instanceof Anthropic.PermissionDeniedError) {
      console.error("[translate] auth error:", err.status, err.message);
      return res.status(503).json({ error: "Неверный или недействительный API-ключ Anthropic." });
    }
    if (err instanceof Anthropic.APIError) {
      console.error("[translate] API error:", err.status, err.message);
      return res.status(502).json({ error: `Ошибка Claude API (${err.status ?? "network"}).` });
    }
    console.error("[translate] error:", err);
    return res.status(502).json({ error: "Не удалось получить ответ от модели." });
  }
});

app.use("/api", (_req, res) => res.status(404).json({ error: "Not found" }));

app.use("/vendor/tasks-vision", express.static(VISION_DIR, { fallthrough: false }));

app.get("/models/hand_landmarker.task", (_req, res) => {
  if (!existsSync(MODEL_FILE)) {
    return res.status(404).type("text/plain").send("Model not found. Run: npm run setup");
  }
  res.sendFile(MODEL_FILE, { maxAge: "7d" });
});

app.use(express.static(PUBLIC_DIR));

// JSON errors (malformed JSON body, payload too large, ...)
app.use((err, req, res, _next) => {
  const status = err.status || err.statusCode || 500;
  if (req.path.startsWith("/api/")) {
    const message =
      err.type === "entity.too.large"
        ? "Запрос слишком большой (максимум 15 МБ)."
        : status < 500
          ? "Некорректный запрос: ожидается JSON."
          : "Внутренняя ошибка сервера.";
    return res.status(status === 413 ? 413 : status < 500 ? 400 : 500).json({ error: message });
  }
  if (status >= 500) console.error(err);
  res.status(status).type("text/plain").send(status === 404 ? "Not found" : "Error");
});

app.listen(PORT, () => {
  console.log(`\n  Ым тілі аудармашысы / Переводчик жестов`);
  console.log(`  → http://localhost:${PORT}\n`);
  if (!hasCredentials()) {
    console.warn(
      "  ⚠ ANTHROPIC_API_KEY табылмады — «AI аудармашы» режимі жұмыс істемейді.\n" +
        "    .env файлына кілтті қосыңыз (cp .env.example .env).\n" +
        "  ⚠ ANTHROPIC_API_KEY не найден — режим «AI» не работает. Добавьте ключ в .env.\n" +
        "    «Менің жестерім» режимі кілтсіз жұмыс істейді / режим своих жестов работает без ключа.\n"
    );
  }
  if (!existsSync(MODEL_FILE)) {
    console.warn("  ⚠ public/models/hand_landmarker.task жоқ / отсутствует — запустите: npm run setup\n");
  }
});
