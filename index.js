import 'dotenv/config';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { Readable } from 'node:stream';
import { writeFileSync, existsSync, chmodSync, readdirSync, unlinkSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Client, GatewayIntentBits, Events, SlashCommandBuilder, MessageFlags } from 'discord.js';
import {
  joinVoiceChannel, entersState, VoiceConnectionStatus, EndBehaviorType,
  createAudioPlayer, createAudioResource, AudioPlayerStatus,
  NoSubscriberBehavior, StreamType,
} from '@discordjs/voice';
import OpusScript from 'opusscript';

// شبكة أمان: خطأ عابر ما يطيّح البوت
process.on('uncaughtException', (e) => console.error('uncaught:', e?.message || e));
process.on('unhandledRejection', (e) => console.error('unhandled rejection:', e?.message || e));

const execFileP = promisify(execFile);

const {
  DISCORD_TOKEN, OPENROUTER_API_KEY, GUILD_ID, VOICE_CHANNEL_ID,
  MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
  TTS_MODEL = 'fish-audio/s2.1-pro-free',
  TTS_VOICE = '',
  VOICE_REPLIES = 'true',
  WAKE_WORDS = 'بوت, يا بوت, bot, hey bot',
  YTDLP_PATH: YTDLP_ENV,
  YTDLP_JS_RUNTIME = 'node',
  YT_COOKIES_B64,
} = process.env;

// yt-dlp: المحمّل وقت npm install، أو الموجود في PATH
const LOCAL_YTDLP = fileURLToPath(new URL(process.platform === 'win32' ? './yt-dlp.exe' : './yt-dlp', import.meta.url));
let YTDLP_PATH = YTDLP_ENV || (existsSync(LOCAL_YTDLP) ? LOCAL_YTDLP : 'yt-dlp');

// وسائط مشتركة: JS runtime (يوتيوب يحتاجه) + كوكيز اختيارية (لو يوتيوب حظر السيرفر)
const YT_ARGS = [];
if (YTDLP_JS_RUNTIME) {
  // مسار node الفعلي (yt-dlp أحياناً ما يلقاه من PATH)
  YT_ARGS.push('--js-runtimes', YTDLP_JS_RUNTIME === 'node' ? `node:${process.execPath}` : YTDLP_JS_RUNTIME);
}
if (process.env.YTDLP_EXTRA_ARGS) YT_ARGS.push(...process.env.YTDLP_EXTRA_ARGS.split(/\s+/).filter(Boolean));
if (YT_COOKIES_B64) {
  const cookiePath = join(tmpdir(), 'yt-cookies.txt');
  writeFileSync(cookiePath, Buffer.from(YT_COOKIES_B64, 'base64'));
  YT_ARGS.push('--cookies', cookiePath);
}

if (!DISCORD_TOKEN || !(OPENROUTER_API_KEY || process.env.GEMINI_API_KEY)) {
  console.error('Missing DISCORD_TOKEN, and either GEMINI_API_KEY or OPENROUTER_API_KEY');
  process.exit(1);
}

// ---------- 1) المودل: صوت -> JSON ----------
const SYSTEM_PROMPT = `You are the ears of a Discord music bot.
You receive a short voice clip (Gulf Arabic dialect, Arabic, or English).
Reply with ONLY a JSON object, no markdown, in this shape:
{"heard":"<transcript>","addressed":true|false,"action":"play|skip|stop|pause|resume|none","query":"<YouTube search text>","reply":"<short spoken reply>"}
Rules:
- addressed is true ONLY if the speaker calls the bot by one of these wake words: ${WAKE_WORDS}.
- If not addressed, action must be "none" and reply must be "".
- reply is ONE very short friendly sentence in the same language/dialect the speaker used (e.g. confirming what you will do). No emojis.
- For "play", query is the song/artist/video to search on YouTube (keep Arabic names in Arabic). Otherwise query is "".
- "skip" = next song, "stop" = stop and clear the queue, "pause"/"resume" as expected.`;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const DEFAULT_GEMINI_MODELS = ['gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite'];
// لكل موديل حد يومي مجاني خاص فيه، فلو خلص واحد ننتقل للي بعده
const GEMINI_MODELS = process.env.GEMINI_MODELS
  ? process.env.GEMINI_MODELS.split(',').map((m) => m.trim()).filter(Boolean)
  : [...new Set([process.env.GEMINI_MODEL, ...DEFAULT_GEMINI_MODELS].filter(Boolean))];
const geminiBlocked = new Map();
const GEMINI_THINKING = process.env.GEMINI_THINKING ?? 'low'; // low | medium | high | ''(بدون)
const GEMINI_TTS_MODEL = process.env.GEMINI_TTS_MODEL || 'gemini-2.5-flash-preview-tts';
const GEMINI_VOICE = process.env.GEMINI_VOICE || 'Kore';
const GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

function extractJson(text) {
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) console.warn('model returned no JSON:', text.slice(0, 300));
  return m ? JSON.parse(m[0]) : null;
}

async function askGeminiWith(model, wavBuffer) {
  const call = (withThinking) => fetch(`${GEMINI_URL}/${model}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{
        role: 'user',
        parts: [
          { text: 'Classify this voice clip.' },
          { inlineData: { mimeType: 'audio/wav', data: wavBuffer.toString('base64') } },
        ],
      }],
      generationConfig: {
        responseMimeType: 'application/json',
        ...(withThinking && GEMINI_THINKING ? { thinkingConfig: { thinkingLevel: GEMINI_THINKING } } : {}),
      },
    }),
    signal: AbortSignal.timeout(30_000),
  });

  let res = await call(true);
  if (res.status === 400) {
    const body = await res.text();
    if (!/thinking/i.test(body)) throw new Error(`Gemini 400: ${body}`);
    console.warn('thinkingConfig rejected, retrying without it');
    res = await call(false);
  }
  if (!res.ok) throw new Error(`Gemini ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const text = (data.candidates?.[0]?.content?.parts ?? []).filter((p) => !p.thought).map((p) => p.text || '').join('');
  return extractJson(text);
}

async function askGemini(wavBuffer) {
  let lastErr;
  for (const model of GEMINI_MODELS) {
    if (Date.now() < (geminiBlocked.get(model) ?? 0)) continue;
    try {
      return await askGeminiWith(model, wavBuffer);
    } catch (e) {
      lastErr = e;
      if (/^Gemini 429/.test(e.message)) {
        geminiBlocked.set(model, Date.now() + 60 * 60 * 1000);
        console.warn(`${model}: free quota exhausted, trying next model`);
      } else if (/^Gemini 404/.test(e.message)) {
        geminiBlocked.set(model, Infinity);
        console.warn(`${model}: not available, skipping`);
      } else {
        throw e;
      }
    }
  }
  throw lastErr ?? new Error('all Gemini models are rate-limited right now');
}

async function askOpenRouter(wavBuffer) {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL,
      temperature: 0,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Classify this voice clip.' },
            { type: 'input_audio', input_audio: { data: wavBuffer.toString('base64'), format: 'wav' } },
          ],
        },
      ],
    }),
  });
  if (!res.ok) throw new Error(`OpenRouter ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return extractJson(data.choices?.[0]?.message?.content ?? '');
}

// إذا فيه GEMINI_API_KEY يستخدم Gemini مباشرة (له طبقة مجانية)، وإلا OpenRouter
async function askModel(wavBuffer) {
  return GEMINI_API_KEY ? askGemini(wavBuffer) : askOpenRouter(wavBuffer);
}

// ---------- 2) الصوت: PCM -> WAV ----------
function pcmToWav(pcm, rate = 48000, ch = 2) {
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVE', 8);
  h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(ch, 22);
  h.writeUInt32LE(rate, 24); h.writeUInt32LE(rate * ch * 2, 28); h.writeUInt16LE(ch * 2, 32);
  h.writeUInt16LE(16, 34); h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([h, pcm]);
}

function rms(pcm) {
  let sum = 0, n = 0;
  for (let i = 0; i + 1 < pcm.length; i += 40) { const s = pcm.readInt16LE(i); sum += s * s; n++; }
  return Math.sqrt(sum / Math.max(n, 1));
}

// تحويل 48kHz ستيريو -> 16kHz مونو (أصغر وأرخص للمودل)
function toMono16k(pcm) {
  const frames = Math.floor(pcm.length / 4);
  const n = Math.floor(frames / 3);
  const out = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    const idx = i * 3 * 4;
    out.writeInt16LE((pcm.readInt16LE(idx) + pcm.readInt16LE(idx + 2)) >> 1, i * 2);
  }
  return out;
}

// ---------- 3) المشغّل (يوتيوب عن طريق yt-dlp) ----------
const player = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Play } });
const queue = [];
let current = null;

async function searchYouTube(query) {
  // رابط مباشر: ما نحتاج بحث (التحميل يتعامل معه)
  if (/^https?:\/\//i.test(query)) return { title: query, url: query };
  try {
    // --flat-playlist يجيب العنوان والمعرّف بدون فتح الفيديو نفسه
    const { stdout } = await execFileP(YTDLP_PATH, [
      ...YT_ARGS, '--flat-playlist', '--print', '%(title)s\n%(id)s', `ytsearch1:${query}`,
    ], { timeout: 30_000 });
    const [title, id] = stdout.trim().split('\n');
    if (!id) throw new Error('no result');
    return { title, url: `https://www.youtube.com/watch?v=${id}` };
  } catch (e) {
    const errs = String(e.stderr || '').split('\n').filter((l) => l.startsWith('ERROR'));
    throw new Error(errs.join(' | ') || e.message);
  }
}

async function ensureYtDlp() {
  try {
    await execFileP(YTDLP_PATH, ['--version'], { timeout: 20_000 });
    console.log('✅ yt-dlp found');
    return;
  } catch { /* مو موجود */ }
  console.log('⬇️  yt-dlp not found, downloading...');
  const win = process.platform === 'win32';
  const asset = win ? 'yt-dlp.exe'
    : process.platform === 'darwin' ? 'yt-dlp_macos'
    : process.arch === 'arm64' ? 'yt-dlp_linux_aarch64' : 'yt-dlp_linux';
  const dest = join(tmpdir(), win ? 'yt-dlp.exe' : 'yt-dlp');
  const res = await fetch(`https://github.com/yt-dlp/yt-dlp/releases/latest/download/${asset}`, { redirect: 'follow' });
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  chmodSync(dest, 0o755);
  YTDLP_PATH = dest;
  console.log('✅ yt-dlp ready at', dest);
}

// ننزّل الأغنية لملف مؤقت أول (أضمن من البث المباشر، وأخطاء yt-dlp تبان في اللوق)
async function runDownload(input, extraArgs) {
  const base = `song-${Date.now()}`;
  const dir = tmpdir();
  await execFileP(YTDLP_PATH, [
    ...YT_ARGS, ...extraArgs, '--no-playlist', '-f', 'bestaudio/best',
    '-o', join(dir, `${base}.%(ext)s`), input,
  ], { timeout: 180_000, maxBuffer: 20 * 1024 * 1024 });
  const f = readdirSync(dir).find((n) => n.startsWith(base) && !n.endsWith('.part') && !n.endsWith('.ytdl'));
  if (!f) throw new Error('downloaded file not found');
  return join(dir, f);
}

// يوتيوب يحظر التحميل من السيرفرات بحسب "عميل" التطبيق، فنجرّب عدة عملاء بالترتيب ونتذكر الناجح
const CLIENT_ATTEMPTS = process.env.YTDLP_EXTRA_ARGS ? [[]] : [
  [],
  ['--extractor-args', 'youtube:player_client=tv'],
  ['--extractor-args', 'youtube:player_client=web_embedded'],
  ['--extractor-args', 'youtube:player_client=mweb'],
  ['--extractor-args', 'youtube:player_client=ios'],
  ['--extractor-args', 'youtube:player_client=android_vr'],
];
const SOUNDCLOUD_FALLBACK = (process.env.SOUNDCLOUD_FALLBACK ?? 'true') === 'true';
let preferredAttempt = 0;
let youtubeBlockedUntil = 0;

function lastErrorLine(e) {
  const msg = String(e.stderr || e.message);
  return (msg.split('\n').filter((l) => l.startsWith('ERROR')).pop() || msg).trim().slice(0, 220);
}

async function downloadAudio(track) {
  let lastErr;
  if (Date.now() >= youtubeBlockedUntil) {
    const order = [preferredAttempt, ...CLIENT_ATTEMPTS.keys()].filter((v, i, a) => a.indexOf(v) === i);
    for (const idx of order) {
      try {
        const file = await runDownload(track.url, CLIENT_ATTEMPTS[idx]);
        preferredAttempt = idx;
        console.log(`✅ YouTube download ok (client attempt ${idx})`);
        return file;
      } catch (e) {
        lastErr = e;
        console.warn(`yt-dlp attempt ${idx} failed: ${lastErrorLine(e)}`);
      }
    }
    youtubeBlockedUntil = Date.now() + 10 * 60 * 1000; // كل المحاولات فشلت: لا تضيّع وقت 10 دقايق
    console.warn('YouTube download blocked from this server; skipping YouTube for 10 minutes');
  }
  if (SOUNDCLOUD_FALLBACK && track.query && !/^https?:\/\//i.test(track.query)) {
    try {
      console.log(`↩️  Trying SoundCloud for: ${track.query}`);
      return await runDownload(`scsearch1:${track.query}`, []);
    } catch (e) {
      lastErr = e;
      console.warn(`SoundCloud failed: ${lastErrorLine(e)}`);
    }
  }
  throw lastErr ?? new Error('no source could download this track');
}

function cleanup(track) {
  if (track?.file) { try { unlinkSync(track.file); } catch { /* ignore */ } track.file = null; }
}

let loading = false;
async function playNext() {
  if (loading) return;
  const next = queue.shift();
  if (!next) { current = null; return; }
  current = next;
  loading = true;
  console.log('⬇️  Downloading:', next.title);
  try {
    next.file = await downloadAudio(next);
  } catch {
    loading = false;
    current = null;
    return playNext(); // الأغنية فشلت: جرّب اللي بعدها
  }
  loading = false;
  if (next.cancelled) { cleanup(next); return playNext(); }
  console.log('▶️  Now playing:', next.title);
  player.play(createAudioResource(next.file, { inputType: StreamType.Arbitrary }));
}

player.on(AudioPlayerStatus.Idle, () => { cleanup(current); playNext(); });
player.on('error', (e) => console.error('player error:', e.message));
player.on('stateChange', (o, n) => console.log(`player: ${o.status} -> ${n.status}`));

async function enqueue(query) {
  const track = await searchYouTube(query);
  track.query = query;
  queue.push(track);
  console.log('➕ Queued:', track.title);
  if (player.state.status === AudioPlayerStatus.Idle) playNext();
  return track;
}

// ---------- رد صوتي (TTS مجاني من OpenRouter) ----------
const ttsPlayer = createAudioPlayer();
let speakChain = Promise.resolve();

async function ttsGemini(text) {
  const res = await fetch(`${GEMINI_URL}/${GEMINI_TTS_MODEL}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY },
    body: JSON.stringify({
      contents: [{ parts: [{ text }] }],
      generationConfig: {
        responseModalities: ['AUDIO'],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: GEMINI_VOICE } } },
      },
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!res.ok) throw new Error(`Gemini TTS ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const part = (data.candidates?.[0]?.content?.parts ?? []).find((p) => p.inlineData);
  if (!part) throw new Error('Gemini TTS returned no audio');
  return pcmToWav(Buffer.from(part.inlineData.data, 'base64'), 24000, 1); // PCM 24kHz مونو
}

// احتياطي مجاني بدون مفتاح (غير رسمي، الصوت بسيط)
async function ttsGoogle(text) {
  const lang = /[\u0600-\u06FF]/.test(text) ? 'ar' : 'en';
  const url = `https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=${lang}&q=${encodeURIComponent(text.slice(0, 200))}`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' }, signal: AbortSignal.timeout(15_000) });
  if (!res.ok) throw new Error(`Google TTS ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function ttsOpenRouter(text) {
  const res = await fetch('https://openrouter.ai/api/v1/audio/speech', {
    method: 'POST',
    headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: TTS_MODEL,
      input: text,
      response_format: 'mp3',
      ...(TTS_VOICE ? { voice: TTS_VOICE } : {}),
    }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!res.ok) throw new Error(`OpenRouter TTS ${res.status}: ${await res.text()}`);
  return Buffer.from(await res.arrayBuffer());
}

let geminiTtsBlockedUntil = 0;
async function synthesize(text) {
  const forced = process.env.TTS_PROVIDER;
  const order = forced ? [forced] : [GEMINI_API_KEY && 'gemini', 'google'].filter(Boolean);
  const impl = { gemini: ttsGemini, google: ttsGoogle, openrouter: ttsOpenRouter };
  let lastErr;
  for (const name of order) {
    if (name === 'gemini' && Date.now() < geminiTtsBlockedUntil) continue;
    try {
      const audio = await impl[name](text);
      console.log(`🔊 TTS ok via ${name} (${audio.length} bytes)`);
      return audio;
    } catch (e) {
      lastErr = e;
      if (name === 'gemini' && /\b429\b/.test(e.message)) {
        geminiTtsBlockedUntil = Date.now() + 60 * 60 * 1000; // حد Gemini اليومي خلص: تجاهله ساعة
        console.warn('Gemini TTS quota exhausted, using Google TTS for the next hour');
      } else {
        console.warn(`TTS ${name} failed: ${String(e.message).slice(0, 300)}`);
      }
    }
  }
  throw lastErr;
}

async function speakNow(text) {
  const audio = await synthesize(text);
  if (!connection) return;

  const wasPlaying = player.state.status === AudioPlayerStatus.Playing;
  if (wasPlaying) player.pause();          // وقّف الموسيقى مؤقتاً
  connection.subscribe(ttsPlayer);         // الرد ياخذ القناة
  ttsPlayer.play(createAudioResource(Readable.from(audio), { inputType: StreamType.Arbitrary }));
  await new Promise((resolve) => {
    const t = setTimeout(resolve, 30_000);
    ttsPlayer.once(AudioPlayerStatus.Idle, () => { clearTimeout(t); resolve(); });
  });
  connection.subscribe(player);            // رجّع الموسيقى
  if (wasPlaying) player.unpause();
}

function say(text) {
  if (VOICE_REPLIES !== 'true' || !text) return Promise.resolve();
  speakChain = speakChain.then(() => speakNow(text)).catch((e) => console.error('TTS failed:', e.message));
  return speakChain;
}

async function runIntent(intent) {
  switch (intent.action) {
    case 'play': if (intent.query) await enqueue(intent.query); break;
    case 'skip': if (current) current.cancelled = true; player.stop(true); break;
    case 'stop': queue.length = 0; if (current) current.cancelled = true; player.stop(true); break;
    case 'pause': player.pause(); break;
    case 'resume': player.unpause(); break;
  }
}

// ---------- 4) الاتصال بالروم + الاستماع ----------
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

const COMMANDS = [
  new SlashCommandBuilder().setName('play').setDescription('شغّل أغنية من يوتيوب')
    .addStringOption((o) => o.setName('query').setDescription('اسم الأغنية أو رابط يوتيوب').setRequired(true)),
  new SlashCommandBuilder().setName('skip').setDescription('تخطَّ الأغنية الحالية'),
  new SlashCommandBuilder().setName('stop').setDescription('أوقف الموسيقى وامسح القائمة'),
].map((c) => c.toJSON());

function registerCommands(guild) {
  return guild.commands.set(COMMANDS)
    .then(() => console.log(`✅ Slash commands registered in "${guild.name}"`))
    .catch((e) => console.error(`Slash registration failed in "${guild.name}": ${e.message} — re-invite the bot with the applications.commands scope`));
}

const ALLOWED_USERS = new Set((process.env.ALLOWED_USER_IDS || '').split(',').map((x) => x.trim()).filter(Boolean));
let target = null;      // { guildId, channelId, adapterCreator }
let connection = null;

function listen(conn) {
  const receiver = conn.receiver;
  const busy = new Set();

  receiver.speaking.on('start', (userId) => {
    if (busy.has(userId) || userId === client.user.id) return;
    if (ALLOWED_USERS.size && !ALLOWED_USERS.has(userId)) return;
    busy.add(userId);

    const opusStream = receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 1200 },
    });
    const decoder = new OpusScript(48000, 2, OpusScript.Application.AUDIO);
    const chunks = [];
    let bad = 0;
    opusStream.on('data', (packet) => {
      try { chunks.push(Buffer.from(decoder.decode(packet))); } catch { bad++; } // تخطَّ الحزمة التالفة
    });
    opusStream.on('error', (e) => console.warn('receive stream error:', e.message));

    opusStream.on('end', async () => {
      busy.delete(userId);
      try { decoder.delete(); } catch { /* ignore */ }
      if (bad) console.log(`   (skipped ${bad} bad opus packets)`);
      const pcm = Buffer.concat(chunks);
      const secs = pcm.length / (48000 * 2 * 2);
      const level = Math.round(rms(pcm));
      console.log(`🎙️ clip from ${userId}: ${secs.toFixed(1)}s, level ${level}`);
      if (secs < 0.6 || secs > 20 || level < 300) { console.log('   (skipped: too short/long/quiet)'); return; }
      try {
        const intent = await askModel(pcmToWav(toMono16k(pcm), 16000, 1));
        console.log(`🎤 ${userId}:`, intent);
        if (intent?.addressed) {
          if (intent.reply) await say(intent.reply);
          if (intent.action !== 'none') await runIntent(intent);
        }
      } catch (e) {
        console.error('voice command failed:', e.message);
      }
    });
  });
}

let reconnectTimer = null;
function scheduleReconnect(conn) {
  if (conn !== connection) return;
  if (conn.state.status !== VoiceConnectionStatus.Destroyed) conn.destroy();
  if (!target || reconnectTimer) return;
  reconnectTimer = setTimeout(() => { reconnectTimer = null; if (target) connect(); }, 5_000);
}

function connect() {
  connection = joinVoiceChannel({ ...target, selfDeaf: false, selfMute: false });
  connection.subscribe(player);
  listen(connection);
  const conn = connection;
  conn.on('error', (e) => { console.error('voice connection error:', e.message); scheduleReconnect(conn); });
  entersState(conn, VoiceConnectionStatus.Ready, 20_000)
    .then(() => { console.log('✅ Voice connection ready'); say('تم، أنا جاهز'); })
    .catch((e) => { console.error('voice connection not ready:', e.message); scheduleReconnect(conn); });
  conn.on(VoiceConnectionStatus.Disconnected, async () => {
    try {
      await Promise.race([
        entersState(conn, VoiceConnectionStatus.Signalling, 5_000),
        entersState(conn, VoiceConnectionStatus.Connecting, 5_000),
      ]);
    } catch {
      scheduleReconnect(conn); // يرجع يدخل لو انطرد/انقطع
    }
  });
}

function leave() {
  target = null;
  queue.length = 0;
  player.stop(true);
  if (connection && connection.state.status !== VoiceConnectionStatus.Destroyed) connection.destroy();
}

client.once(Events.ClientReady, async () => {
  console.log(`✅ Logged in as ${client.user.tag}`);
  await ensureYtDlp().catch((e) => console.error('yt-dlp setup failed:', e.message));
  for (const g of client.guilds.cache.values()) registerCommands(g);
  if (GUILD_ID && VOICE_CHANNEL_ID) {
    const guild = await client.guilds.fetch(GUILD_ID);
    target = { guildId: guild.id, channelId: VOICE_CHANNEL_ID, adapterCreator: guild.voiceAdapterCreator };
    connect();
  }
});

// أوامر نصية احتياطية: !join  !leave  !play اسم الاغنية
client.on(Events.MessageCreate, async (msg) => {
  if (msg.author.bot || !msg.guild) return;
  const [cmd, ...rest] = msg.content.trim().split(/\s+/);

  if (cmd === '!join') {
    const ch = msg.member?.voice?.channel;
    if (!ch) return msg.reply('ادخل روم صوتي أول 🎧');
    if (connection && connection.state.status !== VoiceConnectionStatus.Destroyed) connection.destroy();
    target = { guildId: msg.guild.id, channelId: ch.id, adapterCreator: msg.guild.voiceAdapterCreator };
    connect();
    msg.react('✅');
  } else if (cmd === '!leave') {
    leave();
    msg.react('👋');
  } else if (cmd === '!say' && rest.length) {
    say(rest.join(' '));
    msg.react('🔊');
  } else if (cmd === '!play' && rest.length) {
    try { await enqueue(rest.join(' ')); msg.react('🎶'); }
    catch { msg.reply('ما لقيت شي 😕'); }
  }
});

client.on(Events.GuildCreate, (g) => registerCommands(g));

client.on(Events.InteractionCreate, async (i) => {
  if (!i.isChatInputCommand() || !i.guild) return;

  const reply = (content, ephemeral = false) => (i.deferred || i.replied
    ? i.editReply({ content })
    : i.reply({ content, ...(ephemeral ? { flags: MessageFlags.Ephemeral } : {}) }));

  try {
    const alive = connection && connection.state.status !== VoiceConnectionStatus.Destroyed && target;
    const userVc = i.member?.voice?.channelId;

    if (i.commandName === 'play') {
      if (!alive && !userVc) return await reply('ادخل روم صوتي أول 🎧', true);
      if (alive && userVc !== target.channelId) return await reply('لازم تكون معي في الروم الصوتي 🎧', true);
      if (!alive) {
        target = { guildId: i.guild.id, channelId: userVc, adapterCreator: i.guild.voiceAdapterCreator };
        connect();
      }
      await i.deferReply();
      const wasBusy = Boolean(current) || queue.length > 0;
      try {
        const track = await enqueue(i.options.getString('query', true));
        return await reply(`${wasBusy ? '➕ أضفتها للقائمة' : '🎶 بشغّل'}: **${track.title}**`);
      } catch (e) {
        console.error('/play failed:', e.message);
        return await reply('ما لقيت شي 😕');
      }
    }

    if (i.commandName === 'skip') {
      if (!alive || !current) return await reply('ما فيه شي شغال حالياً', true);
      if (userVc !== target.channelId) return await reply('لازم تكون معي في الروم الصوتي 🎧', true);
      const title = current.title;
      await runIntent({ action: 'skip' });
      return await reply(`⏭️ تخطيت: **${title}**`);
    }

    if (i.commandName === 'stop') {
      if (!alive) return await reply('البوت مو في روم صوتي', true);
      if (userVc !== target.channelId) return await reply('لازم تكون معي في الروم الصوتي 🎧', true);
      await runIntent({ action: 'stop' });
      return await reply('⏹️ تم الإيقاف ومسحت القائمة');
    }
  } catch (e) {
    console.error('slash command failed:', e.message);
    try { await reply('صار خطأ 😕', true); } catch { /* ignore */ }
  }
});

client.login(DISCORD_TOKEN);
