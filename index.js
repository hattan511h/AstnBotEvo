import 'dotenv/config';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { Readable } from 'node:stream';
import { writeFileSync, existsSync, chmodSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Client, GatewayIntentBits, Events } from 'discord.js';
import {
  joinVoiceChannel, entersState, VoiceConnectionStatus, EndBehaviorType,
  createAudioPlayer, createAudioResource, AudioPlayerStatus,
  NoSubscriberBehavior, StreamType,
} from '@discordjs/voice';
import prism from 'prism-media';

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
if (YTDLP_JS_RUNTIME) YT_ARGS.push('--js-runtimes', YTDLP_JS_RUNTIME);
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
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const GEMINI_THINKING = process.env.GEMINI_THINKING ?? 'low'; // low | medium | high | ''(بدون)
const GEMINI_TTS_MODEL = process.env.GEMINI_TTS_MODEL || 'gemini-2.5-flash-preview-tts';
const GEMINI_VOICE = process.env.GEMINI_VOICE || 'Kore';
const GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

function extractJson(text) {
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) console.warn('model returned no JSON:', text.slice(0, 300));
  return m ? JSON.parse(m[0]) : null;
}

async function askGemini(wavBuffer) {
  const call = (withThinking) => fetch(`${GEMINI_URL}/${GEMINI_MODEL}:generateContent`, {
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
  const { stdout } = await execFileP(YTDLP_PATH, [
    ...YT_ARGS, '--no-playlist', '--print', '%(title)s\n%(id)s', `ytsearch1:${query}`,
  ], { timeout: 30_000 });
  const [title, id] = stdout.trim().split('\n');
  if (!id) throw new Error('no result');
  return { title, url: `https://www.youtube.com/watch?v=${id}` };
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

function playNext() {
  const next = queue.shift();
  if (!next) { current = null; return; }
  current = next;
  console.log('▶️  Now playing:', next.title);
  const yt = spawn(YTDLP_PATH, [...YT_ARGS, '--no-playlist', '-f', 'bestaudio/best', '-o', '-', next.url],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  let errBuf = '';
  let bytes = 0;
  yt.stderr.on('data', (d) => { errBuf = (errBuf + d).slice(-4000); });
  yt.stdout.on('data', (d) => { bytes += d.length; });
  yt.on('error', (e) => console.error('yt-dlp spawn error:', e.message));
  yt.on('close', (code, sig) => {
    console.log(`yt-dlp exited: code=${code} signal=${sig} audioBytes=${bytes}`);
    if (code && errBuf) console.error('yt-dlp stderr:\n' + errBuf.trim().slice(-1200));
  });
  next.proc = yt;
  player.play(createAudioResource(yt.stdout, { inputType: StreamType.Arbitrary }));
}

player.on(AudioPlayerStatus.Idle, () => { current?.proc?.kill(); playNext(); });
player.on('error', (e) => console.error('player error:', e.message));
player.on('stateChange', (o, n) => console.log(`player: ${o.status} -> ${n.status}`));

async function enqueue(query) {
  const track = await searchYouTube(query);
  queue.push(track);
  console.log('➕ Queued:', track.title);
  if (player.state.status === AudioPlayerStatus.Idle) playNext();
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

async function synthesize(text) {
  const forced = process.env.TTS_PROVIDER;
  const order = forced ? [forced] : [GEMINI_API_KEY && 'gemini', 'google'].filter(Boolean);
  const impl = { gemini: ttsGemini, google: ttsGoogle, openrouter: ttsOpenRouter };
  let lastErr;
  for (const name of order) {
    try {
      const audio = await impl[name](text);
      console.log(`🔊 TTS ok via ${name} (${audio.length} bytes)`);
      return audio;
    } catch (e) {
      lastErr = e;
      console.warn(`TTS ${name} failed: ${e.message}`);
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
    case 'skip': current?.proc?.kill(); player.stop(true); break;
    case 'stop': queue.length = 0; current?.proc?.kill(); player.stop(true); break;
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

let target = null;      // { guildId, channelId, adapterCreator }
let connection = null;

function listen(conn) {
  const receiver = conn.receiver;
  const busy = new Set();

  receiver.speaking.on('start', (userId) => {
    if (busy.has(userId) || userId === client.user.id) return;
    busy.add(userId);

    const opusStream = receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 1200 },
    });
    const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
    const chunks = [];
    opusStream.pipe(decoder);
    decoder.on('data', (c) => chunks.push(c));
    opusStream.on('error', () => busy.delete(userId));

    decoder.on('end', async () => {
      busy.delete(userId);
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

client.login(DISCORD_TOKEN);
