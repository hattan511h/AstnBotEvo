import 'dotenv/config';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { Readable } from 'node:stream';
import { writeFileSync, existsSync } from 'node:fs';
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
const YTDLP_PATH = YTDLP_ENV || (existsSync(LOCAL_YTDLP) ? LOCAL_YTDLP : 'yt-dlp');

// وسائط مشتركة: JS runtime (يوتيوب يحتاجه) + كوكيز اختيارية (لو يوتيوب حظر السيرفر)
const YT_ARGS = [];
if (YTDLP_JS_RUNTIME) YT_ARGS.push('--js-runtimes', YTDLP_JS_RUNTIME);
if (YT_COOKIES_B64) {
  const cookiePath = join(tmpdir(), 'yt-cookies.txt');
  writeFileSync(cookiePath, Buffer.from(YT_COOKIES_B64, 'base64'));
  YT_ARGS.push('--cookies', cookiePath);
}

if (!DISCORD_TOKEN || !OPENROUTER_API_KEY) {
  console.error('Missing DISCORD_TOKEN or OPENROUTER_API_KEY in .env');
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

async function askModel(wavBuffer) {
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
  const text = data.choices?.[0]?.message?.content ?? '';
  const m = text.match(/\{[\s\S]*\}/);
  return m ? JSON.parse(m[0]) : null;
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

function playNext() {
  const next = queue.shift();
  if (!next) { current = null; return; }
  current = next;
  console.log('▶️  Now playing:', next.title);
  const yt = spawn(YTDLP_PATH, ['-q', ...YT_ARGS, '--no-playlist', '-f', 'bestaudio/best', '-o', '-', next.url],
    { stdio: ['ignore', 'pipe', 'ignore'] });
  yt.on('error', (e) => console.error('yt-dlp error:', e.message));
  next.proc = yt;
  player.play(createAudioResource(yt.stdout, { inputType: StreamType.Arbitrary }));
}

player.on(AudioPlayerStatus.Idle, () => { current?.proc?.kill(); playNext(); });
player.on('error', (e) => console.error('player error:', e.message));

async function enqueue(query) {
  const track = await searchYouTube(query);
  queue.push(track);
  console.log('➕ Queued:', track.title);
  if (player.state.status === AudioPlayerStatus.Idle) playNext();
}

// ---------- رد صوتي (TTS مجاني من OpenRouter) ----------
const ttsPlayer = createAudioPlayer();
let speakChain = Promise.resolve();

async function speakNow(text) {
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
  if (!res.ok) throw new Error(`TTS ${res.status}: ${await res.text()}`);
  const audio = Buffer.from(await res.arrayBuffer());
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
      if (secs < 0.6 || secs > 20 || rms(pcm) < 300) return; // تجاهل الضجيج/الكلام الطويل
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

function connect() {
  connection = joinVoiceChannel({ ...target, selfDeaf: false, selfMute: false });
  connection.subscribe(player);
  listen(connection);
  const conn = connection;
  conn.on(VoiceConnectionStatus.Disconnected, async () => {
    try {
      await Promise.race([
        entersState(conn, VoiceConnectionStatus.Signalling, 5_000),
        entersState(conn, VoiceConnectionStatus.Connecting, 5_000),
      ]);
    } catch {
      conn.destroy();
      if (target) setTimeout(connect, 3_000); // يرجع يدخل لو انطرد/انقطع
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
  } else if (cmd === '!play' && rest.length) {
    try { await enqueue(rest.join(' ')); msg.react('🎶'); }
    catch { msg.reply('ما لقيت شي 😕'); }
  }
});

client.login(DISCORD_TOKEN);
