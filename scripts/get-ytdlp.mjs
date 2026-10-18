// يحمّل yt-dlp (نسخة standalone ما تحتاج Python) وقت npm install
import { writeFileSync, chmodSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

let asset;
if (process.platform === 'linux') asset = process.arch === 'arm64' ? 'yt-dlp_linux_aarch64' : 'yt-dlp_linux';
else if (process.platform === 'darwin') asset = 'yt-dlp_macos';
else if (process.platform === 'win32') asset = 'yt-dlp.exe';

if (!asset) { console.log('get-ytdlp: unsupported platform, skipping'); process.exit(0); }

const dest = fileURLToPath(new URL(process.platform === 'win32' ? '../yt-dlp.exe' : '../yt-dlp', import.meta.url));
const url = `https://github.com/yt-dlp/yt-dlp/releases/latest/download/${asset}`;

try {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  chmodSync(dest, 0o755);
  console.log(`get-ytdlp: downloaded ${asset}`);
} catch (e) {
  console.warn('get-ytdlp: download failed (install yt-dlp manually):', e.message);
}
