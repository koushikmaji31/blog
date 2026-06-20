// Post-build: generate a static HTML file per blog post with its own
// Open Graph / Twitter meta tags, so link-preview crawlers (LinkedIn, X,
// Slack, ...) get real per-post metadata. Humans still boot the SPA, which
// renders the correct post from the URL.
//
// Output: dist/blog/<id>.html  (served by GitHub Pages at /blog/blog/<id>)
//
// No browser / puppeteer needed — we just string-inject into the built
// dist/index.html template.

import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const blogsDir = join(root, 'src/content/blogs');
const distDir = join(root, 'dist');

const ORIGIN = 'https://koushikmaji31.github.io';
const BASE = '/blog'; // vite `base`, without trailing slash

function parseFrontmatter(md) {
  const m = md.match(/^---\s*\n([\s\S]*?)\n---/);
  if (!m) return {};
  const data = {};
  for (const line of m[1].split('\n')) {
    const i = line.indexOf(':');
    if (i < 0) continue;
    const key = line.slice(0, i).trim();
    const val = line.slice(i + 1).trim().replace(/^["']|["']$/g, '');
    data[key] = val;
  }
  return data;
}

function esc(s = '') {
  return s
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function setMeta(html, { title, description, url, image }) {
  const t = esc(title);
  const d = esc(description);
  const u = esc(url);
  const img = esc(image);
  let out = html;
  out = out.replace(/<title>[\s\S]*?<\/title>/, `<title>${t}</title>`);
  out = out.replace(/(<meta name="description" content=")[^"]*(")/, `$1${d}$2`);
  out = out.replace(/(<meta property="og:title" content=")[^"]*(")/, `$1${t}$2`);
  out = out.replace(/(<meta property="og:description" content=")[^"]*(")/, `$1${d}$2`);
  out = out.replace(/(<meta property="og:type" content=")[^"]*(")/, `$1article$2`);
  out = out.replace(/(<meta property="og:url" content=")[^"]*(")/, `$1${u}$2`);
  out = out.replace(/(<meta property="og:image" content=")[^"]*(")/, `$1${img}$2`);
  out = out.replace(/(<meta name="twitter:title" content=")[^"]*(")/, `$1${t}$2`);
  out = out.replace(/(<meta name="twitter:description" content=")[^"]*(")/, `$1${d}$2`);
  out = out.replace(/(<meta name="twitter:image" content=")[^"]*(")/, `$1${img}$2`);
  return out;
}

const template = readFileSync(join(distDir, 'index.html'), 'utf8');
const files = readdirSync(blogsDir).filter((f) => f.endsWith('.md'));

mkdirSync(join(distDir, 'blog'), { recursive: true });

let count = 0;
for (const file of files) {
  const id = file.replace(/\.md$/, '');
  const fm = parseFrontmatter(readFileSync(join(blogsDir, file), 'utf8'));
  const title = fm.title || id;
  const description = fm.excerpt || '';
  const url = `${ORIGIN}${BASE}/blog/${id}`;
  const perPostImg = `og/${id}.png`;
  const image = existsSync(join(distDir, perPostImg))
    ? `${ORIGIN}${BASE}/${perPostImg}`
    : `${ORIGIN}${BASE}/og/default.png`;

  const out = setMeta(template, { title, description, url, image });
  writeFileSync(join(distDir, 'blog', `${id}.html`), out);
  count++;
}

console.log(`prerender-meta: wrote ${count} per-post HTML file(s) to dist/blog/`);
