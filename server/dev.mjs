/**
 * Local dev adapter: same worker code, Node's http server instead of workerd.
 *   node server/dev.mjs          -> http://0.0.0.0:3000
 */
import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import worker from "../worker/index.js";

const PUBLIC = fileURLToPath(new URL("../public/", import.meta.url));
const ROOT = fileURLToPath(new URL("../", import.meta.url));
const PORT = Number(process.env.PORT || 3000);

// .env at the repo root, KEY=value, never overrides the real environment
async function loadEnv() {
  const env = { ...process.env };
  try {
    const text = await readFile(join(ROOT, ".env"), "utf8");
    for (const line of text.split("\n")) {
      const t = line.trim();
      if (!t || t.startsWith("#") || !t.includes("=")) continue;
      const [k, ...rest] = t.split("=");
      if (!env[k.trim()]) env[k.trim()] = rest.join("=").trim().replace(/^["']|["']$/g, "");
    }
  } catch {
    /* no .env is fine */
  }
  return env;
}

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".pdf": "application/pdf",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

const env = await loadEnv();
console.log(`[dev] nace_key=${Boolean(env.NACE_API_KEY)} gemini_key=${Boolean(env.GEMINI_API_KEY)}`);

http
  .createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

    if (url.pathname.startsWith("/api/")) {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const body = chunks.length ? Buffer.concat(chunks) : undefined;

      const init = { method: req.method, headers: new Headers(req.headers) };
      if (body && req.method !== "GET" && req.method !== "HEAD") init.body = body;

      const response = await worker.fetch(new Request(url, init), env);
      res.writeHead(response.status, Object.fromEntries(response.headers));
      res.end(Buffer.from(await response.arrayBuffer()));
      return;
    }

    // static
    let path = decodeURIComponent(url.pathname);
    if (path === "/") path = "/index.html";
    const filePath = join(PUBLIC, normalize(path).replace(/^(\.\.[/\\])+/, ""));

    try {
      const info = await stat(filePath);
      if (!info.isFile()) throw new Error("not a file");
      const bytes = await readFile(filePath);
      res.writeHead(200, {
        "content-type": TYPES[extname(filePath)] || "application/octet-stream",
        "cache-control": "no-store",
      });
      res.end(bytes);
    } catch {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("404");
    }
  })
  .listen(PORT, "0.0.0.0", () => console.log(`[dev] listening on http://0.0.0.0:${PORT}`));
