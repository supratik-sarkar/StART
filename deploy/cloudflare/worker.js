/**
 * Cloudflare Worker Gateway for StART v4.5.
 *
 * Directs:
 * - Static Assets -> Cloudflare Static Assets (zero worker executions for cached static files)
 * - Dynamic /api/* -> Oracle Cloud Always Free Origin (with HMAC signature)
 */

async function hmacSha256Hex(secret, message) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return Array.from(new Uint8Array(signature))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Hex(bytes) {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 1. Dynamic API routes: forward to Oracle backend with origin authentication
    if (url.pathname.startsWith("/api/")) {
      const originBase = env.ORACLE_ORIGIN_URL || "http://127.0.0.1:8000";
      const targetUrl = new URL(url.pathname + url.search, originBase);

      const bodyBytes = request.method === "POST" || request.method === "PUT"
        ? await request.arrayBuffer()
        : new Uint8Array(0);

      const timestamp = (Date.now() / 1000).toFixed(0);
      const nonce = crypto.randomUUID();
      const bodyDigest = await sha256Hex(bodyBytes);

      const canonicalString = `${request.method.toUpperCase()}:${url.pathname}:${timestamp}:${nonce}:${bodyDigest}`;
      const secret = env.START_ORIGIN_SECRET || "start-dev-origin-secret-local-only";
      const signature = await hmacSha256Hex(secret, canonicalString);

      const forwardHeaders = new Headers(request.headers);
      forwardHeaders.set("X-StART-Origin-Sig", signature);
      forwardHeaders.set("X-StART-Origin-Ts", timestamp);
      forwardHeaders.set("X-StART-Origin-Nonce", nonce);
      forwardHeaders.set("X-Forwarded-Host", url.hostname);

      const originReq = new Request(targetUrl.toString(), {
        method: request.method,
        headers: forwardHeaders,
        body: bodyBytes.byteLength > 0 ? bodyBytes : undefined,
        redirect: "follow",
      });

      return fetch(originReq);
    }

    // 2. Static Assets fallback (served directly from Cloudflare Static Assets binding)
    if (env.ASSETS) {
      return env.ASSETS.fetch(request);
    }

    return new Response("Not Found", { status: 404 });
  },
};
