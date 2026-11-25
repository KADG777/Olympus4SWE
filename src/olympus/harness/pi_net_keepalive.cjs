"use strict";

// Preloaded into Pi's Node.js subprocess via NODE_OPTIONS=--require so long
// reasoning/streaming requests keep TCP connections alive across 60s NAT/gateway
// idle windows and never hit short body/headers read timeouts.

(() => {
  const MIN_READ_TIMEOUT_MS = 300_000;
  const DEFAULT_READ_TIMEOUT_MS = 600_000;
  const DEFAULT_KEEPALIVE_MS = 20_000;

  function parsePositiveInt(raw, fallback) {
    const parsed = Number.parseInt(String(raw ?? ""), 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  const keepAliveMs = parsePositiveInt(
    process.env.OLYMPUS_PI_TCP_KEEPALIVE_MS,
    DEFAULT_KEEPALIVE_MS
  );
  const readTimeoutMs = Math.max(
    MIN_READ_TIMEOUT_MS,
    parsePositiveInt(process.env.OLYMPUS_PI_READ_TIMEOUT_MS, DEFAULT_READ_TIMEOUT_MS)
  );

  // 1. Enable TCP keep-alive on every outbound net.Socket connection.
  try {
    const net = require("net");
    if (net && net.Socket && typeof net.Socket.prototype.connect === "function") {
      const origConnect = net.Socket.prototype.connect;
      if (!origConnect.__olympusKeepAlivePatched) {
        const patched = function olympusKeepAliveConnect(...args) {
          const result = origConnect.apply(this, args);
          try {
            if (typeof this.setKeepAlive === "function") {
              this.setKeepAlive(true, keepAliveMs);
            }
          } catch (_) {
            // Best-effort only; never disrupt socket connection setup.
          }
          return result;
        };
        patched.__olympusKeepAlivePatched = true;
        net.Socket.prototype.connect = patched;
      }
    }
  } catch (_) {
    // Ignore on non-Node or restricted runtimes.
  }

  // 2. Ensure Node's undici global fetch dispatcher enforces >= 300s read
  // timeouts and TCP keep-alive while preserving EnvHttpProxyAgent proxy rules.
  try {
    const sym = Symbol.for("undici.globalDispatcher.1");
    const current = globalThis[sym];
    if (current && typeof current.constructor === "function") {
      const DispatcherCtor = current.constructor;
      const next = new DispatcherCtor({
        headersTimeout: readTimeoutMs,
        bodyTimeout: readTimeoutMs,
        connect: {
          keepAlive: true,
          keepAliveInitialDelay: keepAliveMs,
        },
      });
      globalThis[sym] = next;
    }
  } catch (_) {
    // Keep default dispatcher if runtime differs.
  }
  // 3. Strip this preload from process.env.NODE_OPTIONS after loading in Pi so
  // child Node.js processes spawned via bash (test runners, dev servers under test)
  // do not inherit long keep-alive / 600s undici dispatcher overrides.
  try {
    const rawOpts = process.env.NODE_OPTIONS;
    if (typeof rawOpts === "string" && rawOpts.trim()) {
      const fs = require("fs");
      const path = require("path");
      const selfResolved = path.resolve(__filename);
      let selfReal = selfResolved;
      let selfStat = null;
      try {
        selfReal = fs.realpathSync(selfResolved);
      } catch (_) {}
      try {
        selfStat = fs.statSync(selfReal);
      } catch (_) {}

      const isSelfTarget = (candidate) => {
        if (!candidate) return false;
        const resolved = path.resolve(candidate);
        if (resolved === selfResolved || resolved === selfReal) return true;
        try {
          if (fs.realpathSync(candidate) === selfReal) return true;
        } catch (_) {}
        if (selfStat) {
          try {
            const st = fs.statSync(candidate);
            if (st.dev === selfStat.dev && st.ino === selfStat.ino) return true;
          } catch (_) {}
        }
        return false;
      };

      // Tokenize NODE_OPTIONS matching Node's ParseNodeOptionsEnvVar: only double
      // quotes delimit quoted spans (single quotes are literal path characters).
      const tokens = [];
      let i = 0;
      const len = rawOpts.length;
      while (i < len) {
        while (i < len && /\s/.test(rawOpts[i])) i += 1;
        if (i >= len) break;
        const start = i;
        let decoded = "";
        let inDoubleQuote = false;
        while (i < len) {
          const ch = rawOpts[i];
          if (inDoubleQuote) {
            if (ch === "\\" && i + 1 < len) {
              decoded += rawOpts[i + 1];
              i += 2;
              continue;
            }
            if (ch === '"') {
              inDoubleQuote = false;
              i += 1;
              continue;
            }
            decoded += ch;
            i += 1;
          } else {
            if (/\s/.test(ch)) break;
            if (ch === '"') {
              inDoubleQuote = true;
              i += 1;
              continue;
            }
            if (ch === "\\" && i + 1 < len) {
              decoded += rawOpts[i + 1];
              i += 2;
              continue;
            }
            decoded += ch;
            i += 1;
          }
        }
        tokens.push({ raw: rawOpts.slice(start, i), decoded });
      }

      const kept = [];
      for (let idx = 0; idx < tokens.length; idx += 1) {
        const { raw, decoded } = tokens[idx];
        if (decoded === "--require" || decoded === "-r") {
          const next = tokens[idx + 1];
          if (next && isSelfTarget(next.decoded)) {
            idx += 1;
            continue;
          }
        } else if (decoded.startsWith("--require=")) {
          if (isSelfTarget(decoded.slice("--require=".length))) {
            continue;
          }
        } else if (decoded.startsWith("-r=")) {
          if (isSelfTarget(decoded.slice("-r=".length))) {
            continue;
          }
        }
        kept.push(raw);
      }
      if (kept.length > 0) {
        process.env.NODE_OPTIONS = kept.join(" ");
      } else {
        delete process.env.NODE_OPTIONS;
      }
    }
  } catch (_) {
    // Never disrupt Pi startup if NODE_OPTIONS parsing fails.
  }
})();
