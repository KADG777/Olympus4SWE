import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const OUT_DIR = join(ROOT, "..", "src", "olympus", "web", "static", "graph");

/** Files whose bytes define the bundle. tests/unit/test_web_ariadne.py mirrors this list. */
const STAMP_FILES = ["package.json", "pnpm-lock.yaml", "tsconfig.json", "vite.config.ts"];
const STAMP_DIRS = ["src"];

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

/** sha256 over sorted `relpath \0 bytes \0` records, so Python can verify freshness offline. */
export function sourceStamp(): string {
  const files = [
    ...STAMP_FILES.map((name) => join(ROOT, name)),
    ...STAMP_DIRS.flatMap((dir) => walk(join(ROOT, dir))),
  ]
    .map((path) => ({ path, rel: relative(ROOT, path).split(sep).join("/") }))
    .sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
  const hash = createHash("sha256");
  for (const { path, rel } of files) {
    hash.update(rel, "utf8");
    hash.update("\0");
    hash.update(readFileSync(path));
    hash.update("\0");
  }
  return hash.digest("hex");
}

function stampBanner(): Plugin {
  return {
    name: "ariadne-source-stamp",
    apply: "build",
    enforce: "post",
    generateBundle(_options, bundle) {
      const banner = `/*! ariadne-graph | source-sha256:${sourceStamp()} */\n`;
      for (const output of Object.values(bundle)) {
        if (output.type === "chunk") {
          output.code = banner + output.code;
        } else if (output.fileName.endsWith(".css")) {
          output.source = banner + String(output.source);
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), stampBanner()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    outDir: OUT_DIR,
    emptyOutDir: true,
    target: "es2020",
    cssCodeSplit: false,
    sourcemap: false,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 4096,
    lib: {
      entry: join(ROOT, "src", "main.tsx"),
      name: "AriadneGraphBundle",
      formats: ["iife"],
      fileName: () => "ariadne-graph.js",
      cssFileName: "ariadne-graph",
    },
  },
});
