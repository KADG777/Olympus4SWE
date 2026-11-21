/**
 * Olympus4SWE Pi scope guard.
 *
 * Fast in-loop feedback for `write` / `edit` calls whose target escapes the
 * task's authorized write scope or hits a forbidden shared-contract glob.
 * The deterministic post-flight Diff Gate in Olympus remains authoritative:
 * this extension only helps the model self-correct early.
 */
import * as path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

function readList(name: string): string[] {
  try {
    const parsed = JSON.parse(process.env[name] ?? "[]");
    return Array.isArray(parsed) ? parsed.map(String).filter((item) => item.trim() !== "") : [];
  } catch {
    return [];
  }
}

const writeScope = readList("OLYMPUS_PI_WRITE_SCOPE");
const forbiddenGlobs = readList("OLYMPUS_PI_FORBIDDEN_GLOBS");
const workspaceRoot = process.cwd();

function globToRegExp(pattern: string): RegExp {
  let out = "^";
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index];
    if (char === "*") {
      if (pattern[index + 1] === "*") {
        index += 1;
        if (pattern[index + 1] === "/") {
          index += 1;
          out += "(?:.*/)?";
        } else {
          out += ".*";
        }
      } else {
        out += "[^/]*";
      }
    } else if (char === "?") {
      out += "[^/]";
    } else if (char === "/") {
      out += "/";
    } else {
      out += char.replace(/[.+^${}()|[\]\\]/g, "\\$&");
    }
  }
  return new RegExp(`${out}$`);
}

function relativeTarget(raw: unknown): string | null {
  if (typeof raw !== "string" || raw.trim() === "") {
    return null;
  }
  const absolute = path.resolve(workspaceRoot, raw);
  const relative = path.relative(workspaceRoot, absolute);
  if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
    return null;
  }
  return relative.split(path.sep).join("/");
}

function normalizePattern(raw: string): string {
  let pattern = raw.replace(/\\/g, "/");
  while (pattern.startsWith("./")) {
    pattern = pattern.slice(2);
  }
  return pattern.replace(/\/+$/, "");
}

function hasGlobChars(pattern: string): boolean {
  return pattern.includes("*") || pattern.includes("?") || pattern.includes("[");
}

function contains(relative: string, rawPattern: string): boolean {
  const pattern = normalizePattern(rawPattern);
  if (pattern === "" || pattern === ".") {
    return true;
  }
  if (hasGlobChars(pattern)) {
    return globToRegExp(pattern).test(relative);
  }
  return relative === pattern || relative.startsWith(`${pattern}/`);
}

function matches(relative: string, rawPattern: string): boolean {
  const pattern = normalizePattern(rawPattern);
  if (pattern === "" || pattern === ".") {
    return true;
  }
  if (hasGlobChars(pattern)) {
    return globToRegExp(pattern).test(relative);
  }
  return (
    relative === pattern ||
    relative.startsWith(`${pattern}/`) ||
    pattern.startsWith(`${relative}/`)
  );
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event) => {
    if (event.toolName !== "write" && event.toolName !== "edit") {
      return;
    }
    const input = event.input as { path?: string; file_path?: string } | undefined;
    const relative = relativeTarget(input?.path ?? input?.file_path);
    if (relative === null) {
      return {
        block: true,
        reason: "Olympus scope guard: target path escapes the sandbox workspace.",
      };
    }
    const forbidden = forbiddenGlobs.find(
      (pattern) => contains(relative, pattern) || matches(relative, pattern),
    );
    if (forbidden !== undefined) {
      return {
        block: true,
        reason: `Olympus scope guard: '${relative}' matches forbidden shared-contract glob '${forbidden}'.`,
      };
    }
    if (!writeScope.some((pattern) => contains(relative, pattern))) {
      return {
        block: true,
        reason: `Olympus scope guard: '${relative}' is outside write_scope ${JSON.stringify(writeScope)}.`,
      };
    }
  });
}