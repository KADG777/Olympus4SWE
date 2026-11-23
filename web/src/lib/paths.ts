export interface BreadcrumbSegment {
  label: string;
  path: string;
}

const SCP_PATTERN = /^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:([^:\s]+)$/;
const SHORTHAND_PATTERN = /^[A-Za-z0-9_.-]+\/([A-Za-z0-9_.-]+)$/;

function stripGitSuffix(segment: string): string {
  return segment.endsWith(".git") ? segment.slice(0, -4) : segment;
}

export function buildBreadcrumbs(currentPath: string): BreadcrumbSegment[] {
  const raw = currentPath.trim();
  if (!raw) return [];
  if (raw.startsWith("/")) {
    const parts = raw.slice(1).split("/").filter(Boolean);
    const segments: BreadcrumbSegment[] = [{ label: "/", path: "/" }];
    let acc = "";
    for (const part of parts) {
      acc = `${acc}/${part}`;
      segments.push({ label: part, path: acc });
    }
    return segments;
  }
  const driveMatch = /^([A-Za-z]:)[\\/]*(.*)$/.exec(raw);
  if (driveMatch) {
    const drive = `${driveMatch[1]}\\`;
    const parts = driveMatch[2].split(/[\\/]+/).filter(Boolean);
    const segments: BreadcrumbSegment[] = [{ label: driveMatch[1], path: drive }];
    let acc = driveMatch[1];
    for (const part of parts) {
      acc = `${acc}\\${part}`;
      segments.push({ label: part, path: acc });
    }
    return segments;
  }
  return [{ label: raw, path: raw }];
}

export function joinDirPath(parentDir: string, folderName: string): string {
  const parent = parentDir.trim();
  const folder = folderName.trim();
  if (!parent || !folder) return "";
  if (parent === "/") return `/${folder}`;
  const normalized = parent.replace(/[\\/]+$/, "");
  const sep = normalized.includes("\\") && !normalized.includes("/") ? "\\" : "/";
  return `${normalized}${sep}${folder}`;
}

export function deriveCloneFolderName(rawUrl: string): string {
  const value = rawUrl.trim();
  if (!value || value.startsWith("-") || /\s/.test(value)) return "";
  if (value.startsWith("/") || value.startsWith("~") || value.startsWith(".")) return "";

  if (value.includes("://")) {
    try {
      const parsed = new URL(value);
      const scheme = parsed.protocol.replace(/:$/, "").toLowerCase();
      if (!["https", "http", "ssh", "git"].includes(scheme)) return "";
      const segments = parsed.pathname.split("/").filter(Boolean);
      if (segments.length < 2) return "";
      if (segments.some((s) => s === "." || s === "..")) return "";
      return stripGitSuffix(segments[segments.length - 1]);
    } catch {
      return "";
    }
  }

  const scp = SCP_PATTERN.exec(value);
  if (scp) {
    const segments = scp[1].split("/").filter(Boolean);
    if (segments.length < 2) return "";
    if (segments.some((s) => s === "." || s === "..")) return "";
    return stripGitSuffix(segments[segments.length - 1]);
  }

  const shorthand = SHORTHAND_PATTERN.exec(value);
  if (shorthand) {
    return stripGitSuffix(shorthand[1]);
  }

  return "";
}
