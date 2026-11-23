import { useEffect, useState, type CSSProperties } from "react";

import type { PantheonGod, SeatState } from "../api/types";
import { GOD_META } from "../lib/labels";

export interface GodAvatarAsset {
  idleSrc?: string;
  workingSrc?: string;
  frames?: number;
  fps?: number;
  label: string;
}

export type PantheonAvatarManifest = Record<PantheonGod, GodAvatarAsset>;

export type GodAvatarManifestOverride = Partial<
  Record<PantheonGod, Partial<GodAvatarAsset>>
>;

declare global {
  interface Window {
    __OLYMPUS_AVATAR_MANIFEST__?: GodAvatarManifestOverride;
  }
}

export const AVATAR_MANIFEST_CHANGE_EVENT = "olympus:avatar-manifest-change";

export const PANTHEON_AVATAR_MANIFEST: PantheonAvatarManifest = {
  prometheus: {
    label: "Prometheus — Align & Needs",
  },
  athena: {
    label: "Athena — Design & Architecture",
  },
  hephaestus: {
    label: "Hephaestus — Build & Sandbox",
  },
  momus: {
    label: "Momus — Review & Audit",
  },
  argus: {
    label: "Argus — Verify & Gate",
  },
};

export interface GodAvatarProps {
  god: PantheonGod;
  state: SeatState;
  size?: "sm" | "lg";
  manifestOverride?: GodAvatarManifestOverride;
  className?: string;
}

function readWindowManifest(): GodAvatarManifestOverride | undefined {
  if (typeof window === "undefined") return undefined;
  return window.__OLYMPUS_AVATAR_MANIFEST__;
}

export function resolveGodAvatarAsset(
  god: PantheonGod,
  manifestOverride?: GodAvatarManifestOverride,
  windowManifest?: GodAvatarManifestOverride,
): GodAvatarAsset {
  const base = PANTHEON_AVATAR_MANIFEST[god];
  const fromWindow = windowManifest?.[god];
  const fromProp = manifestOverride?.[god];
  return {
    ...base,
    ...fromWindow,
    ...fromProp,
    label: fromProp?.label ?? fromWindow?.label ?? base.label,
  };
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return false;
    }
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  return reduced;
}

function useWindowManifestOverride(): GodAvatarManifestOverride | undefined {
  const [override, setOverride] = useState<GodAvatarManifestOverride | undefined>(() =>
    readWindowManifest(),
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const sync = () => setOverride(readWindowManifest());
    window.addEventListener(AVATAR_MANIFEST_CHANGE_EVENT, sync);
    return () => window.removeEventListener(AVATAR_MANIFEST_CHANGE_EVENT, sync);
  }, []);

  return override;
}

function GodEmblemSvg({ god }: { god: PantheonGod }) {
  switch (god) {
    case "prometheus":
      return (
        <svg
          className="god-avatar-svg"
          viewBox="0 0 64 64"
          fill="none"
          aria-hidden="true"
        >
          <circle
            cx="32"
            cy="32"
            r="27"
            stroke="var(--rule)"
            strokeWidth="1.25"
            strokeDasharray="2 3"
          />
          <path
            d="M32 10C37.5 16.5 41 21.5 41 27.5C41 32.8 36.9 37 32 37C27.1 37 23 32.8 23 27.5C23 23.8 25.1 20.1 27.6 17.5C27.5 21 29.2 22.8 31 23.6C30.2 19 30.8 14.5 32 10Z"
            fill="var(--accent-soft)"
            stroke="currentColor"
            strokeWidth="2.1"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M32 23.5C34.4 26.2 35.6 28.5 35.6 31C35.6 33.1 34 34.8 32 34.8C30 34.8 28.4 33.1 28.4 31C28.4 28.7 29.7 26.4 32 23.5Z"
            fill="var(--surface-paper)"
            stroke="var(--accent)"
            strokeWidth="1.6"
          />
          <path
            d="M21.5 37.5H42.5L39.2 42.5H24.8L21.5 37.5Z"
            fill="var(--surface-sunk)"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path
            d="M27.5 42.5L29.5 54H34.5L36.5 42.5"
            fill="var(--surface-paper)"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path
            d="M32 42.5V54"
            stroke="var(--rule-hard)"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      );

    case "athena":
      return (
        <svg
          className="god-avatar-svg"
          viewBox="0 0 64 64"
          fill="none"
          aria-hidden="true"
        >
          <circle
            cx="32"
            cy="32"
            r="27"
            stroke="var(--rule)"
            strokeWidth="1.25"
            strokeDasharray="2 3"
          />
          <path
            d="M20 18L26.5 22H37.5L44 18L42.5 28.5C44.5 31.8 45 36.2 43.2 41.2C41.4 46 37.2 49.5 32 51.5C26.8 49.5 22.6 46 20.8 41.2C19 36.2 19.5 31.8 21.5 28.5L20 18Z"
            fill="var(--accent-soft)"
            stroke="currentColor"
            strokeWidth="2.1"
            strokeLinejoin="round"
          />
          <circle
            cx="26.8"
            cy="30"
            r="4.2"
            fill="var(--surface-paper)"
            stroke="currentColor"
            strokeWidth="1.9"
          />
          <circle
            cx="37.2"
            cy="30"
            r="4.2"
            fill="var(--surface-paper)"
            stroke="currentColor"
            strokeWidth="1.9"
          />
          <circle cx="26.8" cy="30" r="1.4" fill="currentColor" />
          <circle cx="37.2" cy="30" r="1.4" fill="currentColor" />
          <path
            d="M32 32.5L29.6 36.4L32 38.8L34.4 36.4L32 32.5Z"
            fill="var(--surface-paper)"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinejoin="round"
          />
          <path
            d="M25.5 42.5H38.5M27.5 46H36.5"
            stroke="var(--rule-hard)"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
      );

    case "hephaestus":
      return (
        <svg
          className="god-avatar-svg"
          viewBox="0 0 64 64"
          fill="none"
          aria-hidden="true"
        >
          <circle
            cx="32"
            cy="32"
            r="27"
            stroke="var(--rule)"
            strokeWidth="1.25"
            strokeDasharray="2 3"
          />
          <path
            d="M15 33.5H47.5V39C47.5 41.2 45.7 43 43.5 43H39L41.5 49.5H23.5L26 43H22.5C18.2 43 15.8 39.2 15 33.5Z"
            fill="var(--live-soft)"
            stroke="currentColor"
            strokeWidth="2.1"
            strokeLinejoin="round"
          />
          <path
            d="M19.5 49.5H45.5"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
          />
          <path
            d="M24 23.5L36.5 16L40.5 22.5L28 30L24 23.5Z"
            fill="var(--surface-paper)"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path
            d="M35.5 25.5L46.5 31.8"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
          />
          <path
            d="M19 16.5L21 19.5M45 14L43.5 17.5M49.5 21.5L46 22.5"
            stroke="var(--accent)"
            strokeWidth="1.7"
            strokeLinecap="round"
          />
        </svg>
      );

    case "momus":
      return (
        <svg
          className="god-avatar-svg"
          viewBox="0 0 64 64"
          fill="none"
          aria-hidden="true"
        >
          <circle
            cx="32"
            cy="32"
            r="27"
            stroke="var(--rule)"
            strokeWidth="1.25"
            strokeDasharray="2 3"
          />
          <path
            d="M16.5 21.5C21.2 18.8 27.8 18.8 32.5 21.5V34.5C32.5 41.2 28.4 46.2 24.5 48C20.6 46.2 16.5 41.2 16.5 34.5V21.5Z"
            fill="var(--surface-sunk)"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path
            d="M20.5 29H23.2M26 29H28.7M21.2 37.5C23.2 35.8 26 35.8 28 37.5"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
          />
          <circle
            cx="39.5"
            cy="31"
            r="9.5"
            fill="var(--live-soft)"
            stroke="currentColor"
            strokeWidth="2.1"
          />
          <circle
            cx="39.5"
            cy="31"
            r="4.5"
            fill="var(--surface-paper)"
            stroke="var(--live)"
            strokeWidth="1.5"
          />
          <path
            d="M46.2 37.8L51.5 43.2"
            stroke="currentColor"
            strokeWidth="2.6"
            strokeLinecap="round"
          />
        </svg>
      );

    case "argus":
      return (
        <svg
          className="god-avatar-svg"
          viewBox="0 0 64 64"
          fill="none"
          aria-hidden="true"
        >
          <circle
            cx="32"
            cy="32"
            r="27"
            stroke="var(--rule)"
            strokeWidth="1.25"
            strokeDasharray="2 3"
          />
          <path
            d="M15.5 24.5L32 14.5L48.5 24.5"
            stroke="var(--rule-hard)"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M14 34.5C18.8 26.8 25 23.5 32 23.5C39 23.5 45.2 26.8 50 34.5C45.2 42.2 39 45.5 32 45.5C25 45.5 18.8 42.2 14 34.5Z"
            fill="var(--ok-soft)"
            stroke="currentColor"
            strokeWidth="2.1"
            strokeLinejoin="round"
          />
          <circle
            cx="32"
            cy="34.5"
            r="6.2"
            fill="var(--surface-paper)"
            stroke="currentColor"
            strokeWidth="2"
          />
          <circle cx="32" cy="34.5" r="2.4" fill="currentColor" />
          <path
            d="M21.5 49.5H42.5"
            stroke="var(--rule-hard)"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
        </svg>
      );
  }
}

export function GodAvatar({
  god,
  state,
  size = "lg",
  manifestOverride,
  className,
}: GodAvatarProps) {
  const reducedMotion = useReducedMotion();
  const windowManifest = useWindowManifestOverride();
  const [failedSrcs, setFailedSrcs] = useState<Record<string, true>>({});

  const asset = resolveGodAvatarAsset(god, manifestOverride, windowManifest);
  const meta = GOD_META[god];

  const markFailed = (src: string) => {
    setFailedSrcs((prev) => (prev[src] ? prev : { ...prev, [src]: true }));
  };

  const canPlayWorking =
    state === "working" &&
    !reducedMotion &&
    Boolean(asset.workingSrc) &&
    !failedSrcs[asset.workingSrc as string];

  const hasIdleImage =
    Boolean(asset.idleSrc) && !failedSrcs[asset.idleSrc as string];

  const isSprite =
    canPlayWorking &&
    typeof asset.frames === "number" &&
    Number.isFinite(asset.frames) &&
    asset.frames > 1;

  const spriteFrames = isSprite ? Math.max(2, Math.round(asset.frames!)) : 1;
  const spriteFps =
    typeof asset.fps === "number" && Number.isFinite(asset.fps) && asset.fps > 0
      ? asset.fps
      : 8;
  const spriteDuration = `${(spriteFrames / spriteFps).toFixed(3)}s`;

  let mode: "working-sprite" | "working-image" | "idle-image" | "fallback-svg";
  if (isSprite) {
    mode = "working-sprite";
  } else if (canPlayWorking) {
    mode = "working-image";
  } else if (hasIdleImage) {
    mode = "idle-image";
  } else {
    mode = "fallback-svg";
  }

  const assetFailed =
    (Boolean(asset.workingSrc) && Boolean(failedSrcs[asset.workingSrc as string])) ||
    (Boolean(asset.idleSrc) && Boolean(failedSrcs[asset.idleSrc as string]));

  const spriteStyle = {
    "--avatar-sprite-frames": String(spriteFrames),
    "--avatar-sprite-duration": spriteDuration,
  } as CSSProperties;

  return (
    <span
      className={className ? `god-avatar ${className}` : "god-avatar"}
      data-god={god}
      data-state={state}
      data-size={size}
      data-avatar-mode={mode}
      data-reduced-motion={reducedMotion ? "true" : "false"}
      data-asset-fallback={assetFailed ? "true" : undefined}
      title={asset.label || `${meta.name} · ${meta.role}`}
      aria-hidden="true"
    >
      <span className="god-avatar-frame">
        {mode === "working-sprite" && asset.workingSrc ? (
          <span className="god-avatar-sprite-strip" style={spriteStyle}>
            <img
              className="god-avatar-sprite-img"
              src={asset.workingSrc}
              alt=""
              draggable={false}
              onError={() => markFailed(asset.workingSrc!)}
            />
          </span>
        ) : mode === "working-image" && asset.workingSrc ? (
          <img
            className="god-avatar-img"
            src={asset.workingSrc}
            alt=""
            draggable={false}
            onError={() => markFailed(asset.workingSrc!)}
          />
        ) : mode === "idle-image" && asset.idleSrc ? (
          <img
            className="god-avatar-img"
            src={asset.idleSrc}
            alt=""
            draggable={false}
            onError={() => markFailed(asset.idleSrc!)}
          />
        ) : (
          <GodEmblemSvg god={god} />
        )}
      </span>
    </span>
  );
}
