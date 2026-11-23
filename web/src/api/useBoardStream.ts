import { useEffect, useRef } from "react";

import { boardStreamUrl, listRuns } from "./client";
import type { RunListResponse } from "./types";

export interface BoardStreamHandlers {
  apply: (snapshot: RunListResponse) => void;
}

export interface BoardStreamOptions {
  /** `false` (or `?nosse=1`) disables live SSE. */
  sse: boolean;
  reconnectDelayMs?: number;
}

export function openBoardStream(
  handlers: BoardStreamHandlers,
  options: BoardStreamOptions,
): () => void {
  let disposed = false;
  let source: EventSource | null = null;
  let timer: number | null = null;
  const delay = options.reconnectDelayMs ?? 1500;

  const scheduleReconnect = (): void => {
    source?.close();
    source = null;
    if (timer !== null || disposed) return;
    timer = window.setTimeout(() => {
      timer = null;
      if (!disposed) void syncAndConnect();
    }, delay);
  };

  const connect = (): void => {
    if (disposed || typeof EventSource === "undefined" || !options.sse) return;
    const stream = new EventSource(boardStreamUrl());
    source = stream;
    const isCurrent = (): boolean => !disposed && source === stream;

    stream.addEventListener("board", (event: MessageEvent<string>) => {
      if (!isCurrent()) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        return;
      }
      if (!parsed || typeof parsed !== "object") return;
      const frame = parsed as Partial<RunListResponse>;
      if (!frame.ok || !Array.isArray(frame.runs) || !Array.isArray(frame.pantheon)) return;
      handlers.apply(frame as RunListResponse);
    });

    stream.addEventListener("error", () => {
      if (isCurrent()) scheduleReconnect();
    });
  };

  async function syncAndConnect(): Promise<void> {
    try {
      const snapshot = await listRuns();
      if (disposed) return;
      if (snapshot.ok) handlers.apply(snapshot);
    } catch {
      if (!disposed) scheduleReconnect();
      return;
    }
    connect();
  }

  const onOffline = (): void => {
    if (source) scheduleReconnect();
  };

  window.addEventListener("offline", onOffline);
  if (options.sse) {
    queueMicrotask(() => {
      if (!disposed) connect();
    });
  }

  return () => {
    disposed = true;
    window.removeEventListener("offline", onOffline);
    source?.close();
    source = null;
    if (timer !== null) window.clearTimeout(timer);
  };
}

export function useBoardStream(
  handlers: BoardStreamHandlers,
  options: BoardStreamOptions,
): void {
  const latest = useRef(handlers);
  latest.current = handlers;
  const { sse, reconnectDelayMs } = options;

  useEffect(() => {
    if (!sse) return undefined;
    return openBoardStream(
      {
        apply: (snapshot) => latest.current.apply(snapshot),
      },
      { sse, reconnectDelayMs },
    );
  }, [sse, reconnectDelayMs]);
}
