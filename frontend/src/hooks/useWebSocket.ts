import { useEffect, useRef, useState } from "react";
import type { WSStatusMessage } from "@/types";

const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

export function useWebSocket(runId: string | null) {
  const [lastMessage, setLastMessage] = useState<WSStatusMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const pingInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!runId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/${runId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // Send ping every 25s to keep connection alive
      pingInterval.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send("ping");
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSStatusMessage;
        if (msg.type !== "keepalive") {
          setLastMessage(msg);
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (pingInterval.current) clearInterval(pingInterval.current);
    };

    ws.onerror = () => {
      setConnected(false);
    };

    return () => {
      if (pingInterval.current) clearInterval(pingInterval.current);
      ws.close();
    };
  }, [runId]);

  return { lastMessage, connected };
}
