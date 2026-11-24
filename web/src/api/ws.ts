// Thin WebSocket client for /ws with auto-reconnect.
import type { ServerMsg } from "./types";

export class WSClient {
  private ws: WebSocket | null = null;
  private closed = false;

  constructor(
    private url: string,
    private onMsg: (m: ServerMsg) => void,
    private onOpen: () => void,
    private onClose: () => void,
  ) {}

  connect(): void {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => this.onOpen();
    this.ws.onmessage = (e) => {
      try {
        this.onMsg(JSON.parse(e.data) as ServerMsg);
      } catch {
        /* ignore malformed frames */
      }
    };
    this.ws.onclose = () => {
      this.onClose();
      if (!this.closed) setTimeout(() => this.connect(), 1000);
    };
  }

  send(msg: object): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
  }
}

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}
