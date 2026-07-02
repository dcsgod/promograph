import type { TpoEvent } from "./events";

const BASE = "/api";

export interface Health {
  status: string;
  llm_available: boolean;
  mode: string;
  genie_backend: string;
  graph_backend: string;
  data_ready: boolean;
  model_ready: boolean;
  llm_model: string;
}

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}

export async function getExamples(): Promise<{ prompts: string[]; products: string[] }> {
  const r = await fetch(`${BASE}/examples`);
  return r.json();
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

/** POST a JSON body to an SSE endpoint and invoke onEvent for each parsed event. */
async function streamSSE(
  path: string,
  body: unknown,
  onEvent: (ev: TpoEvent) => void,
): Promise<void> {
  const resp = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.body) throw new Error("no response body");
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line (\n\n or \r\n\r\n)
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const dataLine = frame
        .split(/\r?\n/)
        .find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      const json = dataLine.slice(5).trim();
      if (!json) continue;
      try {
        onEvent(JSON.parse(json) as TpoEvent);
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}

export function sendChat(sessionId: string, message: string, onEvent: (ev: TpoEvent) => void) {
  return streamSSE("/chat", { session_id: sessionId, message }, onEvent);
}

export function sendNodeClick(
  sessionId: string,
  nodeId: string,
  nodeType: string,
  label: string,
  onEvent: (ev: TpoEvent) => void,
) {
  return streamSSE(
    "/node-click",
    { session_id: sessionId, node_id: nodeId, node_type: nodeType, label },
    onEvent,
  );
}
