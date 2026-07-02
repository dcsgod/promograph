import { useCallback, useEffect, useRef, useState } from "react";
import ChatPanel, { type ChatMessage } from "./components/chat/ChatPanel";
import GraphPanel from "./components/graph/GraphPanel";
import * as api from "./lib/api";
import type { Health } from "./lib/api";
import type { Recommendation, TpoEvent } from "./lib/events";
import type { AppEdge, AppNode } from "./lib/graph";

const SESSION_ID = Math.random().toString(36).slice(2);

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [examples, setExamples] = useState<string[]>([]);

  // graph state kept in refs (mutated during streaming) mirrored to state for render
  const nodesRef = useRef<Map<string, AppNode>>(new Map());
  const edgesRef = useRef<AppEdge[]>([]);
  const [nodes, setNodes] = useState<Map<string, AppNode>>(new Map());
  const [edges, setEdges] = useState<AppEdge[]>([]);

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => {});
    api.getExamples().then((e) => setExamples(e.prompts)).catch(() => {});
  }, []);

  const flush = () => {
    setNodes(new Map(nodesRef.current));
    setEdges([...edgesRef.current]);
  };

  const handleEvent = useCallback((ev: TpoEvent) => {
    switch (ev.type) {
      case "token":
        setMessages((ms) => {
          const last = ms[ms.length - 1];
          if (!last || last.role !== "assistant") return ms;
          return [...ms.slice(0, -1), { ...last, content: last.content + ev.text }];
        });
        break;
      case "tool":
        if (ev.status === "start") {
          setMessages((ms) => {
            const last = ms[ms.length - 1];
            if (!last || last.role !== "assistant") return ms;
            return [...ms.slice(0, -1), { ...last, tools: [...(last.tools ?? []), ev.name] }];
          });
        }
        break;
      case "graph_node":
        nodesRef.current.set(ev.id, {
          id: ev.id, node_type: ev.node_type, label: ev.label, metrics: ev.metrics || {},
        });
        flush();
        break;
      case "graph_edge":
        edgesRef.current.push({
          source: ev.source, target: ev.target, edge_type: ev.edge_type, weight: ev.weight,
        });
        flush();
        break;
      case "node_update": {
        const n = nodesRef.current.get(ev.id);
        if (n) n.metrics = { ...n.metrics, ...ev.metrics };
        flush();
        break;
      }
      case "recommendation":
        setRecommendation(ev.data);
        break;
      case "error":
        setMessages((ms) => {
          const last = ms[ms.length - 1];
          if (!last || last.role !== "assistant") return ms;
          return [...ms.slice(0, -1), { ...last, notice: ev.message }];
        });
        break;
      case "done":
        setStreaming(false);
        break;
    }
  }, []);

  const startTurn = async (
    runner: (onEvent: (ev: TpoEvent) => void) => Promise<void>,
    userText: string | null,
  ) => {
    if (streaming) return;
    setStreaming(true);
    setRecommendation(null);
    setMessages((ms) => [
      ...ms,
      ...(userText ? [{ role: "user", content: userText } as ChatMessage] : []),
      { role: "assistant", content: "", tools: [] } as ChatMessage,
    ]);
    try {
      await runner(handleEvent);
    } catch (e) {
      handleEvent({ type: "error", message: String(e) });
      setStreaming(false);
    }
  };

  const onSend = (text: string) =>
    startTurn((cb) => api.sendChat(SESSION_ID, text, cb), text);

  const onNodeClick = (n: AppNode) =>
    startTurn(
      (cb) => api.sendNodeClick(SESSION_ID, n.id, n.node_type, n.label, cb),
      `Drill into ${n.label}`,
    );

  return (
    <div className="app">
      <header>
        <div className="brand">
          <strong>Promograph</strong> · Graph-Steered Promotion Optimization
        </div>
        {health && (
          <div className="status">
            <Badge on={health.data_ready} label="data" />
            <Badge on={health.model_ready} label="model" />
            <span className={`mode ${health.mode}`}>
              {health.mode === "llm" ? `LLM: ${health.llm_model}` : "deterministic (no Ollama)"}
            </span>
            <span className="be">genie:{health.genie_backend} · graph:{health.graph_backend}</span>
          </div>
        )}
      </header>
      <div className="panes">
        <div className="left">
          <ChatPanel
            messages={messages}
            streaming={streaming}
            recommendation={recommendation}
            examples={examples}
            onSend={onSend}
          />
        </div>
        <div className="right">
          {nodes.size === 0 ? (
            <div className="graph-empty">The product subgraph will build here as you ask.</div>
          ) : (
            <GraphPanel nodes={nodes} edges={edges} onNodeClick={onNodeClick} />
          )}
          <Legend />
        </div>
      </div>
    </div>
  );
}

function Badge({ on, label }: { on: boolean; label: string }) {
  return <span className={`badge ${on ? "ok" : "bad"}`}>{label}</span>;
}

function Legend() {
  const items = [
    ["#2563eb", "focus"], ["#f97316", "substitute"], ["#dc2626", "cannibalized"],
    ["#0d9488", "co-purchase"], ["#16a34a", "halo gain"], ["#7c3aed", "promo"], ["#6b7280", "category"],
  ] as const;
  return (
    <div className="legend">
      {items.map(([c, l]) => (
        <span key={l}><i style={{ background: c }} />{l}</span>
      ))}
    </div>
  );
}
