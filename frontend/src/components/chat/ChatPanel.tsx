import { useEffect, useRef } from "react";
import type { Recommendation } from "../../lib/events";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  tools?: string[];
  notice?: string;
}

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  recommendation: Recommendation | null;
  examples: string[];
  onSend: (text: string) => void;
}

export default function ChatPanel({ messages, streaming, recommendation, examples, onSend }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, recommendation]);

  const submit = () => {
    const v = inputRef.current?.value.trim();
    if (v && !streaming) {
      onSend(v);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask a promotion question. Click any graph node to steer the analysis.</p>
            <div className="examples">
              {examples.filter(Boolean).map((ex, i) => (
                <button key={i} className="chip" onClick={() => onSend(ex)}>{ex}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.tools && m.tools.length > 0 && (
              <div className="tools">
                {m.tools.map((t, j) => <span key={j} className="tool">{t}</span>)}
              </div>
            )}
            {m.notice && <div className="notice">{m.notice}</div>}
            <div className="bubble">{m.content || (streaming && i === messages.length - 1 ? "…" : "")}</div>
          </div>
        ))}
        {recommendation && <RecommendationCard rec={recommendation} />}
        <div ref={endRef} />
      </div>
      <div className="composer">
        <input
          ref={inputRef}
          placeholder="e.g. What's the optimal discount for ground coffee?"
          onKeyDown={(e) => e.key === "Enter" && submit()}
          disabled={streaming}
        />
        <button onClick={submit} disabled={streaming}>{streaming ? "…" : "Send"}</button>
      </div>
    </div>
  );
}

function RecommendationCard({ rec }: { rec: Recommendation }) {
  const positive = rec.net_incremental_margin > 0;
  return (
    <div className="reco">
      <div className="reco-head">
        <span className="reco-title">Recommendation · {rec.label}</span>
        <span className={`pill ${rec.recommended_depth > 0 ? "go" : "hold"}`}>
          {rec.recommended_depth > 0 ? `${rec.recommended_depth_pct}% off` : "Do not promote"}
        </span>
      </div>
      <div className="reco-grid">
        <Metric label="Net margin" value={`$${rec.net_incremental_margin.toFixed(0)}`} good={positive} />
        <Metric label="Clearance value" value={`$${rec.clearance_value.toFixed(0)}`} />
        <Metric label="Net ROI" value={rec.net_roi.toFixed(2)} good={rec.net_roi > 0} />
        <Metric label="Blended margin" value={`${rec.blended_margin_pct.toFixed(0)}%`} good={rec.margin_ok} />
        <Metric label="Weeks of supply" value={rec.weeks_of_supply.toFixed(0)} />
        <Metric label="Own lift" value={`${rec.impact.own_lift_units.toFixed(0)} u`} />
      </div>
      <div className="reco-net">
        <span>Cannibalization ${rec.impact.cannibalization_margin.toFixed(0)}</span>
        <span>Halo +${rec.impact.halo_margin.toFixed(0)}</span>
      </div>
      <p className="reco-why">{rec.rationale}</p>
    </div>
  );
}

function Metric({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${good === undefined ? "" : good ? "pos" : "neg"}`}>{value}</div>
    </div>
  );
}
