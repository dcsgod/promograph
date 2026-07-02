import { useEffect, useMemo } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  useReactFlow,
  type Node,
  type Edge,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { type AppEdge, type AppNode, nodeColor, relationOf } from "../../lib/graph";

/** Refit the viewport whenever the node set changes (nodes stream in async). */
function Refit({ count }: { count: number }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const t = setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 60);
    return () => clearTimeout(t);
  }, [count, fitView]);
  return null;
}

interface Props {
  nodes: Map<string, AppNode>;
  edges: AppEdge[];
  onNodeClick: (n: AppNode) => void;
}

const GROUP_X: Record<string, number> = {
  focus: 0, category: 0, promo: 0, substitute: -360, copurchase: 360, other: 0,
};

function spread(count: number, gap = 90): number[] {
  const start = -((count - 1) * gap) / 2;
  return Array.from({ length: count }, (_, i) => start + i * gap);
}

export default function GraphPanel({ nodes, edges, onNodeClick }: Props) {
  const focusId = useMemo(() => {
    for (const n of nodes.values()) if (n.metrics?.focus) return n.id;
    for (const n of nodes.values()) if (n.node_type === "SKU") return n.id;
    return null;
  }, [nodes]);

  const { rfNodes, rfEdges } = useMemo(() => {
    const list = [...nodes.values()];
    const byRel: Record<string, AppNode[]> = {
      focus: [], substitute: [], copurchase: [], category: [], promo: [], other: [],
    };
    for (const n of list) byRel[relationOf(n, focusId, edges)].push(n);

    const positioned: Node[] = [];
    const place = (arr: AppNode[], x: number, yBase: number, vertical: boolean) => {
      const offs = spread(arr.length);
      arr.forEach((n, i) => {
        const rel = relationOf(n, focusId, edges);
        const color = nodeColor(rel, n.metrics || {});
        const metricLine = metricLabel(n);
        positioned.push({
          id: n.id,
          position: vertical ? { x, y: yBase + offs[i] } : { x: yBase + offs[i] * 1.6, y: x },
          data: { label: labelWith(n, metricLine) },
          style: {
            background: color, color: "white", border: "none", borderRadius: 10,
            padding: "6px 10px", fontSize: 11, width: 150, textAlign: "center",
            whiteSpace: "pre-line",
            boxShadow: rel === "focus" ? "0 0 0 3px #93c5fd" : "0 1px 3px rgba(0,0,0,.3)",
          },
        });
      });
    };
    place(byRel.focus, GROUP_X.focus, 0, true);
    place(byRel.substitute, GROUP_X.substitute, 0, true);
    place(byRel.copurchase, GROUP_X.copurchase, 0, true);
    place(byRel.category, 0, -220, true);
    place(byRel.other, -360, 240, true);
    // promos along the bottom (horizontal)
    place(byRel.promo, 240, 0, false);

    const rfEdges: Edge[] = edges.map((e, i) => ({
      id: `e${i}`,
      source: e.source,
      target: e.target,
      label: e.edge_type === "SUBSTITUTES" ? "sub" : e.edge_type === "CO_PURCHASED" ? "halo" : "",
      animated: e.edge_type === "CO_PURCHASED",
      style: { stroke: edgeColor(e.edge_type), strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: edgeColor(e.edge_type) },
      labelStyle: { fontSize: 9, fill: "#94a3b8" },
    }));
    return { rfNodes: positioned, rfEdges };
  }, [nodes, edges, focusId]);

  return (
    <div style={{ height: "100%", width: "100%" }}>
      <ReactFlowProvider>
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.2}
          proOptions={{ hideAttribution: true }}
          onNodeClick={(_, node) => {
            const n = nodes.get(node.id);
            if (n && n.node_type === "SKU") onNodeClick(n);
          }}
        >
          <Background color="#1e293b" gap={20} />
          <Controls showInteractive={false} />
          <Refit count={rfNodes.length} />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}

function edgeColor(t: string): string {
  if (t === "SUBSTITUTES") return "#f97316";
  if (t === "CO_PURCHASED") return "#0d9488";
  if (t === "PROMOTED_IN") return "#7c3aed";
  return "#475569";
}

function metricLabel(n: AppNode): string {
  const m = n.metrics || {};
  if (m.delta_margin !== undefined) {
    const v = m.delta_margin as number;
    return `${v >= 0 ? "+" : ""}$${v.toFixed(0)}`;
  }
  if (m.net_incremental_margin !== undefined) {
    const v = m.net_incremental_margin as number;
    return `net ${v >= 0 ? "+" : ""}$${v.toFixed(0)}`;
  }
  if (m.recommended_depth !== undefined) return `${((m.recommended_depth as number) * 100).toFixed(0)}% off`;
  return "";
}

function labelWith(n: AppNode, metric: string): string {
  return metric ? `${n.label}\n${metric}` : n.label;
}
