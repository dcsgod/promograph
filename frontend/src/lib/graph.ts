// App-side graph model + layout, derived from the streamed graph events.

export interface AppNode {
  id: string;
  node_type: string; // SKU | Category | PromoEvent | CustomerSegment
  label: string;
  metrics: Record<string, any>;
}
export interface AppEdge {
  source: string;
  target: string;
  edge_type: string;
  weight: number;
}

export type Relation = "focus" | "substitute" | "copurchase" | "category" | "promo" | "other";

export function relationOf(node: AppNode, focusId: string | null, edges: AppEdge[]): Relation {
  if (focusId && node.id === focusId) return "focus";
  if (node.node_type === "Category") return "category";
  if (node.node_type === "PromoEvent") return "promo";
  const e = edges.find((x) => x.source === focusId && x.target === node.id);
  if (e?.edge_type === "SUBSTITUTES") return "substitute";
  if (e?.edge_type === "CO_PURCHASED") return "copurchase";
  return "other";
}

// role after prediction (cannibalized / halo) overrides base colour
export function nodeColor(rel: Relation, metrics: Record<string, any>): string {
  const role = metrics.role as string | undefined;
  if (role === "cannibalized" || (metrics.delta_margin ?? 0) < -0.5) return "#dc2626"; // red
  if (role === "halo" && (metrics.delta_margin ?? 0) > 0.5) return "#16a34a"; // green
  switch (rel) {
    case "focus": return "#2563eb"; // blue
    case "substitute": return "#f97316"; // orange (cannibalization risk)
    case "copurchase": return "#0d9488"; // teal (halo candidate)
    case "category": return "#6b7280"; // gray
    case "promo": return "#7c3aed"; // purple
    default: return "#475569";
  }
}
