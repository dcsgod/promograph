// Mirrors backend/api/events.py - the LLM <-> Graph contract.

export interface TokenEvent { type: "token"; text: string; }
export interface ToolEvent { type: "tool"; name: string; status: "start" | "done" | "error"; detail: string; }
export interface GraphNodeEvent {
  type: "graph_node"; id: string; node_type: string; label: string; metrics: Record<string, any>;
}
export interface GraphEdgeEvent {
  type: "graph_edge"; source: string; target: string; edge_type: string; weight: number;
}
export interface NodeUpdateEvent { type: "node_update"; id: string; metrics: Record<string, any>; }
export interface RecommendationEvent { type: "recommendation"; data: Recommendation; }
export interface ErrorEvent { type: "error"; message: string; }
export interface DoneEvent { type: "done"; mode: string; }

export type TpoEvent =
  | TokenEvent | ToolEvent | GraphNodeEvent | GraphEdgeEvent
  | NodeUpdateEvent | RecommendationEvent | ErrorEvent | DoneEvent;

export interface Recommendation {
  product_id: number;
  label: string;
  recommended_depth: number;
  recommended_depth_pct: number;
  net_incremental_margin: number;
  clearance_value: number;
  total_value: number;
  net_roi: number;
  blended_margin_pct: number;
  margin_ok: boolean;
  inventory: number;
  weeks_of_supply: number;
  rationale: string;
  impact: {
    own_lift_units: number;
    cannibalization_margin: number;
    halo_margin: number;
    net_incremental_margin: number;
    top_cannibalized: { product_id: number; label: string; lost_units: number; lost_margin: number }[];
    top_halo: { product_id: number; label: string; gained_units: number; gained_margin: number }[];
  };
}
