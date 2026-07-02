"""Train SKU embeddings over the product graph and save to parquet.

Primary path : GraphSAGE (PyTorch Geometric), unsupervised link prediction.
Fallback path: numpy/sklearn spectral embedding (truncated SVD of the
               normalized adjacency augmented with node features).

Both write the SAME artifact: data/embeddings.parquet with columns
[product_id, e0..e{d-1}], so downstream code is oblivious to which ran.

Run:  python -m backend.gnn.train_graphsage
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import settings
from backend.gnn.build_pyg import GraphArrays, load_graph_arrays

EMB_DIM = 16
EPOCHS = 60
SEED = 42


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401

        return True
    except Exception:
        return False


def train_graphsage(g: GraphArrays, dim: int = EMB_DIM) -> np.ndarray:
    """GraphSAGE with an unsupervised link-prediction objective."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling

    torch.manual_seed(SEED)
    data = Data(
        x=torch.tensor(g.x, dtype=torch.float),
        edge_index=torch.tensor(g.edge_index, dtype=torch.long),
    )

    class SAGE(torch.nn.Module):
        def __init__(self, in_dim: int, hid: int, out: int):
            super().__init__()
            self.c1 = SAGEConv(in_dim, hid)
            self.c2 = SAGEConv(hid, out)

        def forward(self, x, ei):
            x = F.relu(self.c1(x, ei))
            return self.c2(x, ei)

    model = SAGE(g.x.shape[1], 2 * dim, dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    pos = data.edge_index
    n = data.num_nodes
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        z = model(data.x, data.edge_index)
        neg = negative_sampling(pos, num_nodes=n, num_neg_samples=pos.size(1))
        pos_score = (z[pos[0]] * z[pos[1]]).sum(-1)
        neg_score = (z[neg[0]] * z[neg[1]]).sum(-1)
        loss = -(
            F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean()
        )
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/{EPOCHS} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index).cpu().numpy()
    return z


def train_svd_fallback(g: GraphArrays, dim: int = EMB_DIM) -> np.ndarray:
    """Spectral embedding: truncated SVD of normalized adjacency + features."""
    from sklearn.decomposition import TruncatedSVD

    n = len(g.product_ids)
    A = np.zeros((n, n), dtype=float)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k], g.edge_index[1, k]
        A[i, j] += g.edge_weight[k]
    # symmetric normalization  D^-1/2 A D^-1/2
    deg = A.sum(1)
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    A_hat = A * dinv[:, None] * dinv[None, :]
    # augment adjacency structure with node features, then reduce
    M = np.hstack([A_hat, g.x])
    d = min(dim, M.shape[1] - 1, n - 1)
    svd = TruncatedSVD(n_components=d, random_state=SEED)
    z = svd.fit_transform(M)
    if z.shape[1] < dim:  # pad to fixed width
        z = np.hstack([z, np.zeros((n, dim - z.shape[1]))])
    return z


def main() -> None:
    g = load_graph_arrays()
    print(f"Graph: {len(g.product_ids)} SKUs, {g.edge_index.shape[1]} directed edges, "
          f"{g.x.shape[1]} node features")
    if _torch_available():
        print("Training GraphSAGE (PyTorch Geometric)...")
        z = train_graphsage(g)
        method = "graphsage"
    else:
        print("torch/PyG not available - using numpy SVD spectral embedding fallback.")
        z = train_svd_fallback(g)
        method = "svd_fallback"

    # L2-normalize embeddings (stabilizes downstream cosine/dot use)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cols = [f"e{i}" for i in range(z.shape[1])]
    out = pd.DataFrame(z, columns=cols)
    out.insert(0, "product_id", g.product_ids)
    settings.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(settings.embeddings_path, index=False)
    print(f"[{method}] wrote {out.shape[0]}x{z.shape[1]} embeddings -> {settings.embeddings_path}")


if __name__ == "__main__":
    main()
