from typing import Optional

import numpy as np
import torch


def get_topk_similar_indices(
    q: torch.FloatTensor,
    keys: torch.FloatTensor,
    topk: int,
    method: Optional[str] = "cosine",
) -> np.array:
    """Find topk most similar indices of embeddings

    Args:
        q (torch.FloatTensor): query, dim=[1, d]
        keys (torch.FloatTensor): candidates, dim=[N, d]
        topk (int): number of returned indices
        method (Optional[str], optional): which metric. Defaults to "cosine".

    Returns:
        np.array: topk most similar indices
    """
    if method == "cosine":
        similarities = _get_cosine_similarities(q, keys)
    elif method == "dot":
        similarities = _get_dot_similarities(q, keys)
    elif method == "l2":
        similarities = _get_l2_similarities(q, keys)
    elif method == "l1":
        similarities = _get_l1_similarities(q, keys)
    else:
        raise ValueError(f"Method {method} not suppported.")
    
    similarities -= similarities.min()
    similarities /= similarities.max()
    similarities_topk, indices_topk = torch.topk(similarities, topk)
 
    return similarities_topk.cpu().detach().numpy(), indices_topk.cpu().detach().numpy()


def _get_cosine_similarities(q, keys):
    q_norm = _normalize_tensor(q)
    keys_norm = _normalize_tensor(keys)
    return _get_dot_similarities(q_norm, keys_norm)


def _normalize_tensor(x):
    return torch.nn.functional.normalize(x, p=2.0, dim=1)


def _get_dot_similarities(q, keys):
    return torch.mm(q, keys.transpose(0, 1))[0]


def _get_l2_similarities(q, keys):
    return -(((q - keys) ** 2).sum(dim=1)) ** 0.5


def _get_l1_similarities(q, keys):
    return -(torch.abs(q - keys)).sum(dim=1)
