from typing import Optional

import numpy as np
import faiss

def get_topk_similar_indices_faiss_ivf(
    q: np.ndarray,
    faissindex,
    topk: int,
) -> np.array:
    """Find topk most similar indices of embeddings

    Args:
        q (np.ndarray): query, dim=[1, d]
        faissindex (class): candidates, dim=[N, d]
        topk (int): number of returned indices.

    Returns:
        np.array: topk most similar indices
    """
     
    distances, indices = _get_topk_distances_and_indices(q, topk, faissindex)
    similarities = abs(distances[0] - distances[0].max()) + 100
    similarities = similarities / similarities.max()
    
    return similarities, indices[0]

def create_index(keys, nlist, nprobe, method):
    d = keys.shape[1]    
    if method == "cosine":
        quantizer = faiss.IndexFlatIP(d)
        faiss.normalize_L2(keys)
    elif method == "dot":
        quantizer = faiss.IndexFlatIP(d)
    elif method == 'l2':
        quantizer = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Method {method} not suppported.")
    
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    while not index.is_trained:
        index.train(keys)
    index.add(keys) 
    index.nprobe = nprobe
    
    return index 

def _get_topk_distances_and_indices(q, topk, index):
    d, i = index.search(q, topk)
    return d, i