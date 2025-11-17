# backend/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

_logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_OK = True

def load_model(name: str = None):
    global _MODEL, EMBEDDINGS_OK
    if _MODEL is not None:
        return _MODEL
    try:
        nm = name or _MODEL_NAME
        _logger.info(f"Loading embedding model: {nm} (this may take a moment)...")
        _MODEL = SentenceTransformer(nm)
        _logger.info("Embedding model loaded.")
        EMBEDDINGS_OK = True
    except Exception as e:
        _logger.exception("Failed loading embedding model. Falling back to tag-based matching.")
        EMBEDDINGS_OK = False
        _MODEL = None
    return _MODEL

def embed_texts(texts):
    """
    Returns numpy array of embeddings. If model failed to load, returns zero vectors.
    """
    model = load_model()
    if model is None:
        # fallback: zero vectors (matching will rely on tag overlap)
        return np.zeros((len(texts), 384), dtype=float)
    try:
        embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # ensure finite
        embs = np.nan_to_num(embs, copy=False)
        return embs
    except Exception as e:
        _logger.exception("Error while embedding texts. Returning zeros.")
        return np.zeros((len(texts), model.get_sentence_embedding_dimension() if model else 384), dtype=float)

