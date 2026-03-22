"""
cache.py — Two-layer caching for the Vision Bot.

Layer 1 — Exact hash cache (SQLite):
  SHA-256 of raw image bytes → stored result.
  Zero cost: identical images never call Ollama again.

Layer 2 — Semantic caption cache (sqlite-vec + sentence-transformers):
  Embed the caption with all-MiniLM-L6-v2.
  On future queries first check cosine similarity — if a very similar
  caption already exists (score ≥ SIMILARITY_THRESHOLD) return it.
  Useful for near-duplicate or slightly cropped versions of the same photo.

Usage:
  from cache import ImageCache
  cache = ImageCache()
  result = cache.get_by_hash(sha)           # exact hit
  result = cache.get_by_embedding(bytes)    # semantic hit
  cache.store(sha, bytes, result)           # save after Ollama call
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DB_PATH             = Path("vision_cache.db")
EMBED_MODEL         = "all-MiniLM-L6-v2"   # 22 MB, runs on CPU
SIMILARITY_THRESHOLD = 0.92                 # cosine similarity cutoff
EMBED_DIM           = 384                   # all-MiniLM-L6-v2 output dim


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class ImageCache:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._conn    = sqlite3.connect(str(db_path), check_same_thread=False)
        self._embed   = SentenceTransformer(EMBED_MODEL)
        self._setup_tables()
        logger.info("ImageCache ready (db=%s, embed=%s)", db_path, EMBED_MODEL)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _setup_tables(self) -> None:
        cur = self._conn.cursor()

        # Exact-hash table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                sha256  TEXT PRIMARY KEY,
                caption TEXT NOT NULL,
                tags    TEXT NOT NULL,          -- JSON array
                created DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Semantic-embedding table (sqlite-vec stores vectors as BLOBs)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS caption_embeddings (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                sha256  TEXT NOT NULL,
                caption TEXT NOT NULL,
                tags    TEXT NOT NULL,
                vector  BLOB NOT NULL           -- float32 numpy bytes
            )
        """)

        self._conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def sha256(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get_by_hash(self, sha: str) -> dict | None:
        """Layer 1: exact match."""
        row = self._conn.execute(
            "SELECT caption, tags FROM image_cache WHERE sha256 = ?", (sha,)
        ).fetchone()
        if row:
            logger.info("Exact cache HIT (sha=%s…)", sha[:12])
            return {"caption": row[0], "tags": json.loads(row[1])}
        return None

    def get_by_embedding(self, caption_query: str) -> dict | None:
        """
        Layer 2: semantic search over stored captions.
        Embeds caption_query and scans all stored vectors for cosine similarity.
        Returns best match if score ≥ SIMILARITY_THRESHOLD, else None.
        """
        rows = self._conn.execute(
            "SELECT caption, tags, vector FROM caption_embeddings"
        ).fetchall()
        if not rows:
            return None

        q_vec = self._embed.encode(caption_query, normalize_embeddings=True)

        best_score, best_row = -1.0, None
        for caption, tags_json, blob in rows:
            stored_vec = np.frombuffer(blob, dtype=np.float32)
            score = _cosine(q_vec, stored_vec)
            if score > best_score:
                best_score, best_row = score, (caption, tags_json)

        if best_score >= SIMILARITY_THRESHOLD and best_row:
            logger.info("Semantic cache HIT (score=%.3f)", best_score)
            return {"caption": best_row[0], "tags": json.loads(best_row[1])}

        logger.info("Semantic cache MISS (best=%.3f)", best_score)
        return None

    def store(self, sha: str, result: dict) -> None:
        """Persist a result in both cache layers."""
        caption   = result["caption"]
        tags_json = json.dumps(result["tags"])

        # Layer 1
        self._conn.execute(
            "INSERT OR IGNORE INTO image_cache (sha256, caption, tags) VALUES (?,?,?)",
            (sha, caption, tags_json),
        )

        # Layer 2 — embed caption and store vector
        vec  = self._embed.encode(caption, normalize_embeddings=True)
        blob = vec.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO caption_embeddings (sha256, caption, tags, vector) VALUES (?,?,?,?)",
            (sha, caption, tags_json, blob),
        )

        self._conn.commit()
        logger.info("Cached result for sha=%s…", sha[:12])

    def recent_captions(self, limit: int = 3) -> list[dict]:
        """Return the N most recently stored results."""
        rows = self._conn.execute(
            "SELECT caption, tags FROM image_cache ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"caption": r[0], "tags": json.loads(r[1])} for r in rows]
