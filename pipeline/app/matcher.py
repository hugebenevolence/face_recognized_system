"""Nearest-neighbour search over registered faces using pgvector (cosine distance)."""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

log = logging.getLogger("matcher")

# Embeddings from different model packs live in different spaces, so only compare rows
# produced by the model currently loaded. The inner ORDER BY ... LIMIT lets pgvector use
# its HNSW index; we then keep each person's best photo and the top few people.
_QUERY = """
WITH nn AS (
    SELECT e.person_id, 1 - (e.embedding <=> %(q)s) AS sim
    FROM face_embeddings e
    WHERE e.model = %(model)s
    ORDER BY e.embedding <=> %(q)s
    LIMIT 30
)
SELECT p.id, p.full_name, p.person_code, max(nn.sim) AS similarity
FROM nn JOIN persons p ON p.id = nn.person_id
GROUP BY p.id, p.full_name, p.person_code
ORDER BY similarity DESC
LIMIT %(k)s
"""


@dataclass
class Match:
    person_id: int
    name: str
    code: str
    score: float  # cosine similarity, higher is better


class Matcher:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Optional[psycopg.Connection] = None
        self._lock = threading.Lock()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, autocommit=True)
        register_vector(conn)
        return conn

    def top_matches(self, embedding: np.ndarray, model: str, k: int = 3) -> list[Match]:
        """The k closest registered people (best photo each), best first.
        Empty if nobody is registered for this model or the DB is unavailable."""
        params = {"q": embedding, "model": model, "k": k}
        rows: list = []
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None or self._conn.closed:
                        self._conn = self._connect()
                    rows = self._conn.execute(_QUERY, params).fetchall()
                    break
                except psycopg.Error as exc:
                    # Includes "relation does not exist" while the backend is still
                    # creating the schema; drop the connection and retry once.
                    log.warning("match query failed (attempt %d): %s", attempt, exc)
                    if self._conn is not None:
                        self._conn.close()
                    self._conn = None
                    rows = []
        return [Match(person_id=r[0], name=r[1], code=r[2], score=float(r[3])) for r in rows]
