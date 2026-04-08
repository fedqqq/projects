import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
            self,
            cross_encoder_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            use_cross_encoder: bool = False,
            diversity_method: str = "max_per_source",
            max_per_source: int = 2,
            mmr_lambda: float = 0.5,
            min_chunk_length: int = 100,
            max_digit_ratio: float = 0.3,
            bibliography_markers: Optional[List[str]] = None,
            logger: Optional[logging.Logger] = None
    ):
        self.use_cross_encoder = use_cross_encoder
        self.diversity_method = diversity_method
        self.max_per_source = max_per_source
        self.mmr_lambda = mmr_lambda
        self.min_chunk_length = min_chunk_length
        self.max_digit_ratio = max_digit_ratio
        self.bibliography_markers = bibliography_markers or ['arXiv:', 'doi:', 'Proceedings', 'Conference', 'Journal',
                                                             'Vol.', 'pp.']
        self.logger = logger or logging.getLogger(__name__)

        if self.use_cross_encoder:
            self.logger.info(f"Загрузка кросс-энкодера: {cross_encoder_model}")
            self.cross_encoder = CrossEncoder(cross_encoder_model)

    def is_useful_chunk(self, text: str) -> bool:
        if len(text) < self.min_chunk_length:
            return False

        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > self.max_digit_ratio:
            return False

        if any(marker in text for marker in self.bibliography_markers):
            return False

        lines = text.split('\n')
        if lines:
            numbered_lines = 0
            for line in lines[:10]:
                line = line.strip()
                if line and line[0].isdigit() and line[1:2] in ('.', ' '):
                    numbered_lines += 1
            if numbered_lines >= 5:
                return False

        return True

    def filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [c for c in chunks if self.is_useful_chunk(c['text'])]
        self.logger.info(f"Фильтрация: было {len(chunks)} чанков, стало {len(filtered)}")
        return filtered

    def cross_encoder_score(self, query: str, chunks: List[Dict[str, Any]]) -> List[float]:
        pairs = [[query, chunk['text']] for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)
        return scores

    def rerank_with_cross_encoder(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scores = self.cross_encoder_score(query, chunks)
        for chunk, score in zip(chunks, scores):
            chunk['cross_score'] = float(score)
        return sorted(chunks, key=lambda x: x['cross_score'], reverse=True)

    def diversify_by_source(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        per_source = defaultdict(list)
        for chunk in chunks:
            source = chunk.get('source_id') or chunk['metadata'].get('source', 'unknown')
            per_source[source].append(chunk)

        selected = []
        for src, items in per_source.items():
            items.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            selected.extend(items[:self.max_per_source])

        selected.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        self.logger.info(f"Диверсификация по источникам: было {len(chunks)} чанков, стало {len(selected)}")
        return selected

    def mmr_diversify(
            self,
            chunks: List[Dict[str, Any]],
            query_embedding: Optional[np.ndarray] = None,
            embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return []

        if embeddings is None:
            embeddings = [c.get('embedding') for c in chunks]
            if None in embeddings:
                raise ValueError("Для MMR необходимы эмбеддинги чанков (поле 'embedding').")

        if query_embedding is None:
            sim_scores = [c['similarity'] for c in chunks]
        else:
            sim_scores = [self._cos_sim(query_embedding, emb) for emb in embeddings]
            for c, s in zip(chunks, sim_scores):
                c['similarity'] = s

        remaining = list(range(len(chunks)))
        selected = []

        first_idx = np.argmax(sim_scores)
        selected.append(first_idx)
        remaining.remove(first_idx)

        while len(selected) < self.max_per_source * 10 and remaining:
            mmr_scores = []
            for idx in remaining:
                rel = sim_scores[idx]
                div = max([self._cos_sim(embeddings[idx], embeddings[sel]) for sel in selected])
                mmr = self.mmr_lambda * rel - (1 - self.mmr_lambda) * div
                mmr_scores.append(mmr)
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)

        result = [chunks[i] for i in selected]
        self.logger.info(f"MMR диверсификация: было {len(chunks)} чанков, стало {len(result)}")
        return result

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def rerank(
            self,
            query: str,
            chunks: List[Dict[str, Any]],
            query_embedding: Optional[np.ndarray] = None,
            final_top_k: int = 5
    ) -> List[Dict[str, Any]]:

        chunks = self.filter_chunks(chunks)
        if not chunks:
            return []

        if self.use_cross_encoder:
            chunks = self.rerank_with_cross_encoder(query, chunks)

        if self.diversity_method == "max_per_source":
            chunks = self.diversify_by_source(chunks)
        elif self.diversity_method == "mmr":
            if 'embedding' not in chunks[0]:
                self.logger.warning("Для MMR нужны эмбеддинги чанков. Пропускаем диверсификацию.")
            else:
                embeddings = [c['embedding'] for c in chunks]
                chunks = self.mmr_diversify(chunks, query_embedding, embeddings)

        return chunks[:final_top_k]
