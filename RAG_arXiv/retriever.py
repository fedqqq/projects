import logging
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
            self,
            db_config: Dict[str, Any],
            embedding_model_name: str = "all-MiniLM-L6-v2",
            table_name: str = "document_chunks",
            top_k: int = 10,
            score_threshold: Optional[float] = None
    ):
        self.db_config = db_config
        self.table_name = table_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.logger.info(f"Размерность эмбеддингов: {self.embedding_dim}")

        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            self.logger.info("Подключение к БД установлено")
        except Exception as e:
            self.logger.error(f"Ошибка подключения к БД: {e}")
            raise

    def close(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self.logger.info("Соединение с БД закрыто")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            filter_useful: bool = False,
            min_length: int = 100,
            max_digit_ratio: float = 0.3
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        thr = threshold if threshold is not None else self.score_threshold

        fetch_k = k * 3 if filter_useful else k

        query_emb = self._generate_embedding(query)

        sql = f"""
            SELECT DISTINCT ON (source_id, chunk_index)
                id,
                source_id,
                chunk_index,
                page,
                text,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table_name}
        """
        params = [query_emb]

        if thr is not None:
            sql += " WHERE 1 - (embedding <=> %s::vector) >= %s"
            params.extend([query_emb, thr])

        sql += " ORDER BY source_id, chunk_index, similarity DESC LIMIT %s"
        params.append(fetch_k)

        try:
            self.cursor.execute(sql, params)
            results = self.cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении запроса: {e}")
            return []

        if filter_useful:
            results = [r for r in results if self.is_useful_chunk(r['text'], min_length, max_digit_ratio)]
            results = results[:k]

        self.logger.info(f"Найдено {len(results)} чанков для запроса: {query[:50]}...")
        return results

    @staticmethod
    def format_results(
            results: List[Dict[str, Any]],
            include_metadata: bool = True,
            max_text_length: Optional[int] = None
    ) -> str:
        if not results:
            return "Ничего не найдено."

        formatted = []
        for i, res in enumerate(results, 1):
            text = res['text']
            if max_text_length and len(text) > max_text_length:
                text = text[:max_text_length] + "..."

            if include_metadata:
                metadata = res['metadata']
                source = metadata.get('source', res['source_id'])
                page = metadata.get('page', '?')
                title = metadata.get('title', '')
                authors = metadata.get('authors', '')

                header = f"[{i}] Источник: {source}"
                if page:
                    header += f", стр. {page}"
                if title:
                    header += f"\n    Название: {title}"
                if authors:
                    header += f"\n    Авторы: {authors}"
                if 'similarity' in res:
                    header += f"\n    Релевантность: {res['similarity']:.3f}"
                formatted.append(f"{header}\n{text}\n")
            else:
                formatted.append(f"[{i}] {text}\n")

        return "\n".join(formatted)

    def retrieve_formatted(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None,
            include_metadata: bool = True,
            max_text_length: Optional[int] = 500,
            filter_useful: bool = False,
            min_length: int = 100,
            max_digit_ratio: float = 0.3
    ) -> str:
        results = self.retrieve(query, top_k, threshold, filter_useful, min_length, max_digit_ratio)
        return self.format_results(results, include_metadata, max_text_length)

    @staticmethod
    def is_useful_chunk(text: str, min_length: int = 100, max_digit_ratio: float = 0.3) -> bool:
        if len(text) < min_length:
            return False
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > max_digit_ratio:
            return False
        return True
