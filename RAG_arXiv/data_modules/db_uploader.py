import os
import json
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class PGVectorUploader:
    def __init__(
            self,
            processed_dir: str,
            db_config: Dict[str, Any],
            embedding_model_name: str = "all-MiniLM-L6-v2",
            batch_size: int = 100,
            recreate_table: bool = False
    ):
        self.processed_dir = processed_dir
        self.db_config = db_config
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.logger.info(f"Размерность эмбеддингов: {self.embedding_dim}")

        self._init_database(recreate_table)

    @staticmethod
    def _clean_text(text: str) -> str:
        return text.replace('\x00', '').replace('\u0000', '')

    def _clean_metadata(self, metadata: Any) -> Any:
        if isinstance(metadata, dict):
            return {k: self._clean_metadata(v) for k, v in metadata.items()}
        elif isinstance(metadata, list):
            return [self._clean_metadata(item) for item in metadata]
        elif isinstance(metadata, str):
            return self._clean_text(metadata)
        else:
            return metadata

    def _init_database(self, recreate_table: bool):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()

            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.logger.info("Расширение vector включено")

            table_name = "document_chunks"
            if recreate_table:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                self.logger.info(f"Таблица {table_name} удалена")

            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    source_id VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page INTEGER,
                    text TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({self.embedding_dim})
                );
            """)

            self.cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding 
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            self.logger.info(f"Таблица {table_name} готова")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации БД: {e}")
            raise

    def load_json_files(self) -> List[Dict[str, Any]]:
        all_chunks = []
        json_files = [f for f in os.listdir(self.processed_dir) if f.endswith('_chunks.json')]
        self.logger.info(f"Найдено JSON-файлов: {len(json_files)}")

        for json_file in tqdm(json_files, desc="Загрузка JSON"):
            file_path = os.path.join(self.processed_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                source_id = data.get('source_id', json_file.replace('_chunks.json', ''))

                for chunk in data.get('chunks', []):
                    chunk_text = self._clean_text(chunk['text'])
                    chunk_metadata = self._clean_metadata(chunk['metadata'])

                    chunk_data = {
                        'source_id': source_id,
                        'chunk_index': chunk_metadata.get('chunk_index'),
                        'page': chunk_metadata.get('page'),
                        'text': chunk_text,
                        'metadata': chunk_metadata
                    }
                    all_chunks.append(chunk_data)

            except Exception as e:
                self.logger.error(f"Ошибка при загрузке {file_path}: {e}")

        self.logger.info(f"Всего загружено чанков: {len(all_chunks)}")
        return all_chunks

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Генерация эмбеддингов"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return embeddings

    def upload_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            self.logger.warning("Нет чанков для загрузки")
            return

        texts = [chunk['text'] for chunk in chunks]

        self.logger.info("Генерация эмбеддингов...")
        embeddings = self.generate_embeddings(texts)

        insert_data = []
        for chunk, emb in zip(chunks, embeddings):
            insert_data.append((
                chunk['source_id'],
                chunk['chunk_index'],
                chunk['page'],
                chunk['text'],
                json.dumps(chunk['metadata']),
                emb.tolist()
            ))

        self.logger.info("Загрузка в БД...")
        table_name = "document_chunks"
        insert_query = f"""
            INSERT INTO {table_name} 
            (source_id, chunk_index, page, text, metadata, embedding)
            VALUES %s
        """

        try:
            execute_values(
                self.cursor,
                insert_query,
                insert_data,
                page_size=self.batch_size
            )
            self.conn.commit()
            self.logger.info(f"Успешно загружено {len(insert_data)} чанков")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Ошибка при загрузке в БД: {e}")
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
