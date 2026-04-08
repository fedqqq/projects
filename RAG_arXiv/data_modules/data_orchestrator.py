import os
import logging
import time
from .data_ingestion import DataCollector


class Orchestrator:
    def __init__(self, data_collector: DataCollector, pdf_processor, max_pdfs: int = 100,
                 processed_dir: str = "./processed_data"):
        self.collector = data_collector
        self.processor = pdf_processor
        self.max_pdfs = max_pdfs
        self.processed_dir = processed_dir
        self.processed_count = 0
        self.logger = logging.getLogger(__name__)
        os.makedirs(processed_dir, exist_ok=True)

    def run(self):
        self.logger.info(f"Начинаем сбор и обработку. Максимум статей: {self.max_pdfs}")

        while self.processed_count < self.max_pdfs and self.collector.has_more():
            self.logger.info(f"Запрашиваем батч статей, текущий start={self.collector.current_start}")
            self.collector.collect()
            articles = self.collector.documents
            if not articles:
                self.logger.info("Получен пустой батч, завершаем.")
                break

            for article in articles:
                if self.processed_count >= self.max_pdfs:
                    break

                pdf_path = article.get('local_pdf')
                if not pdf_path or not os.path.exists(pdf_path):
                    self.logger.warning(f"PDF отсутствует для статьи {article.get('id')}, пропускаем.")
                    continue

                try:
                    self.logger.info(f"Обработка {pdf_path} (источник: {article['id']})")
                    article_metadata = {
                        'title': article.get('title'),
                        'authors': ', '.join(article.get('authors', [])),
                        'published': article.get('published'),
                    }
                    result = self.processor.process_pdf(pdf_path, source_id=article['full_id'],
                                                        article_metadata=article_metadata)
                    self.processor.save_processed_data(result, self.processed_dir)
                    os.remove(pdf_path)
                    self.logger.info(f"PDF удалён: {pdf_path}")
                    self.processed_count += 1
                    self.logger.info(f"Обработано статей: {self.processed_count}/{self.max_pdfs}")
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке {pdf_path}: {e}", exc_info=True)
                    continue

            time.sleep(1)

        self.logger.info(f"Процесс завершён. Обработано статей: {self.processed_count}")
