import requests
import time
import xml.etree.ElementTree as ET
import os
import logging


class DataCollector:
    def __init__(self, bs, pdf_dir="./pdfs", query=None):
        self.url = "http://export.arxiv.org/api/query"
        self.batch_size = bs
        self.documents = []
        self.pdf_dir = pdf_dir
        self.current_start = 0
        self.total_results = None
        self.query = query
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("../arxiv_collector.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        os.makedirs(self.pdf_dir, exist_ok=True)

    def _build_search_query(self):
        if self.query:
            return self.query
        return '(all:"large language models" OR all:"LLM") AND (all:"recommender systems" OR all:"recommendation")'

    def download_pdf(self, pdf_url, filename):
        filepath = os.path.join(self.pdf_dir, filename)
        try:
            with requests.get(pdf_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.logger.info(f"PDF успешно сохранён: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Ошибка при скачивании {pdf_url}: {e}")
            return None

    def _parse_arxiv_response(self, xml_text):
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        root = ET.fromstring(xml_text)

        total_elem = root.find('opensearch:totalResults', namespaces)
        total_results = int(total_elem.text) if total_elem is not None else 0
        self.logger.info(f"Всего найдено статей по запросу: {total_results}")

        articles = []
        for entry in root.findall('atom:entry', namespaces):
            article_id = entry.find('atom:id', namespaces).text
            if article_id.startswith("http://arxiv.org/abs/"):
                article_id = article_id.replace("http://arxiv.org/abs/", "")

            title = entry.find('atom:title', namespaces).text
            summary = entry.find('atom:summary', namespaces).text

            authors = []
            for author_elem in entry.findall('atom:author', namespaces):
                name = author_elem.find('atom:name', namespaces).text
                authors.append(name)

            pdf_url = None
            full_id = article_id
            for link in entry.findall('atom:link', namespaces):
                if link.get('title') == 'pdf' and link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
                    if pdf_url:
                        full_id = os.path.basename(pdf_url).replace('.pdf', '')
                    break

            primary_category = None
            primary = entry.find('arxiv:primary_category', namespaces)
            if primary is not None:
                primary_category = primary.get('term')

            published = entry.find('atom:published', namespaces)
            published_date = published.text if published is not None else None

            article = {
                'id': article_id,
                'full_id': full_id,
                'title': title,
                'summary': summary,
                'authors': authors,
                'primary_category': primary_category,
                'published': published_date,
                'pdf_url': pdf_url
            }
            articles.append(article)

        return articles, total_results

    def collect(self):
        self.logger.info("Начинаем сбор данных с arXiv")

        params = {
            'search_query': self._build_search_query(),
            'start': self.current_start,
            'max_results': self.batch_size
        }

        try:
            self.logger.info(f"Выполняем запрос к arXiv с параметрами: {params}")
            r = requests.get(self.url, params=params, timeout=30)
            r.raise_for_status()
            self.logger.info("Запрос выполнен успешно, статус: 200")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при запросе к arXiv: {e}")
            return

        try:
            articles, total = self._parse_arxiv_response(r.text)
            self.total_results = total
            self.logger.info(f"Получено статей в ответе: {len(articles)}")
        except ET.ParseError as e:
            self.logger.error(f"Ошибка парсинга XML: {e}")
            return

        self.documents = articles
        self.current_start += len(articles)

        for idx, article in enumerate(articles, 1):
            if article['pdf_url']:
                safe_id = article['full_id'].replace('/', '_')
                filename = f"{safe_id}.pdf"
                self.logger.info(
                    f"[{idx}/{len(articles)}] Скачиваем PDF для статьи {article['full_id']}: {article['title'][:50]}..."
                )
                pdf_path = self.download_pdf(article['pdf_url'], filename)
                article['local_pdf'] = pdf_path
            else:
                self.logger.warning(f"У статьи {article['full_id']} нет ссылки на PDF")

        time.sleep(3)
        self.logger.info("Сбор данных завершён")

    def has_more(self):
        if self.total_results is None:
            return True
        return self.current_start < self.total_results
