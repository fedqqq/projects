import streamlit as st
import logging
import psycopg2
from data_modules.data_ingestion import DataCollector
from data_modules.data_preprocessing import PDFProcessor
from data_modules.data_orchestrator import Orchestrator
from data_modules.db_uploader import PGVectorUploader
from retriever import Retriever
from reranker import Reranker
from generator import Generator
import time
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'rag_db'),
    'user': os.getenv('DB_USER', 'rag_user'),
    'password': os.getenv('DB_PASSWORD', 'rag_password')
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

PROCESSED_DIR = "./arxiv_processed"
PDF_DIR = "./pdfs"
MODEL_PATH = os.getenv('MODEL_PATH')


@st.cache_resource
def init_retriever():
    return Retriever(db_config=DB_CONFIG, top_k=20)


@st.cache_resource
def init_reranker():
    return Reranker(
        diversity_method="max_per_source",
        max_per_source=2,
        use_cross_encoder=False,
        min_chunk_length=80,
        max_digit_ratio=0.25
    )


@st.cache_resource
def init_generator():
    return Generator(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=20,
        temperature=0.5,
        max_tokens=512,
        chat_format="llama-3"
    )


def get_db_stats():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT source_id) FROM document_chunks;")
        unique = cur.fetchone()[0]
        cur.close()
        conn.close()
        return total, unique
    except Exception as e:
        print(f"An exception happened: {e}")
        return 0, 0


def run_collection(topic, batch_size=10, max_pdfs=50):
    collector = DataCollector(bs=batch_size, pdf_dir=PDF_DIR, query=topic)
    processor = PDFProcessor(
        chunk_size=500,
        chunk_overlap=50,
        extract_tables=False,
        extract_formulas=True,
        table_method='pdfplumber'
    )
    orchestrator = Orchestrator(
        data_collector=collector,
        pdf_processor=processor,
        max_pdfs=max_pdfs,
        processed_dir=PROCESSED_DIR
    )
    orchestrator.run()

    uploader = PGVectorUploader(
        processed_dir=PROCESSED_DIR,
        db_config=DB_CONFIG,
        embedding_model_name="all-MiniLM-L6-v2",
        batch_size=100,
        recreate_table=False
    )
    chunks = uploader.load_json_files()
    uploader.upload_chunks(chunks)
    uploader.close()
    return len(chunks)


def main():
    st.set_page_config(page_title="RAG для научных статей", layout="wide")
    st.title("📚 RAG-система для научных статей (arXiv)")

    retriever = init_retriever()
    reranker = init_reranker()
    generator = init_generator()

    with st.sidebar:
        st.header("Статистика БД")
        total, unique = get_db_stats()
        st.metric("Всего чанков", total)
        st.metric("Уникальных статей", unique)

        st.divider()
        mode = st.radio("Режим", ["🔍 Поиск и ответ", "📥 Сбор статей"])

    if mode == "🔍 Поиск и ответ":
        st.header("Задайте вопрос по научным статьям")
        query = st.text_area("Ваш вопрос", height=100, placeholder="Например: What specific tuning techniques are introduced in TALLRec?")
        col1, col2 = st.columns([1, 5])
        with col1:
            top_k = st.number_input("Количество чанков", min_value=1, max_value=50, value=10)
        with col2:
            submit = st.button("Получить ответ", type="primary")

        if submit and query:
            with st.spinner("Ищу релевантные фрагменты..."):
                raw_results = retriever.retrieve(
                    query,
                    top_k=top_k * 2,
                    filter_useful=True,
                    min_length=80,
                    max_digit_ratio=0.2
                )
                final_chunks = reranker.rerank(query, raw_results, final_top_k=top_k)

            if not final_chunks:
                st.warning("Не найдено релевантных фрагментов. Попробуйте изменить запрос или собрать больше статей.")
            else:
                with st.expander("📄 Использованные источники", expanded=False):
                    for i, c in enumerate(final_chunks, 1):
                        src = c.get('source_id', 'unknown')
                        page = c.get('metadata', {}).get('page', '?')
                        title = c.get('metadata', {}).get('title', '')
                        authors = c.get('metadata', {}).get('authors', '')
                        if isinstance(authors, list):
                            authors = ', '.join(authors)
                        sim = c.get('similarity', 0)
                        st.markdown(f"**{i}.** *{title}*  \nАвторы: {authors}  \nИсточник: {src}, стр. {page} (релевантность: {sim:.3f})")
                        st.text(c['text'][:300] + "..." if len(c['text']) > 300 else c['text'])
                        st.divider()

                with st.spinner("Генерирую ответ..."):
                    result = generator.generate_with_chunks(
                        query=query,
                        chunks=final_chunks,
                        include_metadata=True,
                        citation_style="acm"
                    )

                st.subheader("💬 Ответ")
                st.markdown(result['response'])
                st.caption(f"Токены: prompt {result.get('prompt_tokens', 0)}, completion {result.get('completion_tokens', 0)}")

    else:
        st.header("Сбор новых статей с arXiv")
        st.markdown("Введите тему (поисковый запрос в формате arXiv, например: `all:\"large language models\" AND all:\"recommender systems\"`).")
        topic = st.text_input("Тема", value='all:"large language models" AND all:"recommender systems"')
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.number_input("Размер батча", min_value=1, max_value=50, value=10)
        with col2:
            max_pdfs = st.number_input("Максимум статей", min_value=1, max_value=200, value=50)
        with col3:
            start_collection = st.button("🚀 Начать сбор", type="primary")

        if start_collection:
            if not topic:
                st.error("Введите тему для поиска.")
            else:
                with st.spinner("Идёт сбор и обработка статей. Это может занять несколько минут..."):
                    start_time = time.time()
                    try:
                        num_chunks = run_collection(topic, batch_size, max_pdfs)
                        elapsed = time.time() - start_time
                        st.success(f"Сбор завершён! Добавлено {num_chunks} чанков за {elapsed:.1f} сек.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка при сборе: {e}")


if __name__ == "__main__":
    main()
