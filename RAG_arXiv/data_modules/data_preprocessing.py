import os
import re
import json
import logging
from typing import List, Dict, Optional, Any
import fitz
import tiktoken
import pandas as pd
import pdfplumber
import camelot


class PDFProcessor:
    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            encoding_name: str = "cl100k_base",
            extract_tables: bool = True,
            extract_formulas: bool = True,
            table_method: str = "hybrid",
            formula_method: str = "latex",
            preserve_layout: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.extract_tables = extract_tables
        self.table_method = table_method
        self.formula_method = formula_method
        self.preserve_layout = preserve_layout
        self.extract_formulas_enabled = extract_formulas
        self.logger = logging.getLogger(__name__)

    def extract_text_with_positions(self, pdf_path: str) -> List[Dict]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        doc = fitz.open(pdf_path)
        page_elements = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            element_type = self._classify_span(span)

                            element = {
                                "page": page_num + 1,
                                "text": span["text"],
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],
                                "bbox": span["bbox"],
                                "type": element_type,
                                "language": "en"
                            }
                            page_elements.append(element)

        doc.close()
        self.logger.info(f"Извлечено {len(page_elements)} элементов с позициями")
        return page_elements

    @staticmethod
    def _classify_span(span: Dict) -> str:
        text = span["text"].strip()
        font = span["font"].lower()
        flags = span["flags"]

        if re.search(r'[a-zA-Z]?\d+[+\-*/=]\d+|[∑∫∏√∞≈≠≤≥]', text):
            return "formula_inline"

        if re.match(r'^Fig(ure)?\.?\s+\d+', text, re.IGNORECASE):
            return "caption_figure"

        if re.match(r'^Table\s+\d+', text, re.IGNORECASE):
            return "caption_table"

        if re.match(r'^[A-Z][A-Z\s]+$', text) and len(text) > 3:
            return "heading"

        if "bold" in font or flags & 2 ** 4:
            return "bold_text"

        if "italic" in font or flags & 2 ** 1:
            return "italic_text"

        return "regular_text"

    def extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[Dict]:
        tables = []

        try:
            if self.table_method == 'pdfplumber':
                with pdfplumber.open(pdf_path) as pdf:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        page_tables = page.extract_tables()

                        for table_idx, table_data in enumerate(page_tables):
                            if table_data and any(any(row) for row in table_data):
                                df = pd.DataFrame(table_data)

                                if len(df) > 1:
                                    header = df.iloc[0].tolist()
                                    data = df.iloc[1:].reset_index(drop=True)
                                    df = pd.DataFrame(data.values, columns=header)

                                table_dict = {
                                    "page": page_num,
                                    "index": table_idx,
                                    "data": df.to_dict(orient='records'),
                                    "headers": df.columns.tolist() if not df.empty else [],
                                    "shape": df.shape if not df.empty else (0, 0),
                                    "format": "dataframe",
                                    "markdown": df.to_markdown() if self.preserve_layout else None
                                }
                                tables.append(table_dict)

            elif self.table_method == 'camelot':
                tables_df = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
                for i, table in enumerate(tables_df):
                    df = table.df
                    table_dict = {
                        "page": page_num,
                        "index": i,
                        "data": df.to_dict(orient='records'),
                        "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                        "shape": df.shape,
                        "accuracy": table.accuracy,
                        "format": "dataframe",
                        "markdown": df.to_markdown() if self.preserve_layout else None
                    }
                    tables.append(table_dict)

            elif self.table_method == 'hybrid':
                tables = self.extract_tables_from_page(pdf_path, page_num)
                if not tables:
                    tables_df = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice', suppress_stdout=True)
                    for i, table in enumerate(tables_df):
                        df = table.df
                        table_dict = {
                            "page": page_num,
                            "index": i,
                            "data": df.to_dict(orient='records'),
                            "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                            "shape": df.shape,
                            "format": "dataframe",
                            "markdown": df.to_markdown() if self.preserve_layout else None
                        }
                        tables.append(table_dict)

        except Exception as e:
            self.logger.warning(f"Ошибка извлечения таблиц на странице {page_num}: {e}")

        return tables

    def extract_formulas(self, text: str) -> List[Dict]:
        formulas = []
        latex_patterns = [
            r'\$[^\$]+\$',
            r'\$\$[^\$]+\$\$',
            r'\\\[.*?\\\]',
            r'\\\(.*?\\\)',
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}'
        ]

        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                formulas.append({
                    "latex": match,
                    "type": "latex_environment",
                    "plain_text": self._latex_to_plain(match) if self.formula_method == 'latex' else None
                })

        unicode_formula_pattern = r'[∑∫∏√∞≈≠≤≥αβγΔπμσλ]+.*?[=+\-*/].*?'
        matches = re.findall(unicode_formula_pattern, text)
        for match in matches:
            if len(match) > 2:
                formulas.append({
                    "text": match,
                    "type": "unicode_math"
                })

        return formulas

    @staticmethod
    def _latex_to_plain(latex: str) -> str:
        replacements = {
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ',
            r'\sum': '∑', r'\int': '∫', r'\infty': '∞',
            r'\rightarrow': '→', r'\leftarrow': '←',
            r'\times': '×', r'\div': '÷', r'\pm': '±',
            r'\leq': '≤', r'\geq': '≥', r'\neq': '≠',
            r'\approx': '≈', r'\cdot': '·',
            r'\mathrm': '', r'\mathbf': '', r'\textit': '',
            r'\{': '', r'\}': '', r'\(': '', r'\)': ''
        }

        plain = latex
        for latex_char, unicode_char in replacements.items():
            plain = plain.replace(latex_char, unicode_char)

        plain = re.sub(r'\\[a-zA-Z]+', '', plain)
        return plain

    def clean_text(self, text: str, preserve_formulas: bool = True) -> str:
        if preserve_formulas and self.extract_formulas:
            formulas = self.extract_formulas(text)
            formula_placeholders = {}

            for i, formula in enumerate(formulas):
                placeholder = f"__FORMULA_{i}__"
                if 'latex' in formula:
                    text = text.replace(formula['latex'], placeholder)
                    formula_placeholders[placeholder] = formula['latex']
                elif 'text' in formula:
                    text = text.replace(formula['text'], placeholder)
                    formula_placeholders[placeholder] = formula['text']

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^\d+$', line):
                cleaned_lines.append(line)

        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)

        if preserve_formulas and self.extract_formulas_enabled:
            for placeholder, formula in formula_placeholders.items():
                text = text.replace(placeholder, f" {formula} ")

        return text.strip()

    def process_pdf(self, pdf_path: str, source_id: str, article_metadata: dict = None) -> Dict[str, Any]:
        self.logger.info(f"Начало обработки: {pdf_path}")
        result = {
            "source_id": source_id,
            "metadata": {
                "filename": os.path.basename(pdf_path),
                "size_bytes": os.path.getsize(pdf_path),
                **article_metadata
            },
            "chunks": [],
            "tables": [],
            "formulas": [],
            "structure": []
        }

        try:
            elements = self.extract_text_with_positions(pdf_path)
            pages = {}
            for elem in elements:
                page = elem["page"]
                if page not in pages:
                    pages[page] = []
                pages[page].append(elem)

            for page_num, page_elements in pages.items():
                page_text = " ".join([e["text"] for e in page_elements])

                cleaned_text = self.clean_text(page_text)

                if self.extract_formulas_enabled:
                    page_formulas = self.extract_formulas(page_text)
                    for formula in page_formulas:
                        formula["page"] = page_num
                        formula["source_id"] = source_id
                        result["formulas"].append(formula)

                if self.extract_tables:
                    page_tables = self.extract_tables_from_page(pdf_path, page_num)
                    for table in page_tables:
                        table["source_id"] = source_id
                        result["tables"].append(table)

                        if table.get("markdown"):
                            cleaned_text += f"\n\n[TABLE {table['index']} on page {page_num}]:\n{table['markdown']}"

                if cleaned_text:
                    page_chunks = self.split_into_chunks(
                        cleaned_text,
                        source_id,
                        page_num,
                        extra_metadata={
                            "has_tables": len(page_tables) > 0,
                            "has_formulas": len(page_formulas) > 0,
                            **article_metadata
                        }
                    )
                    result["chunks"].extend(page_chunks)

            result["metadata"]["pages_processed"] = len(pages)
            result["metadata"]["total_chunks"] = len(result["chunks"])
            result["metadata"]["total_tables"] = len(result["tables"])
            result["metadata"]["total_formulas"] = len(result["formulas"])

            self.logger.info(
                f"Обработка завершена: {len(result['chunks'])} чанков, "
                f"{len(result['tables'])} таблиц, {len(result['formulas'])} формул"
            )

        except Exception as e:
            self.logger.error(f"Критическая ошибка при обработке {pdf_path}: {e}")
            raise

        return result

    def split_into_chunks(
            self,
            text: str,
            source_id: str,
            page_num: Optional[int] = None,
            extra_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        chunks = []
        start = 0
        chunk_index = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            metadata = {
                "source": source_id,
                "page": page_num,
                "chunk_index": chunk_index,
                "start_token": start,
                "end_token": end,
                "tokens_count": len(chunk_tokens),
                "contains_formulas": bool(re.search(r'[∑∫∏√∞≈≠≤≥]', chunk_text)),
                "contains_table_ref": "[TABLE" in chunk_text
            }

            if extra_metadata:
                metadata.update(extra_metadata)

            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })

            next_start = start + self.chunk_size - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start
            chunk_index += 1

        return chunks

    def save_processed_data(self, result: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        source_id = result["source_id"]

        with open(os.path.join(output_dir, f"{source_id}_chunks.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": result["metadata"],
                "chunks": result["chunks"]
            }, f, ensure_ascii=False, indent=2)

        if result["tables"]:
            tables_dir = os.path.join(output_dir, "tables")
            os.makedirs(tables_dir, exist_ok=True)

            for i, table in enumerate(result["tables"]):
                if table.get("data"):
                    df = pd.DataFrame(table["data"])
                    csv_path = os.path.join(tables_dir, f"{source_id}_table_{i + 1}.csv")
                    df.to_csv(csv_path, index=False)

        if result["formulas"]:
            formulas_path = os.path.join(output_dir, f"{source_id}_formulas.json")
            with open(formulas_path, 'w', encoding='utf-8') as f:
                json.dump(result["formulas"], f, ensure_ascii=False, indent=2)

        self.logger.info(f"Результаты сохранены в {output_dir}")
