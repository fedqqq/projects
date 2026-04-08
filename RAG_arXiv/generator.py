import logging
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
import os


class Generator:
    def __init__(
            self,
            model_path: str,
            n_ctx: int = 2048,
            n_gpu_layers: int = 0,
            temperature: float = 0.7,
            max_tokens: int = 1024,
            top_p: float = 0.9,
            repeat_penalty: float = 1.1,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            chat_format: str = "llama-3",
            verbose: bool = False,
            logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.logger.info(f"Загрузка модели из {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )
        self.logger.info("Модель загружена")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.chat_format = chat_format

        self.system_prompt = (
            "You are a helpful research assistant specialized in academic literature. "
            "Answer the user's question based strictly on the provided context. "
            "If the context does not contain enough information, say 'I don't have enough information to answer this question.' "
            "Do not make up facts or use external knowledge. "
            "Cite the sources by including the authors, title, and page number when possible, using the format: "
            "[Author names, Title, Source ID, Page]. "
            "For example: [Smith et al., 'Attention is All You Need', 1706.03762v5, p.3]. "
            "If multiple documents are relevant, synthesize information from them."
        )

    def _build_prompt(
            self,
            query: str,
            context: str,
            system_prompt: Optional[str] = None
    ) -> str:
        sys = system_prompt or self.system_prompt

        if self.chat_format == "llama-3":
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\nContext:\n{context}\n\nQuestion: {query}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif self.chat_format == "chatml":
            prompt = f"<|im_start|>system\n{sys}<|im_end|>\n"
            prompt += f"<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        elif self.chat_format == "vicuna":
            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Context:\n{context}\n\nQuestion: {query}\nASSISTANT:"
        elif self.chat_format == "alpaca":
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{sys}\n\n### Input:\nContext:\n{context}\n\nQuestion: {query}\n\n### Response:\n"
        else:
            prompt = f"System: {sys}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        return prompt

    def generate(
            self,
            query: str,
            context: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(query, context, system_prompt)

        params = {
            "prompt": prompt,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "echo": False,
            "stream": False
        }
        params.update(kwargs)

        try:
            self.logger.info(f"Генерация ответа (prompt length: {len(prompt)} символов)")
            output = self.llm(**params)

            if isinstance(output, dict):
                text = output["choices"][0]["text"]
                finish_reason = output["choices"][0].get("finish_reason", "stop")
                usage = output.get("usage", {})
            else:
                text = output["choices"][0]["text"] if hasattr(output, "choices") else str(output)
                finish_reason = "unknown"
                usage = {}

            result = {
                "response": text.strip(),
                "finish_reason": finish_reason,
                "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)
            }

            self.logger.info(f"Ответ сгенерирован ({result['completion_tokens']} токенов)")
            return result

        except Exception as e:
            self.logger.error(f"Ошибка при генерации: {e}")
            return {
                "response": f"Ошибка генерации: {e}",
                "finish_reason": "error",
                "usage": {},
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

    def generate_with_chunks(
            self,
            query: str,
            chunks: List[Dict[str, Any]],
            system_prompt: Optional[str] = None,
            include_metadata: bool = True,
            max_chars: Optional[int] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            citation_style: str = "acm"
    ) -> Dict[str, Any]:
        context_parts = []
        total_len = 0
        char_limit = max_chars or (self.llm.context_params.n_ctx * 4)

        for i, chunk in enumerate(chunks, 1):
            text = chunk['text']
            if include_metadata:
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', chunk.get('source_id', 'unknown'))
                page = metadata.get('page', '?')
                title = metadata.get('title', 'Untitled')
                authors = metadata.get('authors', 'Unknown')
                if isinstance(authors, list):
                    authors = ', '.join(authors)

                part = (
                    f"[Document {i}]\n"
                    f"Title: {title}\n"
                    f"Authors: {authors}\n"
                    f"Source: {source}, Page: {page}\n"
                    f"Content:\n{text}\n\n"
                )
            else:
                part = text

            if total_len + len(part) > char_limit:
                remaining = char_limit - total_len
                if remaining > 100:
                    context_parts.append(part[:remaining] + "...")
                break

            context_parts.append(part)
            total_len += len(part)

        context = "\n".join(context_parts)

        if system_prompt is None:
            system_prompt = self.system_prompt

        if include_metadata and citation_style:
            citation_instruction = self._get_citation_instruction(citation_style)
            system_prompt += "\n\n" + citation_instruction

        self.logger.info(f"Контекст подготовлен: {len(context)} символов")

        return self.generate(query, context, system_prompt, temperature, max_tokens)

    @staticmethod
    def _get_citation_instruction(style: str) -> str:
        instructions = {
            "acm": (
                "When citing sources, use the following format: "
                "[Author names, Title, Source ID, Page number]. "
                "For example: [Smith et al., 'Attention is All You Need', 1706.03762v5, p.3]."
            ),
            "apa": (
                "When citing sources, use APA style: (Author, Year, Page). "
                "If year is unavailable, use the source ID. "
                "For example: (Smith et al., 2017, p.3)."
            ),
            "ieee": (
                "When citing sources, use IEEE style: [1, p.3] and provide a reference list. "
                "But in this context, please include the source ID and page in brackets, "
                "like [1706.03762v5, p.3]."
            ),
        }
        return instructions.get(style, instructions["acm"])
