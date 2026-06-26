import json
import re

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from app.rag.memory import SessionMemory
from app.rag.retriever import hybrid_retriever
from app.storage.faiss_store import vector_store


SYSTEM_PROMPT = """You are a helpful document assistant powered by hybrid RAG.
Answer using the provided document context first. If the context is insufficient,
clearly say what is missing and add brief, clearly labeled general knowledge only
when it helps the user. Be concise, accurate, and conversational.

Format responses with Markdown: use **bold** for key terms, bullet lists for steps,
`inline code` for values, and short headings (##) when structuring longer answers."""


class ChatEngine:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _build_context(self, query: str) -> tuple[str, list[str]]:
        chunks = hybrid_retriever.retrieve(query)
        if not chunks:
            return "", []
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(f"[{i}] Source: {chunk.source}\n{chunk.text}")
            sources.append(chunk.source)
        return "\n\n".join(context_parts), sorted(set(sources))

    def _build_messages(
        self, query: str, session: SessionMemory, context: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append(
                {"role": "system", "content": f"Document context:\n{context}"}
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": "No document context is available yet. Ask the user to upload files.",
                }
            )
        messages.extend(session.history_for_prompt())
        messages.append({"role": "user", "content": query})
        return messages

    def chat(self, query: str, session: SessionMemory) -> dict:
        context, sources = self._build_context(query)
        messages = self._build_messages(query, session, context)

        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or ""

        session.add("user", query)
        session.add("assistant", answer)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session.session_id,
        }

    def stream_chat(self, query: str, session: SessionMemory):
        context, sources = self._build_context(query)
        messages = self._build_messages(query, session, context)

        stream = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                parts.append(delta)
                yield {"type": "token", "content": delta}

        answer = "".join(parts)
        session.add("user", query)
        session.add("assistant", answer)

        followups = self.generate_suggestions("followup", session, answer)
        yield {
            "type": "done",
            "session_id": session.session_id,
            "sources": sources,
            "suggestions": followups,
        }

    def generate_suggestions(
        self,
        mode: str,
        session: SessionMemory | None = None,
        last_answer: str = "",
    ) -> list[str]:
        if not vector_store.is_ready:
            return [
                "Upload documents to get started",
                "What file types are supported?",
                "How does hybrid RAG work here?",
            ]

        sample = vector_store.sample_context()
        history = ""
        if session and session.messages:
            history = "\n".join(
                f"{msg.role}: {msg.content[:200]}"
                for msg in session.messages[-4:]
            )

        if mode == "followup" and last_answer:
            prompt = f"""Based on this document sample and the latest answer, suggest exactly 3 short
follow-up questions the user might ask next. Return JSON array of strings only.

Documents:
{sample}

Recent chat:
{history}

Latest answer:
{last_answer}
"""
        else:
            prompt = f"""Based on these uploaded documents, suggest exactly 3 short, useful starter
questions a user could ask. Return JSON array of strings only.

Documents:
{sample}
"""

        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON: [\"question1\", \"question2\", \"question3\"]",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        raw = response.choices[0].message.content or "[]"
        return self._parse_suggestions(raw)

    @staticmethod
    def _parse_suggestions(raw: str) -> list[str]:
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                items = json.loads(match.group())
                if isinstance(items, list):
                    return [str(item).strip() for item in items[:3] if str(item).strip()]
        except json.JSONDecodeError:
            pass
        lines = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
        return lines[:3] or [
            "Summarize the uploaded documents",
            "What are the key topics covered?",
            "List important details from the files",
        ]


chat_engine = ChatEngine()
