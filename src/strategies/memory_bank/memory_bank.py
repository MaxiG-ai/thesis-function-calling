from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from src.utils.logger import get_logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger("MemoryBank")


@dataclass
class ToolExecution:
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Dict[str, Any]
    tool_call_id: Optional[str] = None


@dataclass
class InsightRecord:
    trace_id: str
    summary: str


def _parse_json_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"value": payload}
    return {"value": payload}


def extract_tool_executions(messages: Iterable[Dict[str, Any]]) -> List[ToolExecution]:
    tool_calls: Dict[str, Dict[str, Any]] = {}
    tool_names: Dict[str, str] = {}

    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                function = call.get("function", {})
                if not call_id or not isinstance(function, dict):
                    continue
                tool_name = function.get("name", "unknown_tool")
                if tool_name == "retrieve_relevant_history":
                    continue
                tool_calls[call_id] = _parse_json_payload(function.get("arguments"))
                tool_names[call_id] = tool_name

    executions: List[ToolExecution] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if call_id not in tool_calls:
            continue
        raw_output = _parse_json_payload(message.get("content"))
        executions.append(
            ToolExecution(
                tool_name=tool_names.get(call_id, "unknown_tool"),
                raw_input=tool_calls[call_id],
                raw_output=raw_output,
                tool_call_id=call_id,
            )
        )

    return executions


def extract_retrieval_query(messages: Iterable[Dict[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict):
                    continue
                if function.get("name") != "retrieve_relevant_history":
                    continue
                args = _parse_json_payload(function.get("arguments"))
                query = args.get("query") or args.get("input")
                if isinstance(query, str):
                    return query
    return ""


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


class MemoryBank:
    def __init__(self, top_k: int = 3, record_char_limit: int = 2000):
        self.top_k = top_k
        self.record_char_limit = record_char_limit
        self.fact_store: Dict[str, Dict[str, Any]] = {}
        self.insight_store: List[InsightRecord] = []
        self._tool_call_ids: set[str] = set()

    def reset(self) -> None:
        self.fact_store.clear()
        self.insight_store.clear()
        self._tool_call_ids.clear()

    def ingest_tool_executions(
        self,
        user_query: str,
        tool_executions: Iterable[ToolExecution],
        llm_client: Any,
        summary_prompt: Optional[str] = None,
        summary_char_limit: int = 10000,
    ) -> List[str]:
        trace_ids: List[str] = []
        if llm_client is None:
            raise ValueError("llm_client is required to summarize tool executions")

        for execution in tool_executions:
            if execution.tool_call_id and execution.tool_call_id in self._tool_call_ids:
                continue
            trace_id = str(uuid4())
            raw_output = execution.raw_output

            self.fact_store[trace_id] = {
                "trace_id": trace_id,
                "step_id": len(self.fact_store) + 1,
                "tool_name": execution.tool_name,
                "raw_input": execution.raw_input,
                "raw_output": raw_output,
            }

            summary = self._summarize_execution(
                user_query=user_query,
                tool_name=execution.tool_name,
                raw_output=raw_output,
                llm_client=llm_client,
                summary_prompt=summary_prompt,
                summary_char_limit=summary_char_limit,
            )

            self.insight_store.append(InsightRecord(trace_id=trace_id, summary=summary))
            trace_ids.append(trace_id)

            if execution.tool_call_id:
                self._tool_call_ids.add(execution.tool_call_id)

        return trace_ids

    def _summarize_execution(
        self,
        user_query: str,
        tool_name: str,
        raw_output: Dict[str, Any],
        llm_client: Any,
        summary_prompt: Optional[str],
        summary_char_limit: int,
    ) -> str:
        raw_text = json.dumps(raw_output, ensure_ascii=False)
        if len(raw_text) > summary_char_limit:
            raw_text = raw_text[:summary_char_limit]

        prompt = summary_prompt or (
            "Summarize what this tool execution achieved. Highlight key entities "
            "(IDs, coordinates, status) but do not output the full JSON."
        )

        prompt_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"User query: {user_query}\nTool: {tool_name}\nRaw output: {raw_text}",
            },
        ]

        response = llm_client.generate_plain(input_messages=prompt_messages, model="gpt-4-1-mini")
        message = response.choices[0].message
        summary_text = ""
        if isinstance(message, dict):
            summary_text = (message.get("content") or "").strip()
        else:
            summary_text = (getattr(message, "content", "") or "").strip()

        if not summary_text:
            summary_text = f"Tool '{tool_name}' executed."
        return summary_text

    def retrieve_relevant_history(self, query: str) -> str:
        records = self._rank_records(query)[: self.top_k]
        formatted_records = []
        for index, record in enumerate(records, start=1):
            raw_data = self.fact_store.get(record.trace_id, {})
            raw_text = json.dumps(raw_data, ensure_ascii=False)
            raw_text = _truncate_text(raw_text, self.record_char_limit)
            formatted_records.append(
                "\n".join(
                    [
                        f"[RETRIEVED RECORD {index}]",
                        f"Summary: {record.summary}",
                        f"Raw Data: {raw_text}",
                        "-------------------",
                    ]
                )
            )

        return "\n".join(formatted_records)

    def _rank_records(self, query: str) -> List[InsightRecord]:
        if not query:
            return list(self.insight_store)
        summaries = [record.summary for record in self.insight_store]
        if not summaries:
            return []

        vectorizer = TfidfVectorizer()
        summary_vectors = vectorizer.fit_transform(summaries)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, summary_vectors)[0]
        ranked = sorted(
            enumerate(similarities),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        return [self.insight_store[index] for index, _ in ranked]
