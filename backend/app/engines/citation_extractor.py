"""Citation provenance helpers for V3 evidence grounding.

These helpers map citation strings back to retrieved chunks, compute per-source
quality scores, and return node-level grounding summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _parse_citation_parts(citation: str) -> Tuple[Optional[str], Optional[str]]:
    body = citation.strip()
    if body.lower().startswith("source:"):
        body = body.split(":", 1)[1].strip()

    parts = [p.strip() for p in body.split("|") if p.strip()]
    cache_id: Optional[str] = None
    source_url: Optional[str] = None
    for part in parts:
        if part.lower().startswith("cache:"):
            cache_id = part.split(":", 1)[1].strip()
        elif part.startswith("http://") or part.startswith("https://"):
            source_url = part
    return cache_id, source_url


def _authority_score(source_url: Optional[str]) -> float:
    if not source_url:
        return 0.5
    host = (urlparse(source_url).hostname or "").lower()
    if host.endswith(".gov"):
        return 0.95
    if host.endswith(".edu"):
        return 0.9
    if host.endswith(".org"):
        return 0.82
    if source_url.startswith("https://"):
        return 0.72
    return 0.6


def _recency_score(created_at: Any) -> float:
    if not created_at:
        return 0.6
    try:
        dt = created_at
        if isinstance(created_at, str):
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if not isinstance(dt, datetime):
            return 0.6
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days_old = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.6

    if days_old <= 30:
        return 1.0
    if days_old <= 180:
        return 0.85
    if days_old <= 365:
        return 0.72
    return 0.55


def _verification_adjustment(status: str) -> float:
    normalized = (status or "unverified").lower()
    if normalized == "verified":
        return 0.08
    if normalized == "failed":
        return -0.18
    return 0.0


def _citation_quality_score(chunk: Dict[str, Any], source_url: Optional[str]) -> float:
    retrieval = _clamp(chunk.get("_similarity_score", 0.5) or 0.5)
    authority = _authority_score(source_url or chunk.get("source_url") or chunk.get("url"))
    recency = _recency_score(chunk.get("created_at"))
    verification_adj = _verification_adjustment(chunk.get("verification_status", "unverified"))

    score = (0.5 * retrieval) + (0.3 * authority) + (0.2 * recency) + verification_adj
    return round(_clamp(score), 3)


def _normalize_source_label(source: Any) -> str:
    if isinstance(source, str):
        return source.strip()
    if isinstance(source, dict):
        for key in ("source_url", "url", "title", "source_title", "id", "_id"):
            value = source.get(key)
            if value:
                return str(value).strip()
    return str(source).strip()


def build_citation_provenance(
    source_citations: Optional[Iterable[Any]],
    context_chunks: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build a minimal provenance list for a decision node.

    Each provenance entry includes the original citation label plus any matching
    context metadata we already have from retrieval.
    """

    chunks_by_label: Dict[str, Dict[str, Any]] = {}
    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    chunks_by_url: Dict[str, Dict[str, Any]] = {}
    for chunk in context_chunks or []:
        label = _normalize_source_label(chunk)
        if label:
            chunks_by_label[label.lower()] = chunk
        cid = str(chunk.get("id") or chunk.get("_id") or "").strip()
        if cid:
            chunks_by_id[cid.lower()] = chunk
        curl = str(chunk.get("source_url") or chunk.get("url") or "").strip()
        if curl:
            chunks_by_url[curl.lower()] = chunk

    provenance: List[Dict[str, Any]] = []
    for citation in source_citations or []:
        label = _normalize_source_label(citation)
        if not label:
            continue

        cache_id, source_url = _parse_citation_parts(label)
        matched_chunk = {}
        if cache_id:
            matched_chunk = chunks_by_id.get(cache_id.lower(), {})
        if not matched_chunk and source_url:
            matched_chunk = chunks_by_url.get(source_url.lower(), {})
        if not matched_chunk:
            matched_chunk = chunks_by_label.get(label.lower(), {})

        matched = bool(matched_chunk)
        quality_score = _citation_quality_score(matched_chunk if matched else {}, source_url)
        provenance.append(
            {
                "source_label": label,
                "cache_id": cache_id,
                "retrieval_evidence_id": str(matched_chunk.get("id") or matched_chunk.get("_id") or cache_id or ""),
                "source_url": (matched_chunk.get("source_url") or matched_chunk.get("url") or source_url),
                "source_title": matched_chunk.get("source_title") or matched_chunk.get("title"),
                "chunk_index": matched_chunk.get("chunk_index"),
                "retrieval_score": float(matched_chunk.get("_similarity_score", 0.0) or 0.0),
                "verification_status": matched_chunk.get("verification_status", "unverified"),
                "snapshot": (matched_chunk.get("content") or matched_chunk.get("text") or "")[:500],
                "snapshot_trace": (matched_chunk.get("content") or matched_chunk.get("text") or "")[:500],
                "matched": matched,
                "citation_quality_score": quality_score,
            }
        )

    return provenance


def summarize_provenance_quality(provenance: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, float]:
    entries = list(provenance or [])
    if not entries:
        return {
            "coverage": 0.0,
            "quality_score": 0.0,
            "matched_count": 0.0,
            "unmatched_count": 0.0,
            "completeness": 0.0,
        }

    matched_count = sum(1 for item in entries if item.get("matched"))
    quality_values = [float(item.get("citation_quality_score", 0.0) or 0.0) for item in entries]
    unmatched_count = len(entries) - matched_count
    return {
        "coverage": round(matched_count / len(entries), 3),
        "quality_score": round(sum(quality_values) / len(quality_values), 3),
        "matched_count": float(matched_count),
        "unmatched_count": float(unmatched_count),
        "completeness": round(matched_count / len(entries), 3),
    }