"""Shared V2 quality scoring helpers.

These helpers are used by the simulation pipeline and telemetry aggregation
so the runtime and reporting layers share the same quality definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_list(item: Any, key: str) -> List[Any]:
    value = _get_value(item, key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def get_title_novelty_score(title: str, recent_titles: Sequence[str]) -> float:
    """Return a 0-1 novelty score using simple Jaccard distance."""
    if not recent_titles:
        return 0.9

    def jaccard_similarity(left: str, right: str) -> float:
        left_set = set(left.lower().split())
        right_set = set(right.lower().split())
        if not (left_set | right_set):
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    max_similarity = max((jaccard_similarity(title, recent) for recent in recent_titles), default=0.0)
    return round(1.0 - max_similarity, 2)


def get_risk_specificity_score(risk_description: str) -> float:
    """Return a 0-1 score for how concrete and quantified a risk is."""
    generic_phrases = [
        "general uncertainty",
        "risk of",
        "potential issue",
        "unknown factor",
        "undefined challenge",
        "unclear outcome",
        "possible failure",
    ]

    desc_lower = risk_description.lower()
    for phrase in generic_phrases:
        if phrase in desc_lower:
            return 0.2

    words = risk_description.split()
    word_count_score = min(len(words) / 20, 1.0)

    has_number = any(char.isdigit() for char in risk_description)
    has_metric = any(term in desc_lower for term in ["%", "month", "day", "hour", "margin", "revenue", "loss"])
    metric_score = 0.5 if has_number else 0.0
    metric_score += 0.3 if has_metric else 0.0

    return round(0.3 * word_count_score + 0.7 * metric_score, 2)


def compute_quality_score_for_node(node: Any, recent_nodes: Sequence[Any] | None = None) -> float:
    """Compute a composite 0-1 quality score for a decision node."""
    recent_nodes = list(recent_nodes or [])
    citations = _as_list(node, "source_citations")
    risks = _as_list(node, "risks")
    alternatives = _as_list(node, "alternatives")

    citation_score = 1.0 if citations else 0.0
    if len(citations) >= 2:
        citation_score = 0.9

    recent_titles = [str(_get_value(recent, "title", "")) for recent in recent_nodes if _get_value(recent, "title", "")]
    title_novelty = get_title_novelty_score(str(_get_value(node, "title", "")), recent_titles)

    risk_specificity_scores = [
        get_risk_specificity_score(str(_get_value(risk, "description", "")))
        for risk in risks
        if str(_get_value(risk, "description", ""))
    ]
    risk_specificity = sum(risk_specificity_scores) / len(risk_specificity_scores) if risk_specificity_scores else 0.0

    alternatives_score = min(len(alternatives) / 3, 1.0)

    quality_score = (
        0.3 * citation_score
        + 0.25 * title_novelty
        + 0.25 * risk_specificity
        + 0.2 * alternatives_score
    )

    return round(quality_score, 2)


def annotate_node_quality(node: Any, recent_nodes: Sequence[Any] | None = None) -> Any:
    """Mutate a node-like object with V2 quality scores and return it."""
    recent_nodes = list(recent_nodes or [])
    citations = _as_list(node, "source_citations")
    risks = _as_list(node, "risks")

    title = str(_get_value(node, "title", ""))
    recent_titles = [str(_get_value(recent, "title", "")) for recent in recent_nodes if _get_value(recent, "title", "")]
    title_novelty = get_title_novelty_score(title, recent_titles)

    risk_scores = [
        get_risk_specificity_score(str(_get_value(risk, "description", "")))
        for risk in risks
        if str(_get_value(risk, "description", ""))
    ]
    risk_specificity = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    quality_score = compute_quality_score_for_node(node, recent_nodes)

    try:
        setattr(node, "quality_score", quality_score)
        setattr(node, "title_novelty_score", title_novelty)
        setattr(node, "risk_specificity_score", round(risk_specificity, 2))
        if citations and not _get_value(node, "source_citations", None):
            setattr(node, "source_citations", citations)
    except Exception:
        pass

    return node


def summarize_nodes_for_telemetry(nodes: Sequence[Any]) -> Dict[str, float]:
    """Compute telemetry-ready quality metrics from a sequence of nodes."""
    nodes = list(nodes)
    if not nodes:
        return {
            "total_nodes": 0.0,
            "citation_rate": 0.0,
            "diversity_score": 0.0,
            "quality_score": 0.0,
            "error_rate": 0.0,
            "alternatives_count": 0.0,
            "title_novelty": 0.0,
            "risk_specificity": 0.0,
        }

    cited_nodes = 0
    qualities: List[float] = []
    novelty_scores: List[float] = []
    risk_scores: List[float] = []
    alternative_counts: List[float] = []
    fallback_nodes = 0
    recent_window: List[Any] = []

    for node in nodes:
        citations = _as_list(node, "source_citations")
        if citations:
            cited_nodes += 1
        if _get_value(node, "error_reason", None):
            fallback_nodes += 1

        title = str(_get_value(node, "title", ""))
        novelty_scores.append(get_title_novelty_score(title, [str(_get_value(recent, "title", "")) for recent in recent_window[-5:]]))

        node_risks = _as_list(node, "risks")
        per_node_risk_scores = [
            get_risk_specificity_score(str(_get_value(risk, "description", "")))
            for risk in node_risks
            if str(_get_value(risk, "description", ""))
        ]
        risk_scores.append(sum(per_node_risk_scores) / len(per_node_risk_scores) if per_node_risk_scores else 0.0)
        alternative_counts.append(float(len(_as_list(node, "alternatives"))))
        qualities.append(compute_quality_score_for_node(node, recent_window[-5:]))
        recent_window.append(node)

    total_nodes = len(nodes)
    citation_rate = cited_nodes / total_nodes if total_nodes else 0.0
    diversity_score = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
    quality_score = sum(qualities) / len(qualities) if qualities else 0.0
    error_rate = fallback_nodes / total_nodes if total_nodes else 0.0
    alternatives_count = sum(alternative_counts) / len(alternative_counts) if alternative_counts else 0.0
    title_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
    risk_specificity = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

    return {
        "total_nodes": float(total_nodes),
        "citation_rate": round(citation_rate, 3),
        "diversity_score": round(diversity_score, 3),
        "quality_score": round(quality_score, 3),
        "error_rate": round(error_rate, 3),
        "alternatives_count": round(alternatives_count, 3),
        "title_novelty": round(title_novelty, 3),
        "risk_specificity": round(risk_specificity, 3),
    }