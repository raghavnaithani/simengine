from datetime import datetime, timezone
from typing import List, Dict, Any
from backend.app.utils.quality import compute_quality_score_for_node


def _node_citation_rate(node: Dict[str, Any]) -> float:
    # Prefer explicit citation_coverage when present
    if 'citation_coverage' in node and node.get('citation_coverage') is not None:
        try:
            return float(node.get('citation_coverage') or 0.0)
        except Exception:
            return 0.0
    # Fallback: treat presence of any `source_citations` as fully cited (1.0), else 0.0
    citations = node.get('source_citations') or []
    return 1.0 if len(citations) > 0 else 0.0


def filter_export_nodes(nodes: List[Dict[str, Any]], minimum_quality: float = 0.8, min_citation_rate: float = 0.8) -> List[Dict[str, Any]]:
    """Return nodes that satisfy export thresholds.

    Args:
        nodes: Sequence of decision node dicts from DB
        minimum_quality: Minimum quality_score (0.0-1.0)
        min_citation_rate: Minimum citation coverage (0.0-1.0)
    """
    out: List[Dict[str, Any]] = []
    recent_window: List[Dict[str, Any]] = []

    for node in nodes:
        quality = node.get('quality_score')
        if quality is None:
            quality = compute_quality_score_for_node(node, recent_window[-5:])
        citation_rate = _node_citation_rate(node)

        if quality >= float(minimum_quality) and citation_rate >= float(min_citation_rate):
            # Build export record with provenance
            record = {
                'id': node.get('id'),
                'title': node.get('title'),
                'summary': node.get('summary'),
                'confidence_score': node.get('confidence_score'),
                'quality_score': round(float(quality), 3),
                'speculative': bool(node.get('speculative', False)),
                'cache_ids': node.get('source_citations') or [],
                'citation_quality_score': float(node.get('citation_quality_score') or 0.0),
                'citation_coverage': float(node.get('citation_coverage') or citation_rate),
                'urls': node.get('urls') or [p.get('source_url') for p in (node.get('citation_provenance') or []) if p.get('source_url')],
                'raw_node': node,
            }
            out.append(record)
        recent_window.append(node)
    return out


def curate_top_fraction(nodes: List[Dict[str, Any]], fraction: float = 0.1) -> List[Dict[str, Any]]:
    """Select top fraction of nodes by quality_score for golden dataset."""
    if not nodes:
        return []
    scored = []
    for node in nodes:
        q = node.get('quality_score')
        if q is None:
            q = compute_quality_score_for_node(node, [])
        scored.append((float(q), node))
    scored.sort(key=lambda t: t[0], reverse=True)
    take = max(1, int(len(scored) * max(0.0, min(1.0, fraction))))
    return [n for _, n in scored[:take]]


def filter_and_serialize_nodes(
    nodes: List[Dict[str, Any]], 
    minimum_quality: float = 0.8, 
    min_citation_rate: float = 0.8
) -> List[Dict[str, Any]]:
    """Filter nodes by thresholds, serialize with full provenance, and return in deterministic order.
    
    Returns records sorted by created_at (ascending) then id for reproducibility.
    """
    filtered = filter_export_nodes(nodes, minimum_quality, min_citation_rate)
    
    # Sort by created_at (ascending) then id for deterministic ordering
    def sort_key(record):
        created = record.get('raw_node', {}).get('created_at')
        if isinstance(created, datetime):
            created = created.timestamp()
        else:
            created = 0.0
        node_id = record.get('id', '')
        return (created, node_id)
    
    filtered.sort(key=sort_key)
    
    # Return records without raw_node for cleaner output
    result = []
    for record in filtered:
        clean_record = {k: v for k, v in record.items() if k != 'raw_node'}
        result.append(clean_record)
    
    return result
