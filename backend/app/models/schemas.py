from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4
from datetime import datetime, timezone

class Session(BaseModel):
    """Session model for reproducible simulation runs."""
    session_id: str
    prompt: str
    mode: str = "Analytical"
    persona: str = "Skeptical Analyst"
    seed: Optional[int] = None  # For reproducibility
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class KnowledgeChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source_url: str
    source_title: Optional[str] = None
    chunk_index: int = 0
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_days: int = 30
    verification_status: Literal['verified','unverified','failed'] = 'unverified'

class Risk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    severity: Literal['Low','Medium','High','Critical']
    likelihood: Literal['Low','Medium','High']
    mitigation_strategy: Optional[str] = None
    citation: Optional[str] = None

class Alternative(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    action_type: str
    expected_outcome_summary: Optional[str] = None

class DecisionNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    summary: str
    description: str
    time_step: int = 0
    created_by_engine: Optional[str] = None
    prompt_experiment_variant: Optional[str] = None
    prompt_experiment_batch_id: Optional[str] = None
    alternatives: List[Alternative] = []
    risks: List[Risk]
    source_citations: List[str] = []
    citation_provenance: List[dict] = []
    citation_quality_score: float = 0.0
    citation_coverage: float = 0.0
    confidence_score: float = 0.0
    speculative: bool = False
    quality_score: float = 0.0  # V2: 0-1 score from quality filters
    title_novelty_score: float = 0.0  # V2: 0-1 distance from recent nodes
    risk_specificity_score: float = 0.0  # V2: 0-1 non-generic measure
    error_reason: Optional[str] = None  # V1: why node failed or used fallback
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('risks')
    def must_have_risks(cls, v):
        # V1: Relaxed - if no risks, provide default instead of raising
        if not v or len(v) == 0:
            return [{'description': 'General uncertainty.', 'severity': 'Medium', 'likelihood': 'Medium'}]
        return v

    @field_validator('time_step', mode="before")
    def coerce_time_step(cls, v):
        # Accept float or numeric string and coerce to int, default 0
        try:
            if v is None or v == "":
                return 0
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                # attempt to parse numeric string
                if v.strip().isdigit():
                    return int(v.strip())
                try:
                    f = float(v)
                    return int(f)
                except Exception:
                    return 0
            return int(v)
        except Exception:
            return 0

    @field_validator('risks', mode="before")
    def ensure_risks(cls, v):
        # If missing/empty, provide a default risk
        if not v:
            return [
                {
                    'description': 'General uncertainty due to limited data.',
                    'severity': 'Low',
                    'likelihood': 'Low'
                }
            ]
        return v

    @field_validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v

    @field_validator('risks')
    def validate_high_severity_required(cls, v):
        """V1 Relaxed: Accept any severity distribution; don't auto-add or reject.
        
        Per project guide: ensure diversity but don't fail pipeline.
        """
        # Just pass through; let post-processing handle quality/specificity
        return v
    @field_validator('source_citations', mode="before")
    def validate_citations(cls, v):
        # Accept a variety of citation formats from model output. Normalize to
        # start with 'Source: ' for consistency. If v is falsy, return empty list.
        if not v:
            return []
        out = []
        for citation in v:
            if isinstance(citation, str):
                c = citation.strip()
                if not c.lower().startswith('source:'):
                    c = 'Source: ' + c
                out.append(c)
            elif isinstance(citation, dict):
                # prefer readable title or id
                if citation.get('title'):
                    out.append('Source: ' + str(citation.get('title')))
                elif citation.get('_id'):
                    out.append('Source: ' + str(citation.get('_id')))
                else:
                    out.append('Source: ' + str(citation))
            else:
                out.append('Source: ' + str(citation))
        return out

    @field_validator('source_citations', mode="before")
    def normalize_citations(cls, v):
        if not v:
            return []
        out = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # prefer readable title or id
                if item.get('title'):
                    out.append(item.get('title'))
                elif item.get('_id'):
                    out.append(str(item.get('_id')))
                else:
                    out.append(str(item))
        return out


class CuratorReview(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    node_id: str
    session_id: Optional[str] = None
    curator: str = "curator"
    action: Literal['approve', 'reject', 'edit']
    reason: str
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
