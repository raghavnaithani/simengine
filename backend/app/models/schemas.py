from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
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
    alternatives: List[Alternative] = []
    risks: List[Risk]
    source_citations: List[str] = []
    confidence_score: float = 0.0
    speculative: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('risks')
    def must_have_risks(cls, v):
        if not v or len(v) == 0:
            raise ValueError('DecisionNode must include at least one risk')
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
    def validate_risk_severity(cls, v):
        for risk in v:
            if risk.severity == 'Critical' and risk.likelihood == 'High':
                raise ValueError('Critical risks with high likelihood must be mitigated or justified')
        return v

    @field_validator('risks')
    def validate_high_severity_required(cls, v):
        """Ensure at least one High severity risk is present.
        
        Per project guide Section 2.3: "DecisionNode must include risks: 
        List[Risk] and at least one High severity when applicable."
        
        Interpretation: Required for all decision nodes.
        """
        has_high_severity = any(risk.severity == 'High' for risk in v)
        if not has_high_severity:
            raise ValueError(
                'DecisionNode must include at least one High severity risk. '
                'Identify critical failure modes, challenges, or threats.'
            )
        return v

    @field_validator('source_citations', mode="before")
    def validate_citations(cls, v):
        for citation in v:
            if not citation.startswith('Source:'):
                raise ValueError(f'Invalid citation format: {citation}')
        return v

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
