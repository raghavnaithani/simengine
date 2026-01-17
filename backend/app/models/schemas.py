from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from uuid import uuid4
from datetime import datetime

class KnowledgeChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source_url: str
    source_title: Optional[str] = None
    chunk_index: int = 0
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('risks')
    def must_have_risks(cls, v):
        if not v or len(v) == 0:
            raise ValueError('DecisionNode must include at least one risk')
        return v

    @validator('time_step', pre=True, always=True)
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

    @validator('risks', pre=True, always=True)
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
        # Normalize simple string risks into dicts
        out = []
        for item in v:
            if isinstance(item, str):
                out.append({'description': item, 'severity': 'Medium', 'likelihood': 'Medium'})
            elif isinstance(item, dict):
                # ensure required keys exist
                if not item.get('description'):
                    item['description'] = item.get('title') or 'Unknown Risk'
                if item.get('severity') not in ['Low', 'Medium', 'High', 'Critical']:
                    item['severity'] = 'Medium'
                if item.get('likelihood') not in ['Low', 'Medium', 'High']:
                    item['likelihood'] = 'Medium'
                out.append(item)
        if not out:
            return [
                {'description': 'General uncertainty.', 'severity': 'Low', 'likelihood': 'Low'}
            ]
        return out

    @validator('source_citations', pre=True)
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
