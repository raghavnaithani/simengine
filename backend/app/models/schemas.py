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
