# REMAINING FEATURES - COMPREHENSIVE IMPLEMENTATION GUIDE
## Based on Project Guide v1.2 Analysis - January 20, 2026

---

## ✅ COMPLETED FEATURES SUMMARY (4/6)
1. **Risk Layer Enforcement** - Citation validation, risk severity checks, confidence calibration (Jan 17, 2026)
2. **Hybrid Sparse Fallback** - MongoDB text search fallback for vector search < 0.7 (Jan 20, 2026)
3. **Temperature/Persona Injection** - Persona templates in ReasoningEngine (Jan 16, 2026)
4. **API Enhancement (Partial)** - GET /graph with session_id support (wrong URL pattern)

---

## ❌ REMAINING FEATURES (2 CRITICAL + 2 ENHANCEMENTS)

---

### **FEATURE 1: SPECULATIVE FLAG LOGIC (CRITICAL)**
**Status:** Schema exists, enforcement missing  
**Priority:** HIGH (v1.2 must-have per project guide)  
**Effort:** 2-4 hours

#### **WHY IT'S NEEDED (From Project Guide)**
> Section 5: "Any externally assertive claim lacking a matching chunk similarity >= 0.8 must include a citation or be flagged `speculative=true`."

> Section 9 - Hallucination Mitigation: "Speculative marking - Low-confidence claims are flagged `speculative`: the UI must render them with clear visual cues."

**Purpose:**
- **Anti-hallucination mechanism**: Prevents LLM from presenting ungrounded claims as facts
- **User trust**: Visual cues warn users when content is uncertain/speculative
- **Safety-first philosophy**: Aligns with DGS core principle of separating facts from reasoning

#### **WHAT'S ALREADY DONE**
✅ Schema field exists in `schemas.py`:
```python
class DecisionNode(BaseModel):
    speculative: bool = False  # Field exists but never set to True
```

✅ Prompt mentions it in `reasoner.py`:
```python
"5. If claim cannot be grounded, set speculative: true.\n"
```

❌ **NO AUTOMATIC ENFORCEMENT** - LLM can ignore the instruction, no validation check

#### **HOW TO IMPLEMENT (Detailed Steps)**

**STEP 1: Add Speculative Detection Logic to ReasoningEngine**

File: `backend/app/engines/reasoner.py`

Add method after `_calculate_confidence_score()`:

```python
def _should_mark_speculative(
    self, 
    confidence_score: float, 
    context_confidence: float,
    has_citations: bool,
    validation_retries: int
) -> bool:
    """Determine if a DecisionNode should be marked as speculative.
    
    Mark as speculative if ANY of these conditions are true:
    1. Confidence score < 0.5 (low overall confidence)
    2. Context confidence < 0.8 (weak retrieval similarity per project guide section 5)
    3. No citations found AND context_confidence < 0.9 (claims without grounding)
    4. Multiple validation retries (>= 2) indicate unstable reasoning
    
    Per project guide section 9: "Low-confidence claims are flagged speculative"
    Per project guide section 5: "lacking a matching chunk similarity >= 0.8 must be flagged speculative=true"
    
    Args:
        confidence_score: Calculated confidence (0.0-1.0)
        context_confidence: Max retrieval similarity (0.0-1.0)
        has_citations: Whether node includes any source citations
        validation_retries: Number of retries needed for valid output
        
    Returns:
        True if node should be marked speculative
    """
    # Rule 1: Overall confidence too low
    if confidence_score < 0.5:
        record_event(
            level="INFO", 
            action="speculative.low_confidence", 
            message=f"Marking speculative: confidence {confidence_score} < 0.5"
        )
        return True
    
    # Rule 2: Retrieval similarity below threshold (project guide: 0.8)
    if context_confidence < 0.8:
        record_event(
            level="INFO", 
            action="speculative.low_similarity", 
            message=f"Marking speculative: context similarity {context_confidence} < 0.8"
        )
        return True
    
    # Rule 3: No citations and weak grounding
    if not has_citations and context_confidence < 0.9:
        record_event(
            level="INFO", 
            action="speculative.no_citations", 
            message=f"Marking speculative: no citations, similarity {context_confidence} < 0.9"
        )
        return True
    
    # Rule 4: Multiple retries indicate unstable reasoning
    if validation_retries >= 2:
        record_event(
            level="INFO", 
            action="speculative.retries", 
            message=f"Marking speculative: {validation_retries} validation retries"
        )
        return True
    
    return False
```

**STEP 2: Update generate_decision() to Apply Speculative Flag**

In `generate_decision()`, after creating the node, before returning:

```python
# In generate_decision(), after: node = DecisionNode(**clean_data)
# Before: return node

# Determine if node should be marked speculative
has_citations = bool(clean_data.get('source_citations', []))
should_be_speculative = self._should_mark_speculative(
    confidence_score=confidence_score,
    context_confidence=context_confidence,
    has_citations=has_citations,
    validation_retries=validation_retries
)

if should_be_speculative and not node.speculative:
    # Update node to mark as speculative
    clean_data['speculative'] = True
    node = DecisionNode(**clean_data)
    record_event(
        level="INFO", 
        action="speculative.flag_applied", 
        message=f"Node {node.id} marked speculative",
        details={
            "confidence_score": confidence_score,
            "context_confidence": context_confidence,
            "has_citations": has_citations,
            "validation_retries": validation_retries
        }
    )
```

**STEP 3: Add Validation Test**

File: `backend/tests/test_reasoner.py`

Add test case:

```python
@pytest.mark.asyncio
async def test_speculative_flag_low_confidence():
    """Test that low confidence automatically sets speculative=true"""
    from backend.app.engines.reasoner import ReasoningEngine
    
    engine = ReasoningEngine()
    
    # Test _should_mark_speculative logic
    assert engine._should_mark_speculative(
        confidence_score=0.3, 
        context_confidence=0.3, 
        has_citations=False,
        validation_retries=0
    ) == True  # Low confidence should mark speculative
    
    assert engine._should_mark_speculative(
        confidence_score=0.6, 
        context_confidence=0.75, 
        has_citations=False,
        validation_retries=0
    ) == True  # Low similarity should mark speculative
    
    assert engine._should_mark_speculative(
        confidence_score=0.9, 
        context_confidence=0.9, 
        has_citations=True,
        validation_retries=0
    ) == False  # High confidence should NOT mark speculative
```

**STEP 4: Frontend Display (Optional but Recommended)**

File: `frontend/src/components/NodeCard.jsx` (or similar)

```jsx
{node.speculative && (
  <div className="speculative-warning" style={{
    background: '#fff3cd',
    border: '1px solid #ffc107',
    padding: '8px',
    borderRadius: '4px',
    marginTop: '8px',
    fontSize: '0.9em'
  }}>
    ⚠️ <strong>Speculative:</strong> This decision contains unverified or low-confidence claims.
    Review source citations carefully.
  </div>
)}
```

#### **ACCEPTANCE CRITERIA**
- [ ] `_should_mark_speculative()` method added with all 4 rules
- [ ] `generate_decision()` applies speculative flag automatically
- [ ] Test coverage for speculative logic (low confidence, low similarity, no citations, retries)
- [ ] Speculative nodes logged with clear event markers
- [ ] (Optional) Frontend displays speculative warning badge

#### **FILES TO MODIFY**
1. `backend/app/engines/reasoner.py` - Add logic
2. `backend/tests/test_reasoner.py` - Add tests
3. `frontend/src/components/NodeCard.jsx` - Add UI warning (optional)

---

### **FEATURE 2: TELEMETRY METRICS (CRITICAL)**
**Status:** Not implemented  
**Priority:** HIGH (v1.2 short-term per roadmap)  
**Effort:** 3-5 hours

#### **WHY IT'S NEEDED (From Project Guide)**
> Section 11 - Logging and Minimal Telemetry: "Local logs (console or file) should include: `[METRIC] Latency: Xs | CacheHit: True/False | TopSim: 0.87 | Retries: N`"

> Section 16 - Roadmap - Short term: "Minimal telemetry logging for latency and retry rates."

**Purpose:**
- **Performance monitoring**: Track system performance without external SaaS
- **Debugging aid**: Identify bottlenecks (slow scraping, LLM retries, cache misses)
- **Optimization guidance**: Data-driven decisions on what to improve
- **Local-first principle**: All telemetry stays on user's machine

#### **WHAT'S MISSING**
❌ No structured metrics collection
❌ No latency tracking for key operations
❌ No cache hit/miss ratio tracking
❌ No retry count aggregation
❌ No similarity score distribution logging

#### **HOW TO IMPLEMENT (Detailed Steps)**

**STEP 1: Create Metrics Collection Utility**

File: `backend/app/utils/metrics.py` (NEW FILE)

```python
"""
Local telemetry metrics collection for DGS.
All metrics stay on user's machine - no external reporting.
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime
from backend.app.utils.logger import append_log, record_event


class MetricsCollector:
    """Singleton for collecting and logging local telemetry metrics."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metrics_buffer = []
        return cls._instance
    
    def record_metric(
        self, 
        operation: str, 
        latency_ms: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        similarity_score: Optional[float] = None,
        retry_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a metric event for local telemetry.
        
        Per project guide section 11: Log latency, cache hits, similarity, retries
        
        Args:
            operation: Name of operation (e.g., 'rag.retrieval', 'llm.generate', 'scrape.parallel')
            latency_ms: Operation duration in milliseconds
            cache_hit: Whether cache was used (True/False/None)
            similarity_score: Top similarity score for retrieval operations
            retry_count: Number of retries for failed operations
            chunk_count: Number of chunks processed/retrieved
            success: Whether operation succeeded
            details: Additional context
        """
        metric_parts = [f"[METRIC] {operation}"]
        
        if latency_ms is not None:
            metric_parts.append(f"Latency: {latency_ms:.0f}ms")
        
        if cache_hit is not None:
            metric_parts.append(f"CacheHit: {cache_hit}")
        
        if similarity_score is not None:
            metric_parts.append(f"TopSim: {similarity_score:.2f}")
        
        if retry_count is not None:
            metric_parts.append(f"Retries: {retry_count}")
        
        if chunk_count is not None:
            metric_parts.append(f"Chunks: {chunk_count}")
        
        metric_parts.append(f"Success: {success}")
        
        metric_line = " | ".join(metric_parts)
        append_log(metric_line)
        
        # Also record as structured event for analysis
        record_event(
            level="METRIC",
            action=f"metric.{operation}",
            message=metric_line,
            details={
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "similarity_score": similarity_score,
                "retry_count": retry_count,
                "chunk_count": chunk_count,
                "success": success,
                **(details or {})
            }
        )


# Global singleton instance
metrics = MetricsCollector()
```

**STEP 2: Add Context Manager for Latency Tracking**

Add to `backend/app/utils/metrics.py`:

```python
from contextlib import contextmanager

@contextmanager
def track_latency(operation: str, **kwargs):
    """Context manager to automatically track operation latency.
    
    Usage:
        with track_latency('rag.retrieval', cache_hit=True):
            # ... perform operation ...
            pass
    """
    start_time = time.time()
    success = True
    try:
        yield
    except Exception as e:
        success = False
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_metric(
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            **kwargs
        )
```

**STEP 3: Instrument ContextBuilder (RAG Metrics)**

File: `backend/app/engines/scraper.py`

```python
# Add import at top
from backend.app.utils.metrics import metrics, track_latency

# In retrieve_relevant_chunks():
async def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
    """Retrieve relevant chunks with telemetry tracking."""
    
    with track_latency('rag.retrieval'):
        # Vector search
        docs = await query_similar_chunks(query, top_k=k)
        
        cache_hit = len(docs) > 0
        top_similarity = max([d.get('similarity', 0.0) for d in docs], default=0.0)
        
        # Log retrieval metrics
        metrics.record_metric(
            operation='rag.vector_search',
            cache_hit=cache_hit,
            similarity_score=top_similarity,
            chunk_count=len(docs),
            success=True
        )
        
        # Hybrid fallback logic
        if not docs or top_similarity < 0.7:
            with track_latency('rag.text_fallback'):
                # ... existing fallback code ...
                
                fallback_count = len(text_results) if text_results else 0
                metrics.record_metric(
                    operation='rag.text_search',
                    cache_hit=fallback_count > 0,
                    chunk_count=fallback_count,
                    success=fallback_count > 0,
                    details={'trigger': 'low_vector_similarity', 'threshold': 0.7}
                )
        
        return docs

# In build_knowledge_base():
async def build_knowledge_base(self, query: str) -> str:
    """Build knowledge base with scraping metrics."""
    
    with track_latency('rag.ingestion'):
        # ... existing code ...
        
        # After scraping
        metrics.record_metric(
            operation='rag.parallel_scrape',
            latency_ms=scrape_duration_ms,  # calculate this
            chunk_count=total_chunks_stored,
            success=True,
            details={'sources_scraped': len(sources)}
        )
```

**STEP 4: Instrument ReasoningEngine (LLM Metrics)**

File: `backend/app/engines/reasoner.py`

```python
# Add import at top
from backend.app.utils.metrics import metrics, track_latency

# In generate_decision():
async def generate_decision(...) -> DecisionNode:
    
    with track_latency('llm.generate'):
        # ... existing code ...
        
        # After successful node creation
        metrics.record_metric(
            operation='llm.decision_generation',
            retry_count=json_attempt,  # from retry loop
            success=True,
            details={
                'confidence_score': confidence_score,
                'persona': persona,
                'temperature': temperature,
                'has_citations': has_citations
            }
        )
        
        return node

# In _call_model():
async def _call_model(self, prompt: str, temperature: float = 0.7, timeout: float = 300.0) -> str:
    
    with track_latency('llm.api_call', retry_count=attempts):
        # ... existing call ...
        
        metrics.record_metric(
            operation='llm.ollama_call',
            retry_count=attempts,
            success=True,
            details={'model': self.model, 'temperature': temperature}
        )
```

**STEP 5: Add Metrics Aggregation Endpoint (Optional)**

File: `backend/app/main.py`

```python
@app.get('/metrics/summary')
async def get_metrics_summary():
    """Get aggregated metrics summary for debugging/monitoring."""
    db = await get_database()
    
    # Aggregate from events collection
    pipeline = [
        {'$match': {'level': 'METRIC'}},
        {'$group': {
            '_id': '$action',
            'count': {'$sum': 1},
            'avg_latency': {'$avg': '$details.latency_ms'},
            'success_rate': {'$avg': {'$cond': ['$details.success', 1, 0]}}
        }},
        {'$sort': {'count': -1}}
    ]
    
    results = await db['events'].aggregate(pipeline).to_list(length=100)
    
    return {
        'metrics': results,
        'total_operations': sum(r['count'] for r in results)
    }
```

#### **ACCEPTANCE CRITERIA**
- [ ] `metrics.py` utility created with `MetricsCollector` class
- [ ] `track_latency()` context manager implemented
- [ ] ContextBuilder instrumented (vector search, text fallback, scraping)
- [ ] ReasoningEngine instrumented (LLM calls, retries)
- [ ] All metrics follow project guide format: `[METRIC] Operation | Latency | CacheHit | etc`
- [ ] Metrics stored in MongoDB events collection for analysis
- [ ] (Optional) GET /metrics/summary endpoint for aggregation

#### **FILES TO CREATE/MODIFY**
1. **CREATE:** `backend/app/utils/metrics.py` - New utility
2. **MODIFY:** `backend/app/engines/scraper.py` - Add RAG metrics
3. **MODIFY:** `backend/app/engines/reasoner.py` - Add LLM metrics
4. **MODIFY:** `backend/app/main.py` - Add metrics endpoint (optional)

---

### **FEATURE 3: API URL PATTERN FIX (ENHANCEMENT)**
**Status:** Wrong URL pattern  
**Priority:** MEDIUM (spec compliance)  
**Effort:** 1 hour

#### **WHY IT'S NEEDED (From Project Guide)**
> Section 6 - API Surface: "GET /graph/{session_id} - behavior: return full graph JSON with nodes, edges, and node metadata"

> Section 15 - Example branch sequence: "GET /graph/{session_id} endpoint missing"

**Purpose:**
- **REST compliance**: Session ID should be a path parameter, not query param
- **API clarity**: `/graph/{session_id}` is more RESTful than `/graph?session_id=X`
- **Spec alignment**: Project guide explicitly specifies path parameter pattern

#### **CURRENT STATE**
```python
@app.get('/graph')  # Wrong - query param
async def get_graph(session_id: str = None):
    if session_id:
        # ... session-specific logic exists
```

#### **HOW TO IMPLEMENT**

File: `backend/app/main.py`

**STEP 1: Create New Path-Parameter Endpoint**

```python
@app.get('/graph/{session_id}')
async def get_session_graph(session_id: str):
    """Get graph for a specific session (v1.2 spec compliant).
    
    Per project guide section 6: GET /graph/{session_id}
    Returns full graph JSON with nodes, edges, and metadata.
    """
    try:
        sim_engine = SimulationEngine()
        return await sim_engine.get_session_graph(session_id)
    except Exception as e:
        record_event(level='ERROR', action='graph.session.error', message=str(e), details={'session_id': session_id})
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/graph')
async def get_all_graphs():
    """Get all nodes/edges across all sessions (backward compatibility).
    
    Deprecated: Use GET /graph/{session_id} for session-specific graphs.
    This endpoint remains for backward compatibility only.
    """
    try:
        db = await get_database()
        nodes = await db['decision_nodes'].find().to_list(length=1000)
        edges = await db['edges'].find().to_list(length=1000)
        
        # Sanitize MongoDB _id fields
        for n in nodes:
            if '_id' in n:
                n['_id'] = str(n['_id'])
        for e in edges:
            if '_id' in e:
                e['_id'] = str(e['_id'])
        
        return {'nodes': nodes, 'edges': edges, 'deprecated': True}
    except Exception as e:
        record_event(level='ERROR', action='graph.all.error', message=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

**STEP 2: Update Frontend API Calls**

File: `frontend/src/api/graphApi.js` (or wherever API calls are made)

```javascript
// OLD (deprecated):
// const response = await fetch(`/graph?session_id=${sessionId}`);

// NEW (spec compliant):
const response = await fetch(`/graph/${sessionId}`);
```

#### **ACCEPTANCE CRITERIA**
- [ ] `GET /graph/{session_id}` endpoint created with path parameter
- [ ] `GET /graph` endpoint kept for backward compatibility (marked deprecated)
- [ ] Frontend updated to use new URL pattern
- [ ] Both endpoints tested and working

#### **FILES TO MODIFY**
1. `backend/app/main.py` - Add new endpoint
2. `frontend/src/api/graphApi.js` - Update API calls (if exists)

---

### **FEATURE 4: PROMPT TEMPLATES UTILITY (ENHANCEMENT)**
**Status:** Templates inline in reasoner.py  
**Priority:** LOW (code organization)  
**Effort:** 1-2 hours

#### **WHY IT'S NEEDED (From Project Guide)**
> Section 14 - Developer Ergonomics: "backend/app/utils/prompt_templates.py for system prompts, persona injection, citation rules"

> Section 4.2 - ReasoningEngine: "Construct the system prompt template enforcing a strict JSON schema"

**Purpose:**
- **Code organization**: Separate prompt engineering from execution logic
- **Maintainability**: Easier to iterate on prompts without touching engine code
- **Reusability**: Other engines could use same persona/citation templates
- **Testing**: Easier to test prompt variations independently

#### **CURRENT STATE**
All prompts are hardcoded inline in `backend/app/engines/reasoner.py`:
- Persona templates in `_get_persona_prompt()`
- System instructions in `generate_decision()`
- Retry prompts inline

#### **HOW TO IMPLEMENT**

**STEP 1: Create Prompt Templates Utility**

File: `backend/app/utils/prompt_templates.py` (NEW FILE)

```python
"""
Prompt templates for DGS v1.2 ReasoningEngine.
Separates prompt engineering from execution logic.
"""
from typing import Dict

# Persona templates
PERSONA_TEMPLATES: Dict[str, str] = {
    "Skeptical Analyst": (
        "You are a skeptical strategic analyst. You focus on identifying critical risks, "
        "potential failures, and worst-case scenarios. You question assumptions and demand evidence."
    ),
    "Optimistic Founder": (
        "You are an optimistic founder. You focus on opportunities, growth potential, and creative solutions. "
        "You see challenges as opportunities for innovation."
    ),
    "Cautious Regulator": (
        "You are a cautious regulator. You prioritize compliance, risk mitigation, and systematic evaluation. "
        "You require thorough documentation and evidence."
    ),
    "Aggressive Founder": (
        "You are an aggressive founder. You prioritize speed, market capture, and bold moves. "
        "You accept calculated risks for high rewards."
    ),
    "Pessimistic Analyst": (
        "You are a pessimistic analyst. You expect things to go wrong and identify failure modes early. "
        "You emphasize defensive strategies and risk avoidance."
    ),
}

# System instruction template
SYSTEM_INSTRUCTION_TEMPLATE = """You are a strategic simulation engine. {persona_text}

CRITICAL RULES:
1. You MUST respond with ONLY valid JSON. No explanatory text, no markdown, no code blocks.
2. Start your response with {{ and end with }}. Nothing else.
3. Schema: {{id, title, summary, description, time_step (int), risks: [{{description, severity (Low/Medium/High/Critical), likelihood (Low/Medium/High)}}], alternatives: [{{description, action_type}}]}}
4. Every factual claim should ideally include [Source: cache:<id> | <url>] inline where used.
5. If claim cannot be grounded, set speculative: true.
6. Ensure all strings are properly quoted, all commas are correct, no trailing commas.
"""

# Citation enforcement rules
CITATION_RULES = """
CITATION REQUIREMENTS:
- For facts from provided KnowledgeChunks: append [Source: cache:<chunk_id> | <url>]
- For speculative claims: set speculative: true and append [Source: speculative]
- Missing required citations will cause validation failure
"""

# Retry instruction template
RETRY_INSTRUCTION_TEMPLATE = """
IMPORTANT: Previous attempt failed with error: {error}
You MUST output ONLY valid JSON. Check for: missing quotes, trailing commas, unclosed braces, control characters.
"""


def get_persona_prompt(persona: str = "Skeptical Analyst") -> str:
    """Get persona-specific prompt text.
    
    Args:
        persona: Persona name (must be in PERSONA_TEMPLATES)
        
    Returns:
        Persona description text
    """
    return PERSONA_TEMPLATES.get(persona, PERSONA_TEMPLATES["Skeptical Analyst"])


def build_system_instruction(persona: str = "Skeptical Analyst") -> str:
    """Build complete system instruction with persona injection.
    
    Args:
        persona: Persona name
        
    Returns:
        Complete system instruction text
    """
    persona_text = get_persona_prompt(persona)
    return SYSTEM_INSTRUCTION_TEMPLATE.format(persona_text=persona_text)


def build_retry_instruction(error: str) -> str:
    """Build retry instruction with error context.
    
    Args:
        error: Error message from previous attempt
        
    Returns:
        Retry instruction text
    """
    return RETRY_INSTRUCTION_TEMPLATE.format(error=error)
```

**STEP 2: Refactor ReasoningEngine to Use Templates**

File: `backend/app/engines/reasoner.py`

```python
# Add import at top
from backend.app.utils.prompt_templates import (
    get_persona_prompt,
    build_system_instruction,
    build_retry_instruction,
    CITATION_RULES
)

class ReasoningEngine:
    # Remove _get_persona_prompt() method - now in templates.py
    
    async def generate_decision(...):
        # Replace inline instruction building with:
        instruction = build_system_instruction(persona)
        
        full_prompt = f"{instruction}\n\n{CITATION_RULES}\n\nSCENARIO: {prompt}\n\nCONTEXT: {json.dumps(context or {}, default=str)}\n\nJSON OUTPUT:"
        
        # In retry loop, replace inline retry instruction with:
        if json_attempt > 0:
            retry_instruction = build_retry_instruction(last_parse_error)
            retry_prompt = f"{full_prompt}{retry_instruction}\n\nJSON OUTPUT:"
```

**STEP 3: Add Tests for Prompt Templates**

File: `backend/tests/test_prompt_templates.py` (NEW FILE)

```python
import pytest
from backend.app.utils.prompt_templates import (
    get_persona_prompt,
    build_system_instruction,
    build_retry_instruction,
    PERSONA_TEMPLATES
)


def test_get_persona_prompt():
    """Test persona prompt retrieval."""
    skeptical = get_persona_prompt("Skeptical Analyst")
    assert "skeptical" in skeptical.lower()
    assert "risk" in skeptical.lower()
    
    optimistic = get_persona_prompt("Optimistic Founder")
    assert "optimistic" in optimistic.lower()
    assert "opportunit" in optimistic.lower()


def test_build_system_instruction():
    """Test system instruction building with persona."""
    instruction = build_system_instruction("Cautious Regulator")
    assert "cautious regulator" in instruction.lower()
    assert "valid JSON" in instruction
    assert "speculative" in instruction


def test_build_retry_instruction():
    """Test retry instruction with error."""
    error_msg = "Invalid JSON: trailing comma"
    retry = build_retry_instruction(error_msg)
    assert error_msg in retry
    assert "Previous attempt failed" in retry


def test_all_personas_available():
    """Test all documented personas are in templates."""
    expected_personas = [
        "Skeptical Analyst",
        "Optimistic Founder", 
        "Cautious Regulator",
        "Aggressive Founder",
        "Pessimistic Analyst"
    ]
    for persona in expected_personas:
        assert persona in PERSONA_TEMPLATES
```

#### **ACCEPTANCE CRITERIA**
- [ ] `prompt_templates.py` utility created with all templates
- [ ] All persona templates moved from reasoner.py
- [ ] System instruction template with persona injection
- [ ] Citation rules template
- [ ] Retry instruction template
- [ ] ReasoningEngine refactored to use templates
- [ ] Tests for prompt template functions
- [ ] No behavior change (refactor only)

#### **FILES TO CREATE/MODIFY**
1. **CREATE:** `backend/app/utils/prompt_templates.py` - New utility
2. **CREATE:** `backend/tests/test_prompt_templates.py` - New tests
3. **MODIFY:** `backend/app/engines/reasoner.py` - Use templates instead of inline

---

## IMPLEMENTATION PRIORITY ORDER

### Phase 1: Critical Features (1-2 days)
1. **Speculative Flag Logic** (2-4 hours) - Anti-hallucination, v1.2 must-have
2. **Telemetry Metrics** (3-5 hours) - Debugging, performance monitoring

### Phase 2: Enhancements (1 day)
3. **API URL Pattern Fix** (1 hour) - REST compliance
4. **Prompt Templates Utility** (1-2 hours) - Code organization

**Total Estimated Effort:** 7-12 hours (1.5-2 days)

---

## TESTING STRATEGY

### For Each Feature:
1. **Unit Tests** - Test individual functions/methods
2. **Integration Tests** - Test feature in full flow
3. **Manual Verification** - Run simulation and verify logs/output
4. **Regression Check** - Ensure existing 8/8 tests still pass

### Success Criteria:
- All new tests pass
- Existing test suite remains 8/8 passing
- No performance degradation
- Logs show new metrics/flags working
- Documentation updated in project_log.txt

---

## NOTES FROM PROJECT GUIDE

### Key Principles to Follow:
1. **Local-first:** All metrics stay on user's machine (no external SaaS)
2. **Safety-first:** Speculative flag prevents hallucination presentation
3. **Practical:** Keep memory/CPU requirements reasonable
4. **Developer-friendly:** Clear logs, structured code, good tests

### Success Metrics:
- Speculative nodes are clearly flagged when confidence < 0.5 or similarity < 0.8
- Metrics logged in format: `[METRIC] Operation | Latency | CacheHit | TopSim | Retries`
- API follows REST conventions with path parameters
- Prompt templates are DRY and testable

---

## REFERENCES
- Project Guide v1.2: Lines 1-429 (all sections reviewed)
- Section 5: Schema validation rules (speculative flag)
- Section 9: Hallucination mitigation (confidence, citations)
- Section 11: Telemetry requirements (local metrics)
- Section 6: API surface (URL patterns)
- Section 14: Developer ergonomics (utilities structure)
