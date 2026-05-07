import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.engines.reasoner import ReasoningEngine
import pytest
import json

@pytest.fixture
def reasoning_engine():
    return ReasoningEngine()

def test_valid_json_with_citations(reasoning_engine):
    raw_text = """
    {
        "key": "value"
    }
    [Source: cache:valid_citation | http://example.com]
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    assert result["key"] == "value"
    assert "citations" in result
    assert len(result["citations"]) > 0
    assert "cache:valid_citation" in result["citations"][0]

def test_invalid_citation(reasoning_engine):
    raw_text = """
    {
        "key": "value"
    }
    [Source: invalid_format_no_cache]
    """
    with pytest.raises(ValueError, match="Invalid citation format"):
        reasoning_engine._extract_and_clean_json(raw_text)

def test_missing_citations(reasoning_engine):
    raw_text = """
    {
        "key": "value"
    }
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    assert result["key"] == "value"
    assert "citations" in result
    assert result["citations"] == []

def test_empty_input(reasoning_engine):
    raw_text = ""
    with pytest.raises(ValueError, match="Empty or invalid input text"):
        reasoning_engine._extract_and_clean_json(raw_text)

def test_large_input(reasoning_engine):
    raw_text = "{" + ",".join([f"\"key{i}\": \"value{i}\"" for i in range(1000)]) + "} [Source: cache:large_test | http://example.com]"
    result = reasoning_engine._extract_and_clean_json(raw_text)
    assert len(result) > 1000  # 1000 keys + citations field
    assert "citations" in result
    assert len(result["citations"]) > 0


# Speculative Flag Logic Tests
def test_speculative_flag_low_confidence(reasoning_engine):
    """Test that low confidence automatically sets speculative=true"""
    # Low confidence should mark speculative
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.3,
        context_confidence=0.3,
        has_citations=False,
        validation_retries=0
    ) == True
    
    # High confidence should NOT mark speculative
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.9,
        context_confidence=0.9,
        has_citations=True,
        validation_retries=0
    ) == False


def test_speculative_flag_low_similarity(reasoning_engine):
    """Test that low retrieval similarity marks speculative"""
    # Context confidence < 0.8 should mark speculative (per project guide)
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.6,
        context_confidence=0.75,
        has_citations=True,
        validation_retries=0
    ) == True
    
    # Context confidence >= 0.8 should NOT mark speculative
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.85,
        context_confidence=0.85,
        has_citations=True,
        validation_retries=0
    ) == False


def test_speculative_flag_no_citations(reasoning_engine):
    """Test that missing citations with weak grounding marks speculative"""
    # No citations + context < 0.9 should mark speculative
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.8,
        context_confidence=0.85,
        has_citations=False,
        validation_retries=0
    ) == True
    
    # Has citations should NOT trigger this rule
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.8,
        context_confidence=0.85,
        has_citations=True,
        validation_retries=0
    ) == False


def test_speculative_flag_multiple_retries(reasoning_engine):
    """Test that multiple validation retries mark speculative"""
    # 2+ retries should mark speculative
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.9,
        context_confidence=0.9,
        has_citations=True,
        validation_retries=2
    ) == True
    
    # 0-1 retries should NOT mark speculative (if other conditions met)
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.9,
        context_confidence=0.9,
        has_citations=True,
        validation_retries=1
    ) == False


def test_speculative_flag_combined_conditions(reasoning_engine):
    """Test edge cases and combined conditions"""
    # Borderline case: confidence = 0.5 should NOT be speculative (threshold is < 0.5)
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.5,
        context_confidence=0.8,
        has_citations=True,
        validation_retries=0
    ) == False
    
    # Borderline case: context = 0.8 should NOT be speculative (threshold is < 0.8)
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.8,
        context_confidence=0.8,
        has_citations=True,
        validation_retries=0
    ) == False
    
    # Multiple failing conditions
    assert reasoning_engine._should_mark_speculative(
        confidence_score=0.2,
        context_confidence=0.6,
        has_citations=False,
        validation_retries=3
    ) == True

# FIX 4: NEW TEST CASES FOR CITATION ENFORCEMENT FEATURE
# These tests verify the complete citation enforcement pipeline


def test_citation_field_mapping(reasoning_engine):
    """Test that extracted citations are mapped to source_citations field"""
    raw_text = """
    {
        "title": "Test Node",
        "summary": "Test",
        "description": "Test description",
        "time_step": 1,
        "risks": [{"description": "Risk 1", "severity": "Low", "likelihood": "Low"}],
        "alternatives": []
    }
    [Source: cache:abc123 | http://example.com]
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    
    # Check extraction with [Source:] format (FIX 1)
    assert "citations" in result, "Should extract citations field"
    assert len(result["citations"]) > 0, "Should find at least one citation"
    assert "cache:abc123" in result["citations"][0], "Should extract cache ID"
    
    # Apply janitor to simulate real flow
    clean_data = reasoning_engine._janitor_fix_data(result)
    
    # Verify mapping to source_citations (FIX 2)
    assert "source_citations" in clean_data, "source_citations field should exist"
    assert len(clean_data["source_citations"]) > 0, "source_citations should be populated"
    assert clean_data["source_citations"][0].startswith("Source:"), "Should be formatted with Source: prefix"


def test_source_format_extraction(reasoning_engine):
    """Test that [Source:] format (correct format) is extracted"""
    raw_text = """
    {
        "title": "Test",
        "summary": "Test",
        "description": "Test description",
        "time_step": 0,
        "risks": [{"description": "Risk", "severity": "Low", "likelihood": "Low"}],
        "alternatives": []
    }
    [Source: cache:id123 | https://example.com/page]
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    
    # Verify extraction with correct [Source:] format
    assert "citations" in result, "Should extract citations"
    assert len(result["citations"]) > 0, "Should find one citation"
    assert "cache:id123" in result["citations"][0], "Should extract cache ID"
    assert "example.com" in result["citations"][0], "Should extract URL"


def test_citation_format_validation(reasoning_engine):
    """Test that invalid citation formats are rejected"""
    raw_text = """
    {
        "title": "Test",
        "summary": "Test",
        "description": "Test",
        "time_step": 0,
        "risks": [{"description": "Risk", "severity": "Low", "likelihood": "Low"}],
        "alternatives": []
    }
    [Source: invalid_format_no_cache]
    """
    
    # Should raise error for invalid format
    with pytest.raises(ValueError, match="Invalid citation format"):
        reasoning_engine._extract_and_clean_json(raw_text)


def test_no_citations_empty_list(reasoning_engine):
    """Test that nodes without citations get empty source_citations"""
    raw_text = """
    {
        "title": "Test",
        "summary": "Test",
        "description": "Test description",
        "time_step": 1,
        "risks": [{"description": "Risk", "severity": "Low", "likelihood": "Low"}],
        "alternatives": []
    }
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    clean_data = reasoning_engine._janitor_fix_data(result)
    
    # Should have source_citations field but empty
    assert "source_citations" in clean_data, "source_citations field should exist"
    assert clean_data["source_citations"] == [], "source_citations should be empty when no citations found"


def test_multiple_citations_mapping(reasoning_engine):
    """Test that multiple citations are all mapped correctly"""
    raw_text = """
    {
        "title": "Test",
        "summary": "Test",
        "description": "Test description",
        "time_step": 0,
        "risks": [{"description": "Risk", "severity": "Low", "likelihood": "Low"}],
        "alternatives": []
    }
    [Source: cache:first | http://example.com]
    Some text
    [Source: cache:second | http://example.org]
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    clean_data = reasoning_engine._janitor_fix_data(result)
    
    # Should extract and map both citations
    assert len(clean_data["source_citations"]) == 2, "Should have 2 citations"
    assert all(c.startswith("Source:") for c in clean_data["source_citations"]), "All should have Source: prefix"


def test_speculative_when_no_citations(reasoning_engine):
    """Test that nodes without citations are marked speculative (per FIX 3)"""
    should_be_speculative = reasoning_engine._should_mark_speculative(
        confidence_score=0.75,
        context_confidence=0.85,
        has_citations=False,  # No citations
        validation_retries=0
    )
    assert should_be_speculative == True, "Node without citations should be marked speculative"


def test_not_speculative_with_citations(reasoning_engine):
    """Test that nodes with citations avoid speculative marking"""
    should_be_speculative = reasoning_engine._should_mark_speculative(
        confidence_score=0.75,
        context_confidence=0.85,
        has_citations=True,  # Has citations
        validation_retries=0
    )
    assert should_be_speculative == False, "Node with citations and good confidence should not be speculative"


def test_context_citations_are_available_for_model_omissions(reasoning_engine):
    """Retrieved chunks can provide citations when the model omits them."""
    context = {
        "context_confidence": 0.86,
        "chunks": [
            {
                "id": "chunk-1",
                "source_url": "https://example.com/source",
                "content": "Evidence text for the scenario.",
            }
        ],
    }

    assert reasoning_engine._context_citations(context) == [
        "Source: cache:chunk-1 | https://example.com/source"
    ]
    assert "[Source: cache:chunk-1 | https://example.com/source]" in reasoning_engine._compact_context(context)


def test_source_citations_json_field_is_normalized(reasoning_engine):
    """Model-provided source_citations survive janitor normalization."""
    data = {
        "title": "Test",
        "summary": "Test",
        "description": "Test description",
        "time_step": 0,
        "risks": [{"description": "Risk", "severity": "Low", "likelihood": "Low"}],
        "alternatives": [],
        "source_citations": ["Source: cache:id123 | https://example.com/page"],
    }

    clean_data = reasoning_engine._janitor_fix_data(data)
    assert clean_data["source_citations"] == [
        "Source: cache:id123 | https://example.com/page"
    ]


def test_grounding_enforcement_downgrades_uncovered_high_confidence(reasoning_engine):
    clean_data = {
        "source_citations": ["Source: cache:abc | https://example.com"],
        "citation_coverage": 0.2,
        "confidence_score": 0.88,
        "speculative": False,
    }

    reasoning_engine._enforce_grounded_confidence(clean_data)

    assert clean_data["confidence_score"] <= 0.49
    assert clean_data["speculative"] is True


def test_grounding_enforcement_preserves_well_grounded_confidence(reasoning_engine):
    clean_data = {
        "source_citations": ["Source: cache:abc | https://example.com"],
        "citation_coverage": 0.9,
        "confidence_score": 0.82,
        "speculative": False,
    }

    reasoning_engine._enforce_grounded_confidence(clean_data)

    assert clean_data["confidence_score"] == 0.82
    assert clean_data["speculative"] is False


def test_rerank_citations_by_quality_orders_best_first(reasoning_engine):
    clean_data = {
        "source_citations": [
            "Source: cache:low | https://low.example.com",
            "Source: cache:high | https://high.example.com",
        ],
        "citation_provenance": [
            {
                "source_label": "Source: cache:low | https://low.example.com",
                "citation_quality_score": 0.25,
            },
            {
                "source_label": "Source: cache:high | https://high.example.com",
                "citation_quality_score": 0.92,
            },
        ],
    }

    reasoning_engine._rerank_citations_by_quality(clean_data)

    assert clean_data["citation_provenance"][0]["citation_quality_score"] == 0.92
    assert clean_data["source_citations"][0] == "Source: cache:high | https://high.example.com"
