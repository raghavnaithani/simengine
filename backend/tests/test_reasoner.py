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
    [CITATION:valid_citation]
    """
    result = reasoning_engine._extract_and_clean_json(raw_text)
    assert result["key"] == "value"
    assert "citations" in result
    assert result["citations"] == ["valid_citation"]

def test_invalid_citation(reasoning_engine):
    raw_text = """
    {
        "key": "value"
    }
    [CITATION:invalid!citation]
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
    raw_text = "{" + ",".join([f"\"key{i}\": \"value{i}\"" for i in range(1000)]) + "} [CITATION:large_test]"
    result = reasoning_engine._extract_and_clean_json(raw_text)
    assert len(result) == 1001  # 1000 keys + citations
    assert "citations" in result
    assert result["citations"] == ["large_test"]


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
