# Development Rules for SimEngine

## ⚠️ CRITICAL RULE #1: NO INCREMENTAL CHANGE LOOPS

**The Problem:**
Making one change → running tests → seeing error → making another change → running tests again (repeat).
This wastes hours and causes frustration.

**The Solution:**
1. **ANALYZE FIRST** - Read entire error output/problem completely. Do not skip any details.
2. **LIST ALL ISSUES** - Create comprehensive list of EVERY problem found (even if 20+ items).
3. **IDENTIFY ROOT CAUSES** - Connect each issue to specific files and code locations.
4. **FIX EVERYTHING AT ONCE** - Make ALL changes to ALL files needed before running ANY tests.
5. **RUN TESTS ONCE** - Only run tests AFTER all fixes are applied.

**NO EXCEPTIONS** - Never make a change, run test, see error, make another change. That is THE LOOP TO AVOID.

---

## 📋 Template for Complex Debugging Requests

When reporting bugs or test failures, include this in your request:

```
IMPORTANT: Follow DEVELOPMENT_RULES.md - NO INCREMENTAL CHANGE LOOP

Problem: [describe issue]
Error output: [paste full error]
Expected behavior: [what should happen]

Rules to follow:
1. Analyze everything first
2. Fix all issues together
3. Test once
```

---

## 🎯 General Development Principles

### Testing
- All tests must pass before committing
- Mock all external dependencies (DB, API calls, file I/O)
- Tests should run in < 10 seconds for the full suite
- Use proper fixtures to avoid test pollution

### Code Quality
- Follow PEP 8 for Python code
- Use type hints for all function signatures
- Document complex logic with inline comments
- Keep functions focused (single responsibility)

### Async Best Practices
- Use `AsyncMock` for async methods
- Use `MagicMock` for sync methods
- Never block the event loop with sync calls in async functions
- Properly await all coroutines

### Error Handling
- Always include context in error messages
- Log errors with appropriate severity levels
- Provide actionable error messages to users

---

## 🔧 Project-Specific Guidelines

### SimulationEngine
- All external calls must be mockable
- Time steps are immutable once created
- Branch nodes must lock their parent
- Terminal states stop simulation progression

### ContextBuilder (Deep RAG)
- All scraping must be fault-tolerant
- Fallback to keyword search if vector search fails
- Chunk size: 400-700 characters
- Always verify URLs before scraping

### ReasoningEngine
- Retry logic: 3 attempts with exponential backoff
- Validate all LLM responses before parsing
- Citations must be verified against context
- Temperature range: 0.5-0.8

---

## 📝 Commit Guidelines

- Descriptive commit messages (imperative mood)
- Reference issue numbers when applicable
- Keep commits atomic (one logical change per commit)
- Run tests before committing

---

**Last Updated:** 2026-01-20
**Enforced By:** All developers and AI assistants working on this project
