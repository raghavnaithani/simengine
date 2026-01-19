# Instructions for GitHub Copilot

## 🎯 Primary Directive

**NEVER enter incremental change loops.** When debugging or fixing issues:

1. **Analyze COMPLETELY** - Read all errors, all related files, all context
2. **Fix ALL issues AT ONCE** - Make every necessary change before testing
3. **Test ONCE** - Only run tests after all fixes are applied

## Quick Reference Template

When user reports errors, follow this pattern:

```
[Read full error log]
[Identify ALL issues - list them]
[Read ALL related files]
[Make ALL fixes in parallel]
[Run tests ONCE]
```

## Project Context

- **Language:** Python 3.12
- **Framework:** FastAPI (backend), MongoDB (database)
- **Testing:** pytest with AsyncMock/MagicMock
- **Key modules:** SimulationEngine, ContextBuilder, ReasoningEngine

## Critical Rules

1. All external dependencies must be mocked in tests
2. Use `AsyncMock` for async methods, `MagicMock` for sync
3. Patch at instance level when possible (`patch.object()`)
4. Always provide complete context in error analysis
5. Batch independent operations together

## See Also

- [DEVELOPMENT_RULES.md](../DEVELOPMENT_RULES.md) - Complete development guidelines
- [.copilot-rules](../.copilot-rules) - Copilot-specific preferences

---

**Bottom Line:** Think completely, fix completely, test once.
