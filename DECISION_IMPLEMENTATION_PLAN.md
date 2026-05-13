# Decision Graph Simulator Implementation Plan

## End Result
Ship a real-grade decision graph system that produces rich branching outcomes, grounded citations, mandatory risks, and clear confidence signals across the backend, persistence layer, and frontend UI. The final system must behave like a decision simulator, not a chatbot: every simulation should yield a branching graph with distinct options, meaningful outcomes, provenance-backed claims, and visible uncertainty.

## Non-Negotiable Standards
- Local-first execution remains the default.
- Every DecisionNode must contain valid JSON, risks, alternatives, source citations, and confidence metadata.
- Branches must be meaningfully different, not duplicated text with minor wording changes.
- Every node must have real substance: at least 3-6 logical lines or roughly 30-40 words of useful content, even on fallback.
- Fallback content must be clearly marked speculative or partial, never disguised as grounded output, and it must still meet the minimum content depth bar.
- Every implementation step must be validated before moving to the next step.
- No step is considered done until it passes its focused test and its integration check.

## Scope Boundary
This phase is about decision-node quality and branching correctness first. Scraping, Deep RAG finalization, and broader ingestion hardening stay secondary until the node output is consistently rich across multiple branches and multiple runs.

## Current Baseline To Verify First
These checks define the current state before any implementation work:
1. Start a fresh simulation with a real backend process.
2. Confirm the live job completes or fails with a visible reason, not a silent hang.
3. Inspect the persisted `model_responses` document for the fresh job.
4. Confirm whether the new diagnostics exist in Mongo:
   - `attempt_trace`
   - `attempts_used`
   - raw response samples
   - quality issues or failure reasons
5. Confirm the graph returned by `/graph` has more than a shallow two-node structure for normal runs.
6. Confirm the frontend renders the actual graph payload and not repeated generic placeholders.

## Implementation Methodology
Work in narrow slices. For each slice:
1. Identify the single code path that controls the behavior.
2. Write or update the smallest focused test that can fail for the current defect.
3. Implement the fix with the least structural change needed.
4. Run the focused test immediately.
5. Run the integration or full-flow test for that slice.
6. Only then move to the next slice.

If a slice fails, repair that same slice before widening scope.

## Task Order

### Phase 1: Live Status and Output Contract
Goal: prove the active backend is using the intended code and emitting the expected persisted shape.

Tasks:
1. Verify the backend process is the updated one.
2. Run one fresh simulation and record the raw API/job output.
3. Inspect Mongo for the exact `model_responses` schema.
4. Compare the persisted record against the intended diagnostics contract.
5. Ensure the logging path cannot crash on non-ASCII output during token-budget or parse diagnostics.
6. Persist the same diagnostics on both success and failure paths: `raw`, `attempt_trace`, `attempts_used`, `prompt`, and explicit failure reasons.
7. Treat a completed-but-shallow run as a failure of content quality, not a successful implementation.
8. Require `/metrics` to remain observable even when Mongo is unavailable, with a basic health response instead of a hang or crash.

Tests:
- Start simulation endpoint smoke test.
- Job polling completion test.
- Mongo schema inspection for `model_responses`.
- Backend log inspection for model-call retries and parse failures.
- Explicit logger-regression test for Unicode-safe diagnostics emission.
- Persistence regression test for audit fields on both success and failure.

Completion criteria:
- A fresh simulation produces a completed or explainably failed job.
- Mongo contains the expected diagnostic fields for the active code path.
- The failure mode, if any, is now observable instead of ambiguous.
- Backend logs are readable end-to-end for a fresh run, without truncation caused by logging itself.
- A shallow node output is not mistaken for success simply because the job completed.

### Phase 2: Decision Node Quality
Goal: make node output real, rich, and distinct.

Tasks:
1. Strengthen prompt instructions for structured rich output.
2. Ensure each node produces at least 3-6 logical lines or about 30-40 words of usable content.
3. Ensure `alternatives` are materially different and action-specific.
4. Ensure risks are concrete, scenario-aware, and not generic filler.
5. Preserve citations and speculative flags when grounding is weak.
6. Keep node titles, summaries, and descriptions specific to the branch context.
7. Remove fallback normalization that hides weakness behind generic summary/description text.
8. Replace the placeholder `General uncertainty.` risk with 2-3 scenario-specific risks that include severity and likelihood.
9. Reject outputs with empty `alternatives`, duplicated actions, or duplicate descriptions even if the JSON schema is valid.
10. Mark low-grounding nodes as `speculative=true` whenever the content is fallback-like, partial, or only weakly cited.
11. Require that fallback content still exceed the minimum depth bar; a short fallback line is not acceptable.
12. Make branch-specific context change the wording and substance of downstream nodes, not just the title.

Tests:
- Unit test for DecisionNode schema validation.
- Prompt-output regression test with mocked Ollama response.
- Quality-gate test for minimal/empty alternatives.
- Content-depth test for minimum line/word count per node.
- Citation enforcement test.
- Fallback-marking regression test for partial outputs.
- Risk-specificity test that rejects generic placeholder risks.

Completion criteria:
- Nodes contain distinct alternatives and scenario-specific risks.
- Nodes consistently exceed the minimum content depth bar.
- Generic fallback text is only used when explicitly marked partial or speculative, and it never collapses into a one-line pass.
- Fallback nodes remain substantive and visibly uncertain.
- Valid JSON alone is never sufficient if the node is still shallow or generic.

### Phase 3: Branching Graph Fidelity
Goal: ensure branching is rich, visible, and deterministic enough to audit.

Tasks:
1. Verify branch expansion uses the correct frontier logic.
2. Ensure each branch selection produces a different follow-on context.
3. Prevent identical child nodes from repeated branch choices.
4. Persist parent-child edges with clear action labels.
5. Ensure graph retrieval exposes the full branching structure.
6. Verify branching output still remains content-rich at every depth, not just at the root.
7. Ensure `simulate_steps > 1` actually produces multi-step expansion instead of a shallow root-plus-one-child chain.
8. Make edge actions semantically meaningful branch labels, not generic placeholders such as `Time step 1`.
9. Verify that branch choices affect the underlying prompt/context enough to change downstream content, not only node metadata.
10. Reject a graph as complete if it has correct edges but still collapses to one node per timestep with no real branching breadth.

Tests:
- Structural branching test.
- Edge creation and parent linkage test.
- Duplicate-branch prevention test.
- `/graph` payload integrity test.
- Branch breadth test requiring >1 node at a non-root timestep.
- Branch-divergence test comparing downstream node content across sibling branches.

Completion criteria:
- Branching graphs expand beyond a shallow chain for normal scenarios.
- Different selected options create visibly different downstream nodes.
- Edges are present, meaningful, and sufficient but not mistaken for real branching unless breadth is also present.

### Phase 4: Frontend Truthfulness
Goal: make the UI accurately reflect the simulator state.

Tasks:
1. Render node confidence, citations, and speculative state clearly.
2. Show branch options as distinct choices, not repeated labels.
3. Display failure/fallback states explicitly.
4. Ensure the graph view updates from real API data.
5. Keep the interface readable under longer rich outputs.
6. Do not render placeholders when the API returns a real node payload, even if the payload is shallow or partial.
7. Surface branch breadth and node depth signals so the UI makes the graph quality obvious instead of hiding it.
8. Make fallback/speculative nodes visually distinct from grounded nodes.

Tests:
- Frontend data-binding smoke test.
- UI rendering test with mock graph variants.
- Manual end-to-end visual verification.
- API-vs-UI payload equivalence test.
- Fallback/speculative visualization check.

Completion criteria:
- The UI shows the actual branching graph and not placeholders.
- User-selected options visibly influence the next view.
- Confidence/citation/speculative state is visible and matches the API, even on shallow or fallback nodes.

### Phase 5: Harness Hardening
Goal: make the verification suite prove the system in full flow.

Tasks:
1. Update existing backend tests to cover the final schema contract.
2. Update harness tests to cover rich branching and output quality.
3. Add a full-flow scenario that exercises start, branch, graph retrieval, and quality review.
4. Add a regression test for the known shallow-graph failure mode.
5. Add a regression test for missing diagnostics in persistence.
6. Add a regression test that fails when a node is only a thin fallback line instead of substantive content.
7. Add assertions that a job can complete while still failing content-quality checks, so the harness does not confuse execution success with implementation success.
8. Add assertions for `attempt_trace` / `attempts_used` on both success and failure jobs.
9. Add branching-breadth assertions that fail if the graph is only a root-plus-single-child chain.
10. Add risk and citation quality assertions that reject generic placeholder risk text and mocked-looking citation-only grounding.

Tests:
- `backend/tests/test_simulation.py`
- `test_main_harness/test_full_feature_verification.py`
- `test_main_harness/test_full_security_edge_cases.py`
- New full-flow regression scenario test.
- New graph-breadth regression test.
- New fallback-depth/speculation regression test.

Completion criteria:
- The harness catches regressions in both structure and content.
- The harness rejects short fallback outputs that do not meet the minimum content depth bar.
- Full flow can be re-run repeatedly with stable, inspectable results.
- The harness distinguishes valid execution from valid decision-quality output.

## Execution Rules
- Never merge a patch without a focused validation step.
- Never advance to the next task if the current task’s test fails.
- Never accept generic output as final if the guide requires rich branching and grounded claims.
- Record each validation result in the implementation notes or session log.
- Prefer root-cause fixes over surface patches.

## Deliverables At Finish
- Rich branching decision graph behavior.
- Valid DecisionNode output with risks, alternatives, citations, and confidence.
- Each node has substantive output depth, not just schema compliance.
- Observable persistence diagnostics for each generation attempt.
- Updated frontend rendering of the real graph.
- Passing full-flow test harness that covers the major paths, edge cases, and content-depth checks.

## Open Verification Items
- Whether the live backend is still using an older code path for persistence.
- Whether the prompt should be tuned further or the fallback path hardened first.
- Whether current shallow graphs are caused primarily by model content quality or by downstream pruning.
