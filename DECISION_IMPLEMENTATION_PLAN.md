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

Phase 5 verification map (Integrated with test_main_harness)

**Integration Note**: Phase 5 tasks update and extend the existing test_main_harness (located in `test_main_harness/`). Each task maps to one or more existing test files and adds new test cases or assertions. All logging must include DATE/TIME stamps in ISO 8601 format and be appended to `desicion_imp completion_rep`.

1. Update existing backend tests to cover the final schema contract.
   - Relevant phases: Phase 1 output contract and persistence diagnostics, Phase 2 node schema and quality gates, Phase 3 graph structure, Phase 4 UI payload truthfulness.
   - Existing implementation anchors: `backend/app/main.py` job, graph, and metrics endpoints; `backend/app/engines/reasoner.py` diagnostics and fallback path; `backend/app/engines/simulation.py` graph payload.
   - Existing test files to update: `backend/tests/test_simulation.py`, `test_main_harness/test_full_api_engine_integration.py` (enhance schema assertions).
   - Existing tests/logs to read first: `backend/tests/test_simulation.py`, `backend/tests/test_branch_idempotency.py`, Phase 1/Phase 2 notes in `desicion_imp completion_rep`, and the harness README.
   - What Phase 5 must verify: the final schema still includes the diagnostics contract (`raw`, `attempt_trace`, `attempts_used`, `prompt`, explicit failure reasons) and still exposes the node fields that Phase 4 renders (`confidence_score`, `speculative`, `source_citations`, `alternatives`, `risks`, depth/breadth signals). Schema validation must pass on both live jobs and mock runs.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Record the exact job id/session id used for schema verification, the test method name, the fields verified, pass/fail result, and append with **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

2. Update harness tests to cover rich branching and output quality.
   - Relevant phases: Phase 2 rich-content enforcement, Phase 3 branch divergence, Phase 4 truthful rendering of quality signals.
   - Existing implementation anchors: `backend/app/engines/reasoner.py` quality gates and fallback synthesis, `backend/app/engines/simulation.py` branch expansion, `test_main_harness/test_full_feature_verification.py`, `test_main_harness/test_full_security_edge_cases.py`.
   - Existing test files to update: `test_main_harness/test_full_feature_verification.py` (add branching/depth assertions), `test_main_harness/test_full_security_edge_cases.py` (add content quality checks).
   - Existing tests/logs to read first: Phase 2 schema/quality tests, Phase 3 structural branching tests, `scripts/phase4_complete_test.js`, and Phase 4 verification log entries in `desicion_imp completion_rep`.
   - What Phase 5 must verify: the harness should fail shallow content (< 30-40 words per node), duplicated alternatives, placeholder risks (e.g., "General uncertainty."), and misleading fallback output even when JSON schema is technically valid. Assertions must use SecurityTraceLogger or similar patterns to emit JSONL traces of what failed.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** For each test added/updated, record the test method name, the assertion triggered, the failing node id (if applicable), the violated rule, root cause (depth, breadth, citations, fallback/speculative state), and append **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

3. Add a full-flow scenario that exercises start, branch, graph retrieval, and quality review.
   - Relevant phases: Phase 1 job lifecycle and observability, Phase 2 node quality, Phase 3 branching flow, Phase 4 graph payload rendering.
   - Existing implementation anchors: `test_main_harness/test_full_service_e2e.py` (add full-flow test class), the live simulation endpoints `/simulate/start`, `/jobs/{job_id}`, `/graph`, `/jobs/{job_id}/logs`.
   - Existing test files to update: `test_main_harness/test_full_service_e2e.py` (new test method or class for full-flow scenario).
   - Existing tests/logs to read first: Phase 3 end-to-end smoke validation notes, Phase 4 test report JSON, Phase 1 notes in `desicion_imp completion_rep`, and the harness README workflow section.
   - What Phase 5 must verify: one end-to-end scenario should cover the whole path from `/simulate/start` through at least one `/jobs/{job_id}/branch` action, `/graph` retrieval, and final `/jobs/{job_id}/quality` check. The test must emit a complete execution timeline and prove the harness can re-run the same scenario repeatably with stable results.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Capture the full execution timeline with timestamps for each step (start, branch, graph, review), include the job id, final execution status (success/failure), final quality status (pass/fail), and append the summary with **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

4. Add a regression test for the known shallow-graph failure mode.
   - Relevant phases: Phase 3 shallow-graph rejection and breadth test, Phase 4 topology display.
   - Existing implementation anchors: `backend/app/engines/simulation.py` shallow-graph rejection path, Phase 3 branch breadth test, Phase 4 D#/B# enrichment in `frontend/lib/api.ts`.
   - Existing test files to update: `test_main_harness/test_full_feature_verification.py` or new test in `test_main_harness/test_full_security_edge_cases.py`.
   - Existing tests/logs to read first: Phase 3 branch breadth test notes in `desicion_imp completion_rep`, Phase 3 smoke validation notes, and Phase 4 smoke test report showing shallow 2-node/1-edge graph.
   - What Phase 5 must verify: the harness should explicitly fail if the graph collapses to a root-plus-single-child chain or otherwise loses branching breadth, even if the job itself completes successfully. The test must assert that execution success ≠ quality success.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Record the graph shape that triggered the failure (node count, edge count, depth histogram, breadth distribution), the job id, whether failure was due to breadth collapse or missing branch divergence, and append **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

5. Add a regression test for missing diagnostics in persistence.
   - Relevant phases: Phase 1 live status/output contract and persistence diagnostics.
   - Existing implementation anchors: `backend/app/main.py` job and metrics endpoints, Phase 1 persistence checks, runtime log behavior in `error_log.txt` and `/jobs/{job_id}/logs`.
   - Existing test files to update: `test_main_harness/test_full_api_engine_integration.py` (enhance persistence checks for both success and failure).
   - Existing tests/logs to read first: Phase 1 notes about `attempt_trace`, `attempts_used`, raw response samples, explicit failure reasons, and the `/metrics` null-handling issue mentioned in completion report.
   - What Phase 5 must verify: both successful and failed jobs must persist the full diagnostic contract (`raw`, `attempt_trace`, `attempts_used`, `prompt`, explicit failure reasons). The harness should fail if any required field is missing or a runtime path silently omits it.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Note which persistence fields were checked, which were present vs. absent, whether the check was for success or failure jobs, test method name, and append the exact field-checklist outcome with **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

6. Add a regression test that fails when a node is only a thin fallback line instead of substantive content.
   - Relevant phases: Phase 2 node quality and fallback normalization, Phase 4 fallback visualization.
   - Existing implementation anchors: Phase 2 `_fallback_description()`, `_fallback_summary()`, `_janitor_fix_data()`, quality-gate minimum depth rules, Phase 4 unverified/speculative banners.
   - Existing test files to update: `test_main_harness/test_full_feature_verification.py` (add content-depth assertion).
   - Existing tests/logs to read first: Phase 2 quality-gate diagnostics in `desicion_imp completion_rep`, Phase 2 generated node artifact, and Phase 4 fallback/speculative visualization results.
   - What Phase 5 must verify: a fallback node must still have substantive content depth (≥ 30-40 words or 3-6 logical lines) and explicit speculative marking (`speculative=true`). A one-line placeholder or generic fallback summary must fail the harness. Depth validation should be independent of whether the node is speculative.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Save the offending node text sample, word/line count, speculative flag value, the assertion that failed, the node id, and append with **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

7. Add assertions that a job can complete while still failing content-quality checks, so the harness does not confuse execution success with implementation success.
   - Relevant phases: Phase 1 job-completion observability, Phase 2 quality gate rejection, Phase 3 graph completion, Phase 4 truthful display of shallow/fallback state.
   - Existing implementation anchors: Phase 1 completed-but-shallow run notes, Phase 2 quality gate failures, Phase 4 rendering of shallow/fallback nodes.
   - Existing test files to update: `test_main_harness/test_full_service_e2e.py` or `test_main_harness/test_full_security_edge_cases.py` (add separate assertions for execution success vs. quality success).
   - Existing tests/logs to read first: Phase 1 smoke job completed-but-shallow notes, Phase 2 live full-flow notes, and Phase 4 smoke report showing completed-but-shallow 2-node graph.
   - What Phase 5 must verify: the harness must treat execution success (job completed) and content-quality success (depth/richness/breadth met) as separate outcomes. A test must explicitly pass or fail each axis independently so a completed-but-poor job is never misread as a valid implementation.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Record execution status (completed/failed) and quality status (pass/fail) separately, the job id, the specific quality rules that failed (if any), and append both statuses with **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

8. Add assertions for `attempt_trace` / `attempts_used` on both success and failure jobs.
   - Relevant phases: Phase 1 output contract and diagnostic persistence, plus Phase 2 quality-gate retry behavior.
   - Existing implementation anchors: job/model response persistence path, retry behavior observed in Phase 1 logs, failure/success diagnostics in completion report.
   - Existing test files to update: `test_main_harness/test_full_api_engine_integration.py` (enhance diagnostics assertions with attempt tracking).
   - Existing tests/logs to read first: Phase 1 job/log inspection notes in `desicion_imp completion_rep`, Phase 2 retry/quality-gate diagnostics, and live `/jobs/{job_id}/logs` output samples.
   - What Phase 5 must verify: both successful and failed jobs must expose `attempt_trace` and `attempts_used` fields in the job diagnostics, so retry behavior is auditable and reproducible across runs.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Write the attempt trace summary with attempt count, final status (success/failure), any retry-triggering error reasons, the job id, the test method name, and append **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

9. Add branching-breadth assertions that fail if the graph is only a root-plus-single-child chain.
   - Relevant phases: Phase 3 branching fidelity and breadth-first expansion, Phase 4 topology display.
   - Existing implementation anchors: Phase 3 branch breadth test, `/graph` payload integrity test, Phase 4 D#/B# topology enrichment.
   - Existing test files to update: `test_main_harness/test_full_api_engine_integration.py` or `test_main_harness/test_full_feature_verification.py` (add branching-breadth assertions).
   - Existing tests/logs to read first: Phase 3 breadth test notes, Phase 3 end-to-end smoke validation, and Phase 4 smoke report/topology computation logs.
   - What Phase 5 must verify: the harness must detect when the graph has correct edges but insufficient breadth. A root-plus-single-child chain should fail explicitly, even if the job completed and the schema is valid. Breadth must be checked at each depth level.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Include the node-depth histogram, breadth counts per depth level, the exact branch level where breadth collapsed, the job id, the test method name, and append **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

10. Add risk and citation quality assertions that reject generic placeholder risk text and mocked-looking citation-only grounding.
   - Relevant phases: Phase 2 risk-specificity and citation enforcement, Phase 4 citation display.
   - Existing implementation anchors: Phase 2 risk gate logic and fallback synthesis, Phase 4 source-citation display and confidence/speculative badges.
   - Existing test files to update: `test_main_harness/test_full_feature_verification.py` (add risk/citation quality checks).
   - Existing tests/logs to read first: Phase 2 risk-specificity and citation diagnostics in `desicion_imp completion_rep`, Phase 2 generated node artifact, and Phase 4 mock rendering/fallback visualization results.
   - What Phase 5 must verify: risk entries must be scenario-specific and meaningful, not generic placeholders like `General uncertainty.` Citations must not be a substitute for substantive content; a node with many citations but shallow description must fail. Each node's risks must include severity and likelihood fields and be actionable.
   - Logging rule for this task: **[IMPORTANT: Include timestamp and date]** Capture the rejected risk/citation samples (text snippets), the node id, whether the failure was due to generic text, low citation coverage, shallow content depth, or missing risk fields, the test method name, and append **Date/Time in ISO 8601 format** to `desicion_imp completion_rep`.

Tests (Updated for test_main_harness Integration):
- `backend/tests/test_simulation.py` — Enhanced with final schema contract assertions
- `test_main_harness/test_full_feature_verification.py` — Updated with branching/depth/quality assertions
- `test_main_harness/test_full_security_edge_cases.py` — Updated with content quality edge cases
- `test_main_harness/test_full_api_engine_integration.py` — Enhanced persistence and diagnostics checks
- `test_main_harness/test_full_service_e2e.py` — New full-flow end-to-end scenario with quality review
- New in-harness regression tests: shallow-graph failure, fallback-content depth, branching-breadth, risk/citation quality
- All tests emit JSONL traces via SecurityTraceLogger or equivalent pattern to `test_main_harness/runs/latest/raw_logs/`

Completion criteria (Aligned with test_main_harness standards):
- **The harness catches regressions in both structure and content**: All 10 Phase 5 tasks result in new or enhanced test assertions in the main harness that fail when structure or content quality degrades.
- **The harness rejects short fallback outputs that do not meet the minimum content depth bar**: Tasks 6 and 10 specifically enforce ≥30-40 words or 3-6 logical lines per node, and fallback nodes must be explicitly marked speculative.
- **Full flow can be re-run repeatedly with stable, inspectable results**: Task 3's full-flow scenario must pass consistently, and all test outputs are persisted to `test_main_harness/runs/latest/raw_logs/` for inspection and reproducibility.
- **The harness distinguishes valid execution from valid decision-quality output**: Tasks 7 and 9 explicitly separate execution success (job completed) from quality success (depth/breadth/content met), preventing false positives.
- **Every Phase 5 task logs completion with DATE/TIME**: All task logging rules include ISO 8601 timestamps in `desicion_imp completion_rep` following the Phase 3 format established in earlier phases.

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
