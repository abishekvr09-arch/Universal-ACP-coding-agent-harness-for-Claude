# AGENTS.md — how to work in this repo

This harness's edge is **trust**: every behavior verified, surface minimal enough to
audit. Be lazy about the solution, never about understanding or the security floor.

## Before writing code, stop at the first rung that holds
1. Does this need to exist? → no: skip it (YAGNI)
2. Already in this codebase? → reuse it, don't rewrite
3. Stdlib does it? → use it
4. Native platform / SDK feature? → use it (e.g. ACP already negotiates `PROTOCOL_VERSION`)
5. Installed dependency? → use it
6. One line? → one line
7. Only then: the minimum that works

Read fully first. A small diff you don't understand is laziness dressed up as efficiency.

## The floor — never cut for brevity
- The **tool_result invariant**: every `tool_use` gets exactly one `tool_result`
  (denial / cancel / error included). This is the most common harness bug; the loop is built around it.
- Approval gates · persist-before-append/execute (fail-closed) · MCP failure isolation + name sanitization · cache discipline (Law 1).
- Trust-boundary validation · cancellation · teardown.
- Per-turn deepcopy isolation (hooks/injectors can't alias history) · crash/restart tool_result reconciliation · tool-timeout liveness (a non-cooperative hang can't wedge the loop).

If a simplification touches these it isn't lazy, it's negligent. Stop.

## Verify, don't assert (this repo's hard-won rule)
- Don't quote a timing or pass/skip count you haven't measured **this session**.
- Reading code and asserting its runtime behavior is itself an unverified claim — run a probe.
- When two measurements conflict, find the reconciling fact; don't declare one fake.
- For existence/content of an external source, use the unmediated channel (`curl`/`git`),
  not a summarizer — and keep a known-good control.
- One runnable check behind any non-trivial logic — an assert or one small test, no new framework.

## Output
Code first. ≤3 lines of prose after. If the explanation is longer than the code, delete the explanation.
Exception: **verification provenance** (which clean-channel read established a fact) is product, not prose — keep it.
