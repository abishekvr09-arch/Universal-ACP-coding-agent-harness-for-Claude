# claude-harness-acp

A **universal ACP coding agent** — a focused, Claude-first tool-calling loop that **any
[ACP](https://agentclientprotocol.com)-capable host can drive**, and that you can also embed
directly as a Python library. Not a multi-channel assistant; a dev/coding agent core with a
clean plugin API for **tools**, **providers**, and **hooks**.

The universality lives at the protocol: the agent is a standard ACP agent with **zero host
coupling**, so any ACP client (editors like Zed, or a host plugin) drives the *same* artifact.
OpenClaw is the first flagship integration (a thin adapter in [`adapters/openclaw/`](adapters/openclaw/)),
but it is one consumer, not a dependency.

> Status: **Phase 3 feature-complete** — core loop, Claude provider, 5 tools, hooks, SQLite
> store, tool-call repair, compression, ACP server, CLI, and MCP client (**155 tests**).
> The OpenClaw adapter is built and **wire-verified** (TS ↔ Python over ACP, `tsc`-green vs the
> real SDKs; a deterministic $0 fake round-trip #1–#7 covers streaming, the tool_result
> invariant, model propagation, trust-floor rejection, and cancellation across the wire). Phase 4
> packaging is **built + verified offline** (locate-not-bundle); the real-API smoke gate (held) and
> ClawHub/PyPI publish remain.
>
> **Hardened (adversarial):** per-turn deepcopy isolation, crash-resume reconciliation of
> interrupted tool calls, fail-closed persistence, store-backed recovery (CLI `--session-id`
> + ACP `session/load`), and the tool_result invariant under timeout / cancel / denial /
> exception / SIGINT / provider-disconnect (+ a `tool_timeout` backstop) — all probe/test-proven.

## Why it exists

Most agent loops get three things subtly wrong. This one is built around them — that's the trust story:

1. **One `tool_result` per `tool_use` — always.** Denial, cancellation, and tool exceptions each
   emit a synthetic error result. Skipping one 400s the next API call. The loop makes this
   structural, not a thing you remember.
2. **Prompt caching is sacred.** Tool *declarations* are frozen per conversation; the system prompt
   is byte-stable and restored from the session store; cache breakpoints slide over the last 3
   messages. Context only ever mutates via compression at a session boundary.
3. **Tool-call repair is first-class.** Models emit tool calls as prose (bracket / Harmony /
   XML-ish) and with wrongly-typed args. Repair + coercion run in the loop before approval hooks
   see the call.

## Two ways to use it

### 1. Drive it over ACP — any host (the universal path)

The agent serves the Agent Client Protocol (JSON-RPC over stdio). Install the package, then point
any ACP client at the `harness-acp` command — it implements the standard ACP methods (`initialize`,
`session/new`, `session/prompt`, `session/cancel`, `session/close`):

```bash
# Install from source (this repo) — works today:
git clone <this-repo> && cd harness
pip install -e ".[acp]"            # provides the `harness-acp` command
# (once published to PyPI this becomes a one-liner: pip install claude-harness-acp)

export ANTHROPIC_API_KEY=sk-ant-…  # bring your own key
harness-acp                        # serves ACP over stdio; an ACP client spawns this
```

- **ACP-native editors** (Zed, other ACP hosts) connect directly — no adapter, no glue. Configure
  your editor's ACP agent command as `harness-acp`.
- **OpenClaw** drives it through the flagship adapter (`adapters/openclaw/`), which spawns the same
  `harness-acp` and translates OpenClaw's `AgentHarness` contract ↔ ACP. See its
  [README](adapters/openclaw/README.md).
- The per-attempt **model** rides in `session/new`'s `_meta` (the stable ACP channel); the agent
  validates it against its supported models and **fails closed** on anything unadvertised.

There is no per-host plugin to maintain: the universality is the protocol. A host needs a thin
adapter *only* if it has its own plugin model (like OpenClaw); ACP-native hosts need nothing.

### 2. Embed it as a Python library

```python
from harness.core.loop import Agent, AgentConfig
from harness.providers import ClaudeProvider
from harness.tools import default_tools
from harness.hooks import from_tools
from harness.session import SessionStore

tools = default_tools()                       # read, edit, bash, glob, grep
store = SessionStore("harness.db")            # SQLite + WAL
store.create_session("s1", "You are a coding agent.", model="claude-opus-4-8")

agent = Agent(AgentConfig(
    provider=ClaudeProvider(model="claude-opus-4-8"),   # needs ANTHROPIC_API_KEY
    tools=tools,
    system="You are a coding agent.",
    hooks=[from_tools(tools, approver=lambda name, args: input(f"run {name}? [y/N] ") == "y")],
    persist=store.persist_fn("s1"),
))

messages = [{"role": "user", "content": [{"type": "text", "text": "bump VERSION to 0.1.0"}]}]
final = agent.run(messages)
print(final.content[0].text)
```

Install for development (editable, with dev + acp extras):

```bash
python -m pip install -e ".[dev,acp]"
```

## Architecture

Three extension seams around a frozen loop. See [CLAUDE.md](CLAUDE.md) for the full design (the
three laws, the loop contract, type contracts, and the research it's distilled from).

```
src/harness/
  core/      types, loop, budget, repair        # the narrow waist
  providers/ claude, cache breakpoints          # model backends
  tools/     read, edit, bash, glob, grep       # capability
  hooks/     approval, cost                      # middleware
  session/   SQLite store (WAL)                  # persistence
  acp/       ACP server (HarnessAgent)           # interop — ANY ACP host drives this
  mcp/       MCP client                          # interop — consume external tools
  testing.py scripted fake provider             # offline test affordance (shared)
adapters/openclaw/   TS plugin → spawns harness-acp over ACP   # first flagship adapter
```

## Test

```bash
python -m pytest -q
```

All tests run offline against a scripted provider — no key, no network. The invariant suite
(`tests/test_invariant.py`) is the one that must never regress.

## License

MIT
