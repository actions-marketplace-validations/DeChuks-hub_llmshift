# LLMShift plugin for Claude Code

Move a codebase off a retiring AI model and prove the replacement works on your real traffic first. The plugin bundles the [LLMShift](https://github.com/DeChuks-hub/llmshift) MCP server (run with `uvx`, nothing to install first) and a `migrate-models` skill that walks Claude through scan, snapshot, compare, fix and upgrade.

```bash
claude plugin marketplace add DeChuks-hub/llmshift
claude plugin install llmshift@llmshift
```

Then ask Claude, for example: "Which AI models does this repo call, and when do they retire?" or "Move us off gpt-4o and show me what changes." Run the skill yourself with `/llmshift:migrate-models`.

The MCP tools use the same API keys as the CLI (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...). `scan` and `models` need no key. Requires [uv](https://docs.astral.sh/uv/) for `uvx`.
