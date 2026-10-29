---
name: migrate-models
description: Move a codebase off a retiring or deprecated AI model and prove the replacement works before switching, with LLMShift. Use when the user mentions a model deprecation or retirement notice, a shutdown date, which models a repository calls, upgrading GPT-4o, GPT-5, Claude Sonnet or Gemini models, testing a new model on real traffic, or moving tests off the OpenAI Evals platform.
---

# Move off a retiring AI model with LLMShift

The LLMShift MCP tools do the work: `llmshift_scan`, `llmshift_models`, `llmshift_snapshot`, `llmshift_compare`, `llmshift_fix`, `llmshift_pick`, `llmshift_safety`, `llmshift_check`, `llmshift_upgrade` and `llmshift_doctor`. Each tool says whether it calls models (which costs money) or writes files; ask the user before either.

1. **Find what has to move.** Call `llmshift_scan` on the project. For each retired or retiring model, report the shutdown date, the replacement and the files and lines that call it.
2. **Turn real traffic into tests.** If the user has usage logs (OpenAI, Anthropic or Gemini logs, LangSmith or Langfuse exports, CSV), call `llmshift_snapshot` to make a test set from them. If their tests live on the OpenAI Evals platform, tell them to run `llmshift import openai-evals --api --all` in a terminal while the Evals API is still up (OpenAI shuts it down on 30 November 2026).
3. **Check the replacement.** Call `llmshift_compare` with the replacement model and summarise what changed: broken formats, wrong facts, different tone, different tool calls and settings the new model rejects.
4. **Close the gaps.** Call `llmshift_fix` for suggested prompt changes. It never edits code; it says where each prompt lives.
5. **Switch.** Call `llmshift_upgrade` to show the change, and with `apply` only after the user agrees. Offer to open the pull request with the test results.
6. **Keep it checked.** Suggest `llmshift init --github`, which adds the scan and pull request checks to CI.

Never call the replacement safe without a `llmshift_compare` run on the user's own tests. If `llmshift_doctor` shows no API key for a provider, say which variable to set instead of guessing.
