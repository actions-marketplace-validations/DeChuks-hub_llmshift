# Changelog

## 0.1.7 (2026-10-29)

- `llmshift import openai-evals --api --save-raw` also keeps every API response exactly as read: the evals with their graders, every run with its data source and settings, the output items and the data files. They go to a dated file next to the tests (`openai-evals-raw-2026-10-29.json`), or to the file you name. Before, the import turned this data into tests and kept nothing else, so once the API shuts down on 30 November nothing would show what the original graders did. The file holds response bodies only, never the API key, and is never replaced without `--force`, which is checked before any API call. Suggested by Shuttlewright on DEV.

## 0.1.6 (2026-10-28)

- **Security:** the config search stays inside what is yours. Before, outside a git repository, the search went through every parent folder up to the drive root. A config that another user planted higher up, such as `C:\.llmshift\config.yaml` (which any user can create), could name programs that would then run as you. That is the hole git closed as CVE-2022-24765. Now the search:
  - stops at the git root, at your home directory, and at the project an MCP server was started on;
  - never reads a drive root or `/`;
  - on Linux and macOS, ignores files another user owns and says so in the warnings;
  - on Windows, outside your home directory, looks above the current folder only up to its git root.
- `llmshift calendar` writes a page for every model and every provider: the shutdown date, the replacement, the official source, the error a retired model returns and the commands that move a codebase off it, linked from the calendar's table and to each other. It also writes `llms.txt`, so AI assistants can answer "when does this model retire?" from the site. `--base-url` (the site's https:// address) adds canonical links, `sitemap.xml`, `robots.txt` and an RSS feed (`feed.xml`) of the retirements in the next year; the `calendar` workflow passes it.
- `scan` counts in words ("1 file, 1 model reference" instead of "1 files"), and its next step names the replacement of the most urgent model (`llmshift compare --to claude-sonnet-5-5`) and how to keep the check on every pull request (`llmshift init --github`).
- A Claude Code plugin: `claude plugin marketplace add DeChuks-hub/llmshift`, then `claude plugin install llmshift@llmshift`, adds the MCP server (run with `uvx`, nothing to install first) and a `migrate-models` skill that walks Claude through scan, snapshot, compare, fix and upgrade.
- pre-commit hooks: `llmshift-scan` stops a commit that calls a retired model, and `llmshift-scan-retiring` also one that calls a model retiring within 180 days.
- The README sets up the MCP server with `uvx llmshift mcp`, so nothing has to be installed first, and says how to see retiring models as code scanning alerts (`scan --format sarif`).

## 0.1.5 (2026-10-28)

- `llmshift calendar --goatcounter <code>` counts visits to the calendar site with GoatCounter, which sets no cookies, and says so in the page footer. GoatCounter's script is pinned to one version (`count.v5.js`) with its integrity hash, so browsers refuse it if it ever changes.
- Python graders now run as an ordinary module. Their source is written to a file in a temporary directory and imported by a separate Python process, so it is never handed to Python's `exec`. They are still off unless you pass `--allow-python` or set `allow_python_checks: true`. This answers an automated static-analysis advisory, GCVE-110-VULN-PYPI-2026-000499 (CWE-94). A test now fails if a call to `exec` or `eval` appears anywhere in llmshift's own source.
- **Security:** on Windows, git and taskkill now start by their full path. Windows looks in the current directory before `PATH`, so a `git.exe` inside a cloned repository could have run in place of git when you scanned it. That is the untrusted search path GitPython fixed as CVE-2023-40590. Git is found in a `PATH` folder and never in the current one.
- **Security:** the official OpenAI API is recognised by its host name. Before, any base URL containing `api.openai.com`, such as `api.openai.com.example.net`, got the official API's request shapes.
- **Security:** recordings, histories and the other files that are appended to are created readable by their owner only (`0o600`), like the temporary files, because they hold prompts and answers.
- **Changed:** a `command` provider never runs its command through a shell. A string command is split into words as a shell would split them (quotes group words), and nothing in it is expanded, piped or chained, so text in a config can no longer start a second program. On Windows the program splits the string itself. For shell features, name the shell yourself: `command: ["sh", "-c", "a | b"]`. An unclosed quote is a config error.
- Code that security scanners report is gone:
  - The subcommand modules and the OpenAI SDK's schema converters are loaded with ordinary `import` statements, not looked up from strings.
  - `OLLAMA_HOST=0.0.0.0` is recognised with `ipaddress` instead of a hard-coded address.
  - The scanner's `shell=` option is now `shell_script=`, since scanners take any `shell=` argument for a shell.
- `tests/test_security.py` fails on code that scanners report:
  - dynamic code execution;
  - shells, pickle, or unsafe YAML;
  - unverified connections;
  - weak hashes used for security;
  - hidden bidirectional text.
- Bandit reports no issues at any severity on llmshift's code. Every safe-by-design case is marked `# nosec Bxxx` with its reason:
  - errors that are ignored on purpose now use `contextlib.suppress`;
  - retry jitter and recording sampling use the operating system's random source;
  - the JUnit writer escapes XML itself instead of importing `xml.sax`;
  - the maintenance scripts fetch only `https://` addresses.
- A `security` workflow runs Bandit, `pip-audit` and those tests on every push, every pull request and every Monday. It runs Bandit at every severity on llmshift's code, and at medium and above on the tests, scripts and the vendored PyYAML. Releases wait for it before publishing.
- CodeQL (the `security-extended` queries) scans the code and the workflows, and Dependabot proposes updates to actions and dependencies each week.
- The PyPI publish action is pinned to a commit rather than a tag that could move.
- The source archive includes `SECURITY.md`.
- `SECURITY.md` covers:
  - how to report a vulnerability privately;
  - how each change is checked;
  - which features run code you supply (python graders, `python:` and `command` providers) and how to keep them safe.

## 0.1.4 (2026-10-24)

- LLMShift is in the official MCP Registry as `io.github.DeChuks-hub/llmshift`. Each release publishes `server.json` there after PyPI has it, so MCP clients can install the server as `uvx llmshift mcp`.
- The GitHub Action is listed on the GitHub Marketplace. Its description in `action.yml` is shorter, to fit the Marketplace's 125-character limit.
- The README opens with an animated demo of a real run (`docs/demo.gif`): `scan` finding retiring models, then `compare` on Claude Haiku 5.5.
- The release and retirement-calendar workflows use the Node 24 releases of GitHub's artifact and Pages actions, so their runs no longer warn that Node.js 20 is deprecated. A test keeps every workflow, example and generated workflow on them.

## 0.1.3 (2026-10-24)

- Markdown reports (`compare`, `pick`, `safety`, `fix`, `embeddings`), the pull request comments of `check`, the pull requests `upgrade` opens and the GitHub issues `watch` opens end with a linked "Generated by llmshift" line, so whoever reads them can find the tool.

## 0.1.2 (2026-10-24)

- `llmshift mcp` no longer starts a thread while it shuts down. Once the client closed its end, that could crash it on exit (exit code -11, seen once on macOS with Python 3.10) after every request had been answered.

## 0.1.1 (2026-10-24)

- `pick`: when the model you use today is also a candidate, today's cost is measured on the same tests, so its row reads "same" instead of a saving against itself. The header says where today's cost comes from (your recorded usage, or these tests), and "Keep" is recommended whenever today's model is the cheapest that passes.
- `pick` from Python and the MCP `llmshift_pick` tool read `pick.volume` the way the command line does: `100k` in `llmshift.yaml` failed there with a TypeError.
- `demo`: the simulated production logs record the token usage the simulated model actually reports, so today's cost in `pick` matches a test run of the same model.
- `safety`: the header gives the probes each category really ran. Refusals use each test case at most once and add up to 10 harmless questions with alarming words, so with 18 test cases the refusal row shows 28, not "20 probes per category".
- README links work on the PyPI page (they were relative to the repository).
- The tests pass from the source archive on PyPI (one of them needed the repository's CI workflow); CI and the release now run the test suite from the built source archive, as packagers do.

**Safety and security**

- Secrets stay out of sight: API keys with a stray newline or invisible character are cleaned or reported without the key; an empty `base_url` (an unset `${VAR}`) is an error instead of sending your requests and key to the public API; only `http(s)` URLs are used, a URL with `user:password@` is refused without echoing the password, and error messages name only the host (a webhook URL's token is never shown).
- Model output, provider errors and scanned files cannot drive your terminal: escape sequences and bidi controls are shown as text. In PR comments, `@mentions` from model answers notify no one and a prompt or diff cannot close its code block.
- Redaction covers more key formats (xAI, Groq, Perplexity, Fireworks, Hugging Face, Replicate, Stripe, GitLab, npm, temporary AWS keys), `password=`/`api-key:` values, credentials in connection strings, Basic auth headers and the live values of key environment variables.
- Alerts: SMTP logs in only over TLS; Slack and Discord messages cannot ping `@channel`/`@everyone`. MCP tools read and write only inside the project.
- GitHub: a comment by another user is never edited, a POST that may have arrived is never sent again, and a value can no longer inject extra step outputs. The action's `args` are split like shell words (quote an argument with spaces) and never expanded. The workflows (and those `init --github` writes) ask for the least permissions and keep values out of `run:` scripts.
- `upgrade --apply` first saves a file git cannot give back as it is now (no repository, untracked, uncommitted changes) as `NAME.bak` (`--no-backup` skips that), and stops before writing anything if a file was edited after the diff was made.
- Text from models and users cannot show images (tracking pixels) or disguised links in PR comments, job summaries, reports or GitHub, Discord and Teams alerts: `[ ] ( ) !` are escaped and bare URLs are not linked. Slack and Discord alerts show no link previews.
- Bidi override characters ("Trojan Source") are shown as escapes in every report (HTML, Markdown, JSON, JUnit), as they already were on the terminal.
- AWS secret access keys, which have no prefix, are redacted by what surrounds them: any spelling of "secret access key" before them, or the access key id next to them.
- One chat message can no longer freeze an app that records with `wrap()`: redaction time grows in line with the text, milliseconds for a message and under a second per megabyte of the most hostile text found. A text of `-----BEGIN PRIVATE KEY-----` lines with no END held Python for 15 s per 224 KB, and many `password=` settings in a row slowed it too.
- Redaction takes a password with symbols whole (only the start of `xK9#mP2&vL5?qR8(wN3)` was hidden) in `name=value`, JSON, YAML, XML, prose and a tool call's arguments, after a setting name of any length, to the end of a token longer than 256 characters, and an AWS secret key after a plain `key=`. A JSON value under a password or secret key is redacted whatever it holds, a passphrase with spaces too. Left alone: a logprobs entry's `token`, code (`tokens[0]`, `hashlib.sha256(pw)`), placeholders (`${DB_PASSWORD}`, an earlier `<SECRET_1>`) and the working directory (`PWD=/home/bob`).

**Correct results**

- A setting the new model rejects fails the case instead of being left out silently, because the real call gets a 400 error. This covers a temperature other than 1 and any `top_p` or `top_k` on Claude 4.7 and later, a temperature an OpenAI reasoning model refuses, an assistant prefill and a forced tool choice. `--drop-param temperature` (or `drop_params:`) tests without a setting once the code no longer sends it. `upgrade` lists every `temperature`, `top_p` and `top_k` a move to Claude 4.7 or later must remove, and `top_k` from Anthropic and Gemini logs is now kept and sent. `fix` judges prompts on the answers and names rejected settings separately, and `watch` says when they are why the pass rate fell. Answers cached by earlier versions are not reused, because they cannot say what the model refused.
- Pass-rate gates use the exact rate: 18,999 of 20,000 no longer passes a 95% gate. Case ids stay unique and stable however cases are selected, and `--limit 5` stops reading a huge test file after five cases.
- Inferred checks pass on the baseline they came from (JSON keys with odd characters, names like "Domino's Pizza", answers with no words). JSON checks reject `NaN` and `Infinity`, which most parsers refuse (a test whose own baseline has them keeps passing, and `NaN` now equals `NaN` in `json_match`).
- Log sampling keeps a fair share of every prompt template, however late it first appears. Timestamps read the same on Python 3.9 to 3.15 (offsets in hours, decimal commas, basic format, week dates, HTTP dates).
- `scan` resolves names in their own scope (a parameter no longer picks up another function's model), follows `os.getenv("X", "model")` defaults, and reads files past Windows' 260-character path limit. Skipped files are always listed.
- `--repeat`: a request that failed is counted against the pass rate but no longer makes a case "flaky".
- A judge that cannot answer (a wrong key) is reported as such instead of "0 judge calls", and `judge calibrate` keeps the last good result instead of saving an empty one.
- `fix` carries on with rule-based changes when the optimizer fails, stops when it repeats itself, and no longer lists logs or its own reports as places a prompt lives.
- Recording: calls through `with_raw_response` and `with_streaming_response` are logged; wrapping a client per request shares one shadow worker set.
- `wrap()` keeps each SDK method's name, signature and `__wrapped__`, and an async client's methods are coroutine functions, so instructor and LangSmith treat `wrap(AsyncOpenAI())` as async (instructor took it for a synchronous client). A tracer that patches `create` after `wrap()` (LangSmith's `wrap_openai`) no longer gets every call recorded and shadowed twice, and calls through the SDK's `with_raw_response` and `with_streaming_response` under it record the answer, not the HTTP response.
- A judge that answers in prose instead of JSON is read with its negations: "does not pass" is a fail and "doesn't fail" a pass (both were read the other way round), and "fails" counts as a verdict.
- A check that could not run no longer counts as the new model failing. That covers an invalid regex, one stopped after 2 seconds, and a judge that cannot be reached. The case is reported as not checked, with the reason, and left out of the pass rate and its range ("95% range 34%-100% over 2 of 3 cases"). When nothing else failed, `compare` and `check` exit 2 (inconclusive) instead of 1 and say so in their headline, `watch` warns, and `pick` recommends no candidate it could not fully check. A run in which no case could be checked has no pass rate ("n/a", not "0% below the threshold"), and `compare`, `check` and `pick` exit 2 on it.
- `scan` gives a Vertex AI id Vertex's own dates: `claude-3-haiku@20240307` retires on Vertex on 2026-08-23, not on the Anthropic API's 2026-04-20, and 5 of the 7 Claude models Vertex lists were affected. The platform an id names now wins over the rest of its file, so a Bedrock or OpenRouter id in a file that also uses AnthropicVertex is no longer read as Vertex's. Azure's spelling `gpt-35-turbo-0613` is recognised; it came back unknown, so `--fail-on retired` let it pass.
- `pick` counts Gemini's thinking tokens (an answer of 120 visible and 900 thinking tokens was costed as 120), and charges cached input at the model's cache price instead of the full input price, 1-hour cache writes at their own price (twice the input price, not 1.25 times), and a long prompt at its long-context price (Claude Haiku 5.5 costs five times as much above 100K tokens), from prices now in the price list. `compare`'s agent costs follow the same rules, and `pricing:` takes `cached_input`.
- `not_equals` compares the way `equals` does (stripped, `case_sensitive`, `numeric`): `N/A\n` passed both `equals: N/A` and `not_equals: N/A`. Imported `string_check` `eq` and `ne` compare exactly, as OpenAI does.
- Extended thinking is tested. `snapshot` keeps Claude's `thinking` and effort (Sonnet 5.5's `between_tools` too) and Gemini's `thinkingConfig`; only OpenAI's reasoning effort was kept. The Anthropic and Gemini providers send them in the form each model takes:
  - a budget becomes adaptive thinking on Claude 4.7 and later, and adaptive thinking becomes a budget before 4.6;
  - `xhigh` effort becomes `high` on 4.6, which has no `xhigh`, and Opus 4.5 gets its effort (low, medium or high, with the `effort-2025-11-24` beta header) instead of none;
  - turning thinking off becomes `between_tools` on Sonnet 5.5, and is left out on models that always think (Opus 5.5, Fable, Mythos).

  Each such case fails as a rejected setting, with what to send instead, because the real call would get a 400 error. A `temperature` other than 1, a `top_k` or a low `top_p` beside thinking is left out and reported the same way, never the thinking. A value a model refuses is remembered for that value only, so one bad case changes nothing for the others, and a `thinkingConfig` Gemini refused is not sent again.
- The "What changed" summary and the Markdown report quote a regex whole: `^(Yes|No)$` was shown as `/^$/`.
- `facts` reads a time and a sign as people write them: `2:30 PM` against `14:30` failed with "no mention of PM", and `-5` against `minus 5` (or `negative 5`) failed too. "20 minus 5" stays a subtraction.
- A replacement that is itself retired with no successor is no longer recommended. `labs-leanstral-2603` pointed at `labs-leanstral-1-5`, retired on 2026-09-30; the message now says so.
- Providers: `python:` functions honour `timeout` when one is set (and no more than four calls that ran past it keep running), and their cache key covers the project modules they import; `command` providers honour it on Windows and notice an edited script. Compressed and mislabelled responses are read, JSON cut off in transit is fetched again, an HTML error page is summed up in one line, `OLLAMA_HOST` is read the way Ollama reads it, and a mistyped provider gets a "did you mean".
- Settings: values of the wrong type (`timeout: abc`, `headers` as a list, a `base_url` without `https://`) are reported where they are read; unknown or misspelt settings and severity levels are reported with a suggestion, and `severity: {tone: off}` works (YAML reads `off` as false).

**Fewer surprises**

- Files saved as UTF-16 (Windows PowerShell's `>`) are read, and a file that is not text is named in the error.
- `init` takes the baseline from the model your code calls most, and its listed replacement as the candidate, instead of writing a model that is itself retiring.
- `--config` and `--no-color` work after the subcommand too; `llmshift --help | head -1` exits quietly; `doctor` exits 1 when it finds a problem; two reports written in the same second no longer overwrite each other.
- `demo --dir new-folder` works on Windows with Python 3.9 before 3.9.23: it created the project, then stopped with "config file not found".
- The action, the example workflows and those `init --github` writes use the Node 24 releases of GitHub's own actions, so runs no longer warn that Node.js 20 is deprecated.
- An empty model (`--to "$CANDIDATE"` with the variable unset) is refused by every command and MCP tool instead of quietly testing the configured model. `--temperature` must be 0 to 2 and `--max-tokens` at least 1, checked before any request; `embeddings` checks the corpus before it starts.
- `scan` is faster on large repositories: lines that cannot name a model are passed over quickly.
- A `command` provider whose script crashes is reported by its exception (`FileNotFoundError: ...`), not by the line `Traceback (most recent call last):` before it.
- Two `python:` providers on the same file no longer race. In `pick`, one failed at random with "has no callable" while the other was still loading the module.
- The package ships `py.typed`, so mypy checks code that imports `llmshift.record` instead of reporting it as untyped.
- On pull requests from forks and Dependabot, which get no secrets, the workflow `init --github` writes and the example workflows skip the comparison with a notice instead of failing. Pull requests from the repository itself run it as before, whatever the provider.
- A `regex` check or a JSON schema `pattern` that backtracks without end no longer hangs `compare` (`^(\w+\s?)*$` took over a minute on a 141-character answer): the search is stopped after 2 seconds, a pattern stopped twice is not tried again on those answers in that run (the candidate's and a live baseline's count apart, even under one model name) (1,000 such cases would have taken 33 minutes), and the case is reported as not checked with how to rewrite the pattern. Judge replies are read in linear time (one of 40,000 digits took 40 s).
- `watch` shows its pass rate over the cases it is computed on: with 1 of 10 requests unanswered it read `90% (9/9)`, now `90% (9/10)`. Counts say "1 error", not "1 errors".
- A server that asks to wait longer than llmshift's 60-second maximum ("Retry-After: 1 day") fails at once and says so, instead of being asked again every minute. A server that was reached but did not answer in time is reported as slow, not as unreachable.
- The GitHub Action can be used as `@v0`: each release moves that tag once PyPI has the release, and never to a pre-release. Pin an exact version or a commit SHA for builds that must not change.
- `THIRD_PARTY_NOTICES` carries the full licenses of LiteLLM's price list and the vendored PyYAML, in the source archive and the wheel.
- Corrupted watch history, calibration and cache files are survived. `watch` writes the action's `pass-rate` output.
- `import openai-evals` imports odd or hostile exports and registries or says what is wrong, instead of crashing; registry samples outside the registry are not read.
- `shadow report` works from a subfolder, and `snapshot` turns shadow logs into test cases.
- The lifecycle data update holds back shutdown dates that moved earlier for a person to check, and the retirement calendar is rebuilt right after a data update instead of the next day.

## 0.1.0 (2026-10-22)

First release. Runs on Python 3.9 to 3.15 on Linux, macOS and Windows; pure Python, so it installs without a compiler (PyYAML's pure-Python parser is included under `llmshift/_vendor`, MIT License).

**Find what has to move**

- `scan`: model detection for Python (AST), JS/TS and other languages, configs, `.env` and notebooks. Lifecycle data covers 617 models from OpenAI, Anthropic, Google (Gemini API and Vertex AI), Mistral, Cohere, xAI and DeepSeek, built from official deprecation pages on 2026-10-21. Status is computed by date. Output as SARIF, JSON or Markdown, with GitHub annotations.
- `update`: fetches newer lifecycle data and prices; they are validated, stored per user, and used only when newer than the bundled copy. The data in this repository updates itself every day from the OpenAI, Anthropic, Gemini API and Vertex AI deprecation tables (a conservative merge, tested before it is committed); other source pages open an issue when they change.
- `digest`: one message listing the models your repositories use that are retired or retire soon, via Slack, Teams, Discord, e-mail, a webhook or a file.
- `calendar`: a static retirement-calendar website with an iCalendar (RFC 5545) feed and JSON, published daily to GitHub Pages by this repository; a daily watch of the providers' deprecation pages opens an issue when one changes.

**Turn real traffic into tests**

- `snapshot`: usage logs become JSONL test sets. Reads OpenAI chat/responses/batch, Anthropic, Gemini, Helicone, Langfuse/LangSmith-style exports, chat datasets, CSV and `llmshift.record` logs. Reassembles logged streams and pairs batch files per batch. Includes redaction, deduplication, grouping by prompt template, diverse sampling and inferred checks. Logs of any size are streamed, with a fair sample per template beyond `--max-records`.
- Agents: `llmshift.record.agent_run()` links the calls of one task, and unlabelled logs are linked when each request continues the previous one. Runs replay on the new model with the recorded tool results; the `agent` check reports missing, extra or changed tool calls, loops, order changes and step-limit runs, and reports show both step traces and the cost per task.
- `import openai-evals`: hosted Evals API (testing criteria, runs, output items as baselines), exported JSON and the OSS registry, with a fidelity check against recorded verdicts.

**Choose the new model**

- `compare`: format, facts, tone, tool-call, refusal, language, structure, length and similarity checks; an optional LLM judge; and a differential baseline that filters noise and compares the two models case by case (McNemar). Reports in terminal, HTML, JSON, Markdown and JUnit form. Automatically adapts to model API quirks.
- Confidence: every pass rate has a 95% range and the number of cases needed for ±5 points; `--repeat N` finds flaky cases. The judge is only asked when the rule-based checks cannot decide (`--judge-mode always` to ask every time), and `judge label` / `judge calibrate` measure its agreement with your own labels, which later runs quote.
- `pick`: the savings finder. It runs the test set on several cheaper models at once and recommends the cheapest one that still passes, showing each model's pass rate with a 95% range, cost per 1,000 requests, saving against today's model (monthly with `--volume`) and p50 latency. Prices come from a bundled LiteLLM snapshot, `pricing:` or `--price`.
- `safety`: refusal, prompt-injection and PII-leak probes built from your own traffic, run on the old and new model and compared pairwise (McNemar); exits 1 when the new model is significantly riskier.
- `embeddings`: measures how search results change between two embedding models on your own documents and queries (overlap, top-result survival, rank-biased overlap), and estimates re-indexing cost and time.

**Switch**

- `fix`: rule-based, example-based and optional optimizer-based prompt search, validated on the new model, with the location of each prompt in the codebase.
- `upgrade`: "Dependabot for AI models". It swaps retiring model ids where `scan` finds them, renames `max_tokens` for OpenAI's newer models, applies `fix` suggestions, tests the replacement and opens or updates a pull request.
- `check`: runs only the tests for the prompts a pull request edits, with the edited text, and comments with the result (one comment, kept up to date).

**After switching**

- Shadow mode: `wrap(client, shadow="provider:model", shadow_rate=0.05)` copies a share of live calls to a new model in the background, never blocking or breaking the app; `shadow report` checks the copies with the same checks as `compare`.
- `watch`: scheduled live checks with history, McNemar-tested drift on the tests that ran unchanged in both runs, version/fingerprint changes and retirement warnings. Alerts go to Slack, Teams, Discord, webhooks, GitHub issues, e-mail or files.

**Everywhere**

- `mcp`: a Model Context Protocol server (stdio) exposing scan, models, snapshot, compare, fix, pick, safety, check, upgrade and doctor as tools for Claude Code, Cursor, VS Code and other MCP clients.
- Providers: OpenAI (Chat + Responses), Azure OpenAI, Anthropic, Gemini, OpenAI-compatible presets, local servers, `python:` functions, `command` providers and offline mocks.
- `llmshift.record` client wrapper for the OpenAI and Anthropic SDKs: sync, async, streaming and the SDKs' stream helpers, `parse()` with pydantic response formats (logged as JSON schema), tool calls and usage; safe for several threads and processes writing one file. GitHub composite action; offline `demo`.
- Hardened against hostile input: YAML alias bombs, gzip bombs, huge, truncated or corrupt logs, broken Unicode, wrongly typed config values, oversized or deeply nested provider responses, unwritable paths and interrupted runs.
