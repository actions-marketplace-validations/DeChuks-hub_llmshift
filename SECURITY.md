# Security

## Reporting a vulnerability

Please report it privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. Please don't open a public issue for it.

The report goes only to the maintainer. Fixes ship in a new release, and the CHANGELOG names the advisory.

Only the latest release gets security fixes.

## How llmshift is kept safe

Four layers of checks:

- **Tests on every run.** `tests/test_security.py` runs with the rest of the tests, including from the source archive. It fails if llmshift's own code:
  - uses `exec`, `eval` or `compile`;
  - imports code by name anywhere except `python:` providers;
  - uses a shell, pickle, or an unsafe YAML loader;
  - turns off certificate checks or uses `tempfile.mktemp`;
  - uses MD5 or SHA-1 for security;
  - contains `0.0.0.0` or hidden bidirectional text.
- **Scans on every push and pull request, and every Monday.** The `security` workflow runs:
  - [Bandit](https://bandit.readthedocs.io/) at every severity on llmshift's own code, and at medium and above on the rest of the source archive. Any finding fails the run. The few safe-by-design exceptions (the child processes listed below, seeded random sampling, statuses named "pass") are each marked `# nosec Bxxx` in the code, with the reason;
  - `pip-audit` for known vulnerabilities in dependencies;
  - the tests above.

  GitHub's CodeQL also scans the Python code and the workflows.
- **Releases wait for the scans.** `release.yml` runs the same scans and publishes to PyPI only if they pass. Releases use PyPI trusted publishing, so no upload token exists to leak.
- **Dependency updates.** Dependabot opens a pull request each week for updated actions and dependencies.

## Code that llmshift runs on purpose

Two features run Python code that you supply. Both run with your own permissions. Neither is a sandbox.

- **Python graders** (`type: python` checks, usually imported from OpenAI Evals).
  - They are off by default. They run only when you pass `--allow-python` or set `allow_python_checks: true`.
  - Each grader is written to a temporary directory and imported by a separate Python process, which has a timeout.
  - llmshift itself never hands their code to `exec` or `eval`.
- **Your own providers.**
  - `python:my_app.llm:answer` imports your module or file and calls your function, in the same way that pytest imports your tests.
  - A `command` provider runs the program your config names, never through a shell. Quotes group the words of a string command, and nothing in it is expanded, piped or chained. If you want shell features, name the shell yourself, for example `["sh", "-c", "..."]`.

llmshift also runs `git` (to list files and to open pull requests). It always finds git's full path in your `PATH`, never in the current directory, so a `git.exe` inside a cloned repository cannot run in its place.

Because a config can name programs to run, llmshift only reads a config that belongs to your project. It looks in the current directory and its parents, and stops at:
- the git root;
- your home directory;
- the project an MCP server was started on.

It never reads a config from a drive root or `/`, where any user can create `C:\.llmshift`. On Linux and macOS it ignores a config file that another user owns, and says so. On Windows, outside your home directory, it looks above the current directory only up to the git root. Git closed the same hole as CVE-2022-24765.

Treat `llmshift.yaml`, or `[tool.llmshift]` in `pyproject.toml`, like code:

- Don't turn on python graders for evals you haven't read.
- Don't run llmshift on a configuration from a source you don't trust.
  - In GitHub Actions, run it on `pull_request`, not `pull_request_target`. On `pull_request`, a pull request from a fork gets no secrets and only a read-only token.
- Keep API keys in environment variables or repository secrets, never in the config file. Before llmshift turns logs into test files, it redacts the secrets and personal data it recognises.
