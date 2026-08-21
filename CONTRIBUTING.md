# Contributing

## Branch workflow

1. Update local `dev`: `git switch dev && git pull --ff-only`.
2. Create a focused branch: `git switch -c feature/short-name`.
3. Make atomic commits using Conventional Commits.
4. Run `uv run nox` and update documentation when behavior changes.
5. Open a pull request into `dev` using the repository template.
6. Promote tested release candidates from `dev` to `main` through a pull request.

Do not branch features from `main` or open feature pull requests directly into `main`.

## Setup

```powershell
uv sync --frozen
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run nox
```

The hooks format Python, run Ruff and mypy, and reject non-conventional commit messages. CI runs the same nox sessions.

## Commit examples

```text
feat(api): add collection statistics
fix(retrieval): reject mixed vector dimensions
docs: explain local model setup
test: cover empty retrieval results
```

Use `feat` for user-visible capability, `fix` for a defect, and `!` or a `BREAKING CHANGE:` footer for incompatible behavior. Keep refactors, tests, documentation, and CI changes in their own commits when practical.

## Tests

- `uv run nox -s lint` checks Ruff lint and formatting.
- `uv run nox -s typecheck` runs strict mypy.
- `uv run nox -s tests` runs pytest with an 80% coverage floor.
- Mark fast isolated tests with `unit`; use `integration` when local infrastructure is exercised.
- Provider tests must use fakes or HTTP mocks. CI must never require API keys or downloaded models.
- For retrieval, chunking, parsing, prompt, or model changes, run the matching private golden dataset with `uv run local-rag-eval evaluation/private/cases.jsonl` and report metric changes in the pull request.

## Design guardrails

- Preserve independent chat and embedding provider selection.
- Keep the default path local and free.
- Prefer direct provider APIs and the standard library over new framework layers.
- Add a dependency only when it replaces more complexity than it creates.
- Changing embedding models requires a collection rebuild; never mix vector spaces.
- Golden labels are human-authored at question/evidence level. Do not generate “truth” from chunks or use model answers as unreviewed expected results.

