# Contributing to Bio Network

Thanks for your interest in Bio Network. This project is an honest research and
educational effort, and we welcome contributions from computational
neuroscientists, ML researchers, and students.

Please read this guide and the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before
getting started.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch (see [Branch naming](#branch-naming)).
3. Install the package in editable mode with dev dependencies:

   ```bash
   python -m pip install -e ".[dev]"
   ```

4. Run the test suite:

   ```bash
   python -m pytest
   ```

## Branch Naming

Use short, descriptive branch names that state the intent:

- `feature/<short-name>` -- new functionality
- `fix/<short-name>` -- bug fixes
- `docs/<short-name>` -- documentation only
- `chore/<short-name>` -- maintenance, tooling, packaging

Examples: `feature/stdp-rule`, `fix/scheduler-delay-bug`, `docs/m1-neuron-models`.

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
Every commit message must follow the format:

```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

Common types:

- `feat:` -- a new feature (adds or changes user-facing or simulation behavior)
- `fix:` -- a bug fix
- `docs:` -- documentation only
- `chore:` -- maintenance, packaging, CI, formatting, no functional change
- `refactor:` -- code change that neither fixes a bug nor adds a feature
- `test:` -- adding or correcting tests
- `perf:` -- performance improvement

Examples:

```
feat(engine): add Izhikevich regular spiking regime
fix(scheduler): honor post-synaptic spike delay
docs: expand STDP references in ROADMAP
chore: bump dev dependencies
```

Use Semantic Versioning (SemVer) for releases: `MAJOR.MINOR.PATCH`. Versions
before 1.0 are experimental; breaking changes bump the MINOR version.

## Code Style

- Format with [Black](https://github.com/psf/black) (default settings).
- Use [Ruff](https://github.com/astral-sh/ruff) for linting.
- Add type hints to all public functions and methods.
- Write NumPy-style or Google-style docstrings for public APIs.
- Do not add comments that restate the code; prefer a clear identifier or a
  short docstring.

The expected workflow is `black . && ruff check .` before committing.

## Tests

- Tests live in `tests/` and use `pytest`.
- New functionality must come with tests that cover the happy path and at least
  the obvious edge cases.
- Simulations must be deterministic or seeded so tests are reproducible.

Run all checks locally:

```bash
python -m pytest
black --check .
ruff check .
```

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a pull request against `main` using the provided template.
3. Ensure the CI checks pass.
4. Keep the change focused; split unrelated changes into separate PRs.

## Research Standards

- Cite primary literature for any scientific claim (see References in the
  README).
- Be modest: no claims of AGI, consciousness, or replicating the human brain.
- Prefer reproducible, seeded experiments over headline numbers.
