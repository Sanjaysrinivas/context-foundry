"""Single-source local and CI quality gates."""

import nox

nox.options.sessions = ["lint", "typecheck", "tests"]


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """Check lint and formatting with the locked environment."""
    session.run("uv", "run", "--frozen", "ruff", "check", "src", "tests", "noxfile.py")
    session.run("uv", "run", "--frozen", "ruff", "format", "--check", "src", "tests", "noxfile.py")


@nox.session(venv_backend="none")
def typecheck(session: nox.Session) -> None:
    """Run strict static type checking."""
    session.run("uv", "run", "--frozen", "mypy", "src", "tests", "noxfile.py")


@nox.session(venv_backend="none")
def tests(session: nox.Session) -> None:
    """Run tests with coverage. Additional pytest arguments are forwarded."""
    session.run(
        "uv",
        "run",
        "--frozen",
        "pytest",
        "--cov=local_rag",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        *session.posargs,
    )
