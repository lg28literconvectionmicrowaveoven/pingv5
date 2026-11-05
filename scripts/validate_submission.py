#!/usr/bin/env python3
"""Validate one harness submission before it can be merged.

The game's rules for a submission are (docs/rulebook.md, section 12, and
harness_api/harness.py):

  * one ``.py`` file with one concrete ``Harness`` subclass that declares a ``NETID``;
  * only the allowed libraries: the Python standard library, ``harness_api`` and
    ``pydantic`` -- no other third-party packages, no internet, no other models or APIs;
  * no files and no persistent storage.

This script enforces the parts that can be checked statically, then runs the game's own
self-check (``harness_api.check``), which runs the file the way the game does: one process,
one method call per step, within the real time limits, and verifies that every return value
is a valid submission.

    python scripts/validate_submission.py examples/my_harness.py
    python scripts/validate_submission.py examples/my_harness.py --class medusa --rounds 2

Exit code 0 means the submission passes every check.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import tempfile
from pathlib import Path

# Third-party packages a harness may import. Everything else must be in the standard
# library or it is rejected as "not an allowed library".
ALLOWED_PACKAGES = {"harness_api", "pydantic", "pydantic_core"}

# Standard-library modules that reach the filesystem, the network, the OS or other
# processes. Even though they are stdlib, they break the "no internet" and "no files"
# rules and are rejected.
FORBIDDEN_MODULES = {
    # filesystem / persistent storage
    "os", "shutil", "tempfile", "pathlib", "io", "pickle", "shelve", "dbm",
    "sqlite3", "marshal", "glob", "fnmatch",
    # network / internet / other services
    "socket", "ssl", "urllib", "http", "ftplib", "smtplib", "poplib", "imaplib",
    "nntplib", "telnetlib", "xmlrpc", "asyncio",
    # processes / native code / dynamic loading
    "subprocess", "ctypes", "multiprocessing", "signal", "mmap", "importlib",
    "runpy",
}

# Builtins / attribute names that create, write, change or delete files (or load code).
FORBIDDEN_BUILTINS = {"open", "eval", "exec", "compile", "__import__"}
FORBIDDEN_METHODS = {"write", "write_text", "write_bytes", "touch", "mkdir", "makedirs",
                     "dump", "save", "unlink", "remove", "rmdir", "truncate", "open"}


class Problem:
    def __init__(self, line: int, message: str):
        self.line = line
        self.message = message

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}" if self.line else self.message


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def check_imports(tree: ast.Module) -> list[Problem]:
    problems: list[Problem] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                problems.extend(_check_module(alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import: a single .py file has no package
                problems.append(Problem(node.lineno, "relative imports are not allowed"))
                continue
            if node.module:
                problems.extend(_check_module(node.module, node.lineno))
    return problems


def _check_module(module: str, lineno: int) -> list[Problem]:
    top = module.split(".")[0]
    if top in ALLOWED_PACKAGES:
        return []
    if top in FORBIDDEN_MODULES:
        return [Problem(lineno, f"forbidden import: '{module}' (no files, no internet)")]
    if top not in sys.stdlib_module_names:
        return [Problem(lineno, f"not an allowed library: '{module}'")]
    return []


def check_file_ops(tree: ast.Module) -> list[Problem]:
    problems: list[Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
            problems.append(Problem(node.lineno, f"forbidden call: {func.id}(...) "
                                                 f"(no files, no code loading)"))
        elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_METHODS:
            problems.append(Problem(node.lineno, f"forbidden file operation: .{func.attr}(...)"))
    return problems


def _run_self_check(submission: Path, cls: str, rounds: int) -> tuple[bool, list[str]]:
    """Run the game's self-check with the process's working directory set to a fresh
    temporary directory, so a harness that writes files (relative to its cwd) leaves them
    behind and is caught. Returns (passed, created files)."""
    from harness_api.check import check

    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        os.chdir(d)
        try:
            passed = check(submission, cls, rounds)
        finally:
            os.chdir(old)
        created = [str(p.relative_to(d)) for p in Path(d).rglob("*") if p.is_file()
                   or p.is_dir()]
    return passed, created


def validate(submission: Path, cls: str, rounds: int) -> bool:
    print(f"== checking {submission} ==")
    ok = True

    try:
        tree = _parse(submission)
    except SyntaxError as e:
        print(f"FAIL: cannot parse {submission}: {e}")
        return False

    import_problems = check_imports(tree)
    file_problems = check_file_ops(tree)
    for problem in import_problems + file_problems:
        print(f"FAIL: {problem}")
        ok = False
    if import_problems or file_problems:
        print("FAIL: the submission uses forbidden imports or file operations.")

    print(f"\n-- the game's self-check ({cls}, {rounds} round(s)) --")
    passed, created = _run_self_check(submission, cls, rounds)
    ok = ok and passed
    if created:
        ok = False
        print(f"FAIL: the harness created files: {', '.join(sorted(created))}")

    print("\nPASS: submission is valid." if ok else "\nFAIL: submission rejected.")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="validate_submission", description=__doc__.split("\n\n")[0])
    ap.add_argument("submission", type=Path, help="the harness .py file to validate")
    ap.add_argument("--class", dest="cls", default="medusa", help="class to play (default: medusa)")
    ap.add_argument("--rounds", type=int, default=2, help="rounds to play (default: 2)")
    args = ap.parse_args(argv)
    if not args.submission.is_file():
        ap.error(f"no such file: {args.submission}")
    return 0 if validate(args.submission.resolve(), args.cls, max(1, args.rounds)) else 1


if __name__ == "__main__":
    sys.exit(main())
