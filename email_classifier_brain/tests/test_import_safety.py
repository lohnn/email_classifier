
"""Import-Time Safety — regression test for the 2026-10-24 production incident.

The lazy-import refactor (fd5c608) removed `from setfit import SetFitModel`
but left `def _load_model() -> SetFitModel:` — an annotation evaluated at
DEF-TIME on py≤3.13 → NameError at module import → email-classifier
crash-loop on the Pi (systemd counter 886+).

Why the old suite could not catch it: the dev venv here runs python 3.14
(PEP 649 — annotations lazy by default), so the NameError only fires when
annotations are FORCED (get_type_hints) or on the Pi's pre-3.14 python.
These tests encode BOTH halves of the invariant:
  1. importing classify.py must succeed with `setfit` POISONED — the llama
     path never requires setfit at import time;
  2. on files WITHOUT `from __future__ import annotations`, no annotation
     may reference a name that does not exist at module scope (the
     def-time-evaluation class on py≤3.13).
"""

import ast
import builtins
import importlib.util
import os
import sys
import types

import pytest

BRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSERT_FILE = os.path.join(BRAIN, "classify.py")


class _SetfitPoisoner:
    """Import hook that makes ANY import of `setfit` raise ImportError —
    simulates the Pi-side absence of the SetFit stack."""

    def find_module(self, name, path=None):
        if name == "setfit":
            raise ImportError("setfit poisoned for import-safety test")
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "setfit":
            raise ImportError("setfit poisoned for import-safety test")
        return None


def test_classify_imports_cleanly_with_setfit_poisoned(monkeypatch):
    """THE invariant: with EMBEDDER_BACKEND=llama_cpp (default), importing
    classify.py must not require `setfit` at import time. The poison makes
    any setfit import attempt raise — a clean module import proves no such
    attempt happens at module scope (the model LOAD stays lazy)."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setattr(sys, "meta_path", [_SetfitPoisoner()] + sys.meta_path)
    sys.modules.pop("setfit", None)

    spec = importlib.util.spec_from_file_location(
        "classify_poisoned_probe",
        os.path.join(BRAIN, "classify.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)   # would have raised NameError pre-fix

    # and the import-clean module is functional-on-face: backend default llama
    assert module.EMBEDDER_BACKEND == "llama_cpp"
    assert module.CLS_PROMPT.startswith("task: classification")


def test_no_unevaluated_deferred_import_names_in_annotations():
    """AST scan of every brain .py file WITHOUT `from __future__ import
    annotations`: on py≤3.13 those files evaluate annotations at def time,
    so any annotation name must exist at module scope. Catches the exact
    class that crash-looped production (`-> SetFitModel` after the import
    was dropped)."""
    failures = []
    for root, dirs, files in os.walk(BRAIN):
        dirs[:] = [d for d in dirs if d not in ("venv", "__pycache__", "TrainingData", ".pytest_cache")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src, filename=path)
            has_future = any(
                isinstance(n, ImportFrom) and n.module == "__future__"
                and any(a.name == "annotations" for a in n.names)
                for n in tree.body
                for ImportFrom in (ast.ImportFrom,)
            )
            if has_future:
                continue  # annotations stored as strings — safe by construction
            available = {n for n in dir(builtins)}
            for node in tree.body:
                if isinstance(node, (ast.Import,)) or isinstance(node, ast.ImportFrom):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            available.add((a.asname or a.name).split(".")[0])
                    else:
                        for a in node.names:
                            available.add(a.asname or a.name.split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    available.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            available.add(t.id)

            def _fail_names(node):
                bad = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        if not hasattr(builtins, sub.id) and sub.id not in available:
                            bad.add(sub.id)
                return bad

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in list(node.args.args) + list(node.args.kwonlyargs) + list(getattr(node.args, "posonlyargs", [])):
                        if arg.annotation is not None:
                            for bad in _fail_names(arg.annotation):
                                failures.append(f"{path}: argument {arg.arg}: {bad!r} in annotation")
                    if node.returns is not None:
                        for bad in _fail_names(node.returns):
                            failures.append(f"{path}: return {node.name}: {bad!r} in annotation")
    assert not failures, (
        "annotation-time names that violate py≤3.13 eager evaluation:\n  "
        + "\n  ".join(failures)
    )


def test_type_checking_block_present_in_classify():
    """The canonical lazy-annotation pattern: both halves together."""
    with open(ASSERT_FILE, encoding="utf-8") as f:
        src = f.read()
    assert "from __future__ import annotations" in src
    assert "if TYPE_CHECKING:" in src
    assert "from setfit import SetFitModel" in src   # introspection-only
