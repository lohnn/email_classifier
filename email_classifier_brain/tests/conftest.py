
"""
conftest.py — global test bootstrap.

Sets the environment the documented testing pattern requires (CLAUDE.md:
"Set TESTING=true, ADMIN_API_KEY=testkey, and TRAINING_DATA_DIR=<tmpdir> in
env before importing main") BEFORE any test module imports main.

Without this, the alphabetically-first test module's own env setup can race
with classify.py's module-level model load (classify imports setfit and
loads the real model when TESTING is unset, which explodes on machines
without the trained model — e.g. CI or a fresh clone).
"""

import os
import sys
import tempfile

# Make the brain package importable regardless of how pytest was invoked
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

# Skip the heavy ML model load and point training data at a throwaway dir,
# so test modules can safely import main.
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("ADMIN_API_KEY", "testkey")
if not os.environ.get("TRAINING_DATA_DIR"):
    os.environ["TRAINING_DATA_DIR"] = tempfile.mkdtemp(prefix="brain-tests-")
