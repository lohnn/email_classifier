import os
import sys
import importlib
import pytest
from unittest.mock import patch

def test_imap_server_config():
    # Store original environment and modules to restore later
    original_environ = os.environ.copy()
    original_modules = sys.modules.copy()

    try:
        # Set a custom IMAP_SERVER environment variable
        os.environ["IMAP_SERVER"] = "imap.custom.com"

        # Reload imap_client if it's already imported
        if "imap_client" in sys.modules:
            import imap_client
            importlib.reload(imap_client)
        else:
            # We assume imap_client is importable from where we are
            # The test runner usually sets up sys.path
            import imap_client

        # Check if the IMAP_SERVER variable reflects the environment variable
        assert imap_client.IMAP_SERVER == "imap.custom.com", \
            f"Expected 'imap.custom.com', but got '{imap_client.IMAP_SERVER}'"

    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(original_environ)

        # Restore modules (optional, but good practice if other tests run in same process)
        # Note: Reloading modules might have side effects on global state,
        # but for this specific check, restoring sys.modules doesn't undo the reload effect on the object itself
        # if other modules hold references to it. But it's okay for this isolated test run.
