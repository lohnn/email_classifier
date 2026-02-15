import os
import sys
import importlib
from unittest.mock import patch

def test_imap_server_config():
    """
    Test that IMAP_SERVER is loaded from the environment variable.
    """
    # We want to ensure we start fresh or at least reload to pick up the env var
    # Since IMAP_SERVER is a module-level constant, we must reload the module.

    with patch.dict(os.environ, {"IMAP_SERVER": "imap.custom.com"}):
        if "imap_client" in sys.modules:
            import imap_client
            importlib.reload(imap_client)
        else:
            import imap_client

        assert imap_client.IMAP_SERVER == "imap.custom.com"

    # Cleanup: We should probably reload again to restore the original state (or default)
    # so other tests aren't affected by our "imap.custom.com" change if they run after this.
    # The 'patch.dict' context manager restores os.environ, but the module 'imap_client'
    # still holds the value read during the 'with' block until it's reloaded.

    import imap_client
    importlib.reload(imap_client)
