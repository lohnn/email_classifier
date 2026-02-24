"""
test_cleanup.py — Tests for email content cleanup functions
============================================================

Tests clean_subject() and clean_body() from config.py.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import clean_subject, clean_body


# ---------------------------------------------------------------------------
# clean_subject tests
# ---------------------------------------------------------------------------

class TestCleanSubject:

    def test_mime_q_encoding(self):
        raw = "=?UTF-8?Q?Din_orderbekr=C3=A4ftelse?="
        assert clean_subject(raw) == "Din orderbekräftelse"

    def test_mime_b_encoding(self):
        import base64
        encoded = base64.b64encode("Hej världen".encode("utf-8")).decode("ascii")
        raw = f"=?UTF-8?B?{encoded}?="
        assert clean_subject(raw) == "Hej världen"

    def test_plain_subject_passthrough(self):
        assert clean_subject("Normal subject line") == "Normal subject line"

    def test_empty_subject(self):
        assert clean_subject("") == ""

    def test_none_like_empty(self):
        # Guard against None being passed somehow
        assert clean_subject("") == ""

    def test_mixed_encoded_and_plain(self):
        raw = "Important message about Apputveckling Sverige =?UTF-8?B?4pqg77iP?="
        result = clean_subject(raw)
        assert "Important message about Apputveckling Sverige" in result
        # The emoji should be decoded
        assert "=?UTF-8?" not in result

    def test_iso_8859_encoding(self):
        raw = "=?ISO-8859-1?Q?F=F6rfr=E5gan?="
        assert clean_subject(raw) == "Förfrågan"


# ---------------------------------------------------------------------------
# clean_body tests
# ---------------------------------------------------------------------------

class TestCleanBody:

    def test_html_body_extraction(self):
        html = "<html><body><p>Hello world</p></body></html>"
        result = clean_body(html)
        assert "Hello world" in result
        assert "<html>" not in result
        assert "<p>" not in result

    def test_style_removal(self):
        html = "<html><head><style>body { color: red; }</style></head><body><p>Content</p></body></html>"
        result = clean_body(html)
        assert "Content" in result
        assert "color: red" not in result

    def test_script_removal(self):
        html = "<html><body><script>alert('x')</script><p>Visible</p></body></html>"
        result = clean_body(html)
        assert "Visible" in result
        assert "alert" not in result

    def test_plain_text_passthrough(self):
        text = "This is a plain text email body with no HTML."
        result = clean_body(text)
        assert result == text

    def test_empty_body(self):
        assert clean_body("") == ""

    def test_tracking_url_removal(self):
        body = "Click here: http://clicks.example.com/ls/click?upn=" + "a" * 150 + " for more info."
        result = clean_body(body)
        assert "clicks.example.com" not in result
        assert "for more info." in result

    def test_whitespace_collapse(self):
        body = "Line 1\n\n\n\n\n\nLine 2\n\n\n\n\nLine 3"
        result = clean_body(body)
        # Should not have more than double newlines
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_zero_width_space_removal(self):
        body = "Hello\u200cWorld\u200bTest"
        result = clean_body(body)
        assert "\u200c" not in result
        assert "\u200b" not in result
        assert "HelloWorldTest" in result

    def test_complex_html_email(self):
        """Test with a realistic HTML email snippet."""
        html = """<!doctype html><html><head>
        <style>.header { color: blue; }</style>
        </head><body>
        <div class="header"><h1>Order Confirmation</h1></div>
        <p>Thank you for your order!</p>
        <table><tr><td>Product: Shampoo</td><td>Price: 198 Kr</td></tr></table>
        <p>Total: 237 Kr</p>
        </body></html>"""
        result = clean_body(html)
        assert "Order Confirmation" in result
        assert "Thank you for your order!" in result
        assert "Product: Shampoo" in result
        assert "237 Kr" in result
        assert "<html>" not in result
        assert "<style>" not in result
        assert "color: blue" not in result

    def test_br_tags_become_newlines(self):
        html = "<html><body>Line 1<br>Line 2<br>Line 3</body></html>"
        result = clean_body(html)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_short_url_kept(self):
        """Normal (non-tracking) URLs should be preserved."""
        body = "Check out https://example.com/page for details."
        result = clean_body(body)
        assert "https://example.com/page" in result
