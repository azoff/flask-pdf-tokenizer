"""Tests for DocSend view-URL parsing (simple + spaces formats)."""
from main import docsend_doc_id


def test_simple_view_url():
    assert docsend_doc_id("https://docsend.com/view/abc123") == "abc123"


def test_spaces_url_keeps_full_path():
    # The spaces form must keep <space>/d/<doc> — taking only the last
    # segment 404s.
    url = "https://docsend.com/view/vi8t4d2qiqmz6n7w/d/gd9fxvwtrdgmjsfg"
    assert docsend_doc_id(url) == "vi8t4d2qiqmz6n7w/d/gd9fxvwtrdgmjsfg"


def test_strips_query_and_trailing_slash():
    assert docsend_doc_id("https://docsend.com/view/abc123/?foo=bar") == "abc123"


def test_url_without_view_falls_back_to_last_segment():
    assert docsend_doc_id("https://example.com/some/path/xyz") == "xyz"
