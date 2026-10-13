import pytest

from aimu.models._internal.endpoint import recordable_endpoint


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://batlin.local:8000/v1/", "http://batlin.local:8000/v1"),
        ("https://user:secret@api.example.com/v1?api_key=abc#frag", "https://api.example.com/v1"),
        ("https://api.anthropic.com", "https://api.anthropic.com"),
        ("http://[::1]:11434/v1", "http://[::1]:11434/v1"),
        ("http://user:pw@[::1]:8000/v1?k=1", "http://[::1]:8000/v1"),
    ],
)
def test_a_recorded_endpoint_drops_credentials_query_and_trailing_slash(url, expected):
    assert recordable_endpoint(url) == expected


def test_no_url_records_no_endpoint():
    assert recordable_endpoint(None) is None
