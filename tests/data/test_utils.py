"""Tests for generic data utilities."""

from dnanet.data.utils import generate_random_name


class TestGenerateRandomName:
    def test_returns_string(self):
        name = generate_random_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_capitalized(self):
        name = generate_random_name()
        assert name[0].isupper()
