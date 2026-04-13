"""Tests for STRKit."""

import pytest

from dnanet.core.panel import Panel
from dnanet.data.strategies.scaling.kit import STRKit
from dnanet.data.strategies.scaling.size_standard import WEN_ILS


class TestSTRKit:
    def _make_kit(self, **kwargs):
        defaults = dict(
            name="TestKit",
            size_standard=WEN_ILS,
            panel_path="",
            num_dyes=5,
        )
        defaults.update(kwargs)
        return STRKit(**defaults)

    def test_default_dye_mapping(self):
        kit = self._make_kit()
        assert kit.dye_row_from_hid_index(1) == 0
        assert kit.dye_row_from_hid_index(6) == 4

    def test_custom_dye_mapping(self):
        """7-dye kit with no gaps."""
        mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        kit = self._make_kit(num_dyes=7, hid_dye_mapping=mapping)
        assert kit.dye_row_from_hid_index(5) == 4
        assert kit.dye_row_from_hid_index(7) == 6

    def test_invalid_hid_index_raises(self):
        kit = self._make_kit()
        with pytest.raises(ValueError, match="Unknown HID dye index"):
            kit.dye_row_from_hid_index(5)  # skipped in default PPF6C mapping

    def test_frozen(self):
        kit = self._make_kit()
        with pytest.raises(AttributeError):
            kit.name = "changed"
