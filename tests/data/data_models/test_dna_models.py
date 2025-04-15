import pytest

from DNAnet.data.data_models import Panel


def test_panel():
    panel = Panel(pytest.PANEL_PATH)

    # test get_allele_name_by_dye_and_bp
    assert panel.get_allele_name_by_dye_and_bp(0, 81.5) == ('AMEL', 'X')

    with pytest.raises(ValueError) as e:
        panel.get_allele_name_by_dye_and_bp(100, 81.5)
    assert "Unknown dye row" in str(e)

    with pytest.raises(ValueError) as e:
        panel.get_allele_name_by_dye_and_bp(0, 1234)
    assert "Unknown dye row" in str(e)
