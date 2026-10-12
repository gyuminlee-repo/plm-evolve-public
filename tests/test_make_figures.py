import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figure_lib as fl  # noqa: E402


def _axes_count(svg_path):
    root = ET.parse(svg_path).getroot()
    assert root.tag.endswith("svg")
    return len([g for g in root.iter() if g.get("id", "").startswith("axes_")])


@pytest.mark.skipif(not fl.CACHE_DIR.exists(), reason="DMS cache absent")
def test_fig1_svg_valid_three_panels():
    import make_figures as mf
    mf.build_fig1(mf.FIG1_PATH)
    assert mf.FIG1_PATH.exists()
    # 3 main panels (a/b/c); the matched inset adds a 4th axes
    assert _axes_count(mf.FIG1_PATH) >= 3


@pytest.mark.skipif(not fl.CACHE_DIR.exists(), reason="DMS cache absent")
def test_figS1_and_figS2_render():
    import make_figures as mf
    mf.build_figS1(mf.FIGS1_PATH)
    mf.build_figS2(mf.FIGS2_PATH)
    assert _axes_count(mf.FIGS1_PATH) >= 2
    assert _axes_count(mf.FIGS2_PATH) >= 1
