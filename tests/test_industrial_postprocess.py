import sys
from pathlib import Path

ASR_ROOT = Path(__file__).resolve().parents[1]
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from industrial_postprocess import postprocess_text
from industrial_normalizer import normalize_text


def final_text(text):
    return postprocess_text(text)["final_text"]


def test_chinese_seconds_to_digits():
    assert final_text("停顿二十秒") == "停顿20秒"


def test_chinese_degrees_to_digits():
    assert final_text("超过四十度报警") == "超过40度报警"


def test_par_id_term():
    assert final_text("写入par ID时提示值超出范围") == "写入ParID时提示值超出范围"


def test_mc_tcs_coordinate_term():
    assert final_text("不能用MC TCS坐标系") == "不能用mcTCs坐标系"


def test_acopostrak_spaced_term():
    assert final_text("检查AC O P O S trak轨道") == "检查AcOPOStrak轨道"


def test_workstation_case():
    assert final_text("从a工位移动") == "从A工位移动"


def test_normalizer_handles_workstation_case():
    assert normalize_text("从a工位移动")["text"] == "从A工位移动"
