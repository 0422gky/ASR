import sys
from pathlib import Path

ASR_ROOT = Path(__file__).resolve().parents[1]
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from industrial_postprocess import postprocess_text
from industrial_normalizer import normalize_text
from tools.build_industrial_eval_csv import infer_reference_id, reconstruct_asr_text


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


def test_spoken_error_code_after_error_context():
    assert final_text("报错负幺零六七幺八六幺三五") == "报错-1067186135"


def test_spoken_error_code_full_sentence():
    text = "编码器错误提示报错负幺零六七幺八六幺三五怎么回事"
    assert final_text(text) == "编码器错误提示报错-1067186135怎么回事"


def test_existing_error_code_is_preserved():
    assert final_text("报错-1067186135怎么回事") == "报错-1067186135怎么回事"


def test_spoken_error_code_correction_log():
    result = normalize_text("报错负幺零六七幺八六幺三五")
    assert result["correction_log"][0]["rule"] == "spoken_error_code"
    assert result["correction_log"][0]["source"] == "负幺零六七幺八六幺三五"
    assert result["correction_log"][0]["replacement"] == "-1067186135"


def test_reconstruct_asr_text_from_correction_log():
    final = "编码器错误提示报错-1067186135怎么回事"
    log = [{
        "rule": "spoken_error_code",
        "source": "负幺零六七幺八六幺三五",
        "replacement": "-1067186135",
    }]
    assert reconstruct_asr_text(final, log) == "编码器错误提示报错负幺零六七幺八六幺三五怎么回事"


def test_infer_reference_id_from_train_stem():
    assert infer_reference_id("BY-1-7") == "7"
    assert infer_reference_id("17号台词无噪音") == "17"
