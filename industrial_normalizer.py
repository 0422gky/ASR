"""
Industrial ASR text normalization utilities.

The normalizer is deliberately small and rule-based so it can be called after
any ASR backend. It returns both text and a JSON-friendly correction log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


DEFAULT_UNITS = ("秒", "度", "分钟", "毫米", "厘米", "米")
SPOKEN_DIGITS = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "幺": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}


@dataclass
class NormalizerConfig:
    convert_chinese_numbers: bool = True
    normalize_spoken_error_codes: bool = True
    normalize_workstations: bool = True
    normalize_model_case: bool = True
    normalize_unit_spacing: bool = True
    units: tuple[str, ...] = field(default_factory=lambda: DEFAULT_UNITS)


def _cn_to_int(value: str) -> int | None:
    try:
        from cn2an import cn2an

        return int(cn2an(value, "smart"))
    except Exception:
        pass

    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value in digits:
        return digits[value]
    if "十" in value and len(value) <= 3:
        left, _, right = value.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _replace_with_log(
    text: str,
    pattern: re.Pattern,
    repl: Callable[[re.Match], str],
    rule: str,
    log: list[dict],
) -> str:
    def wrapped(match: re.Match) -> str:
        before = match.group(0)
        after = repl(match)
        if before != after:
            log.append({
                "rule": rule,
                "source": before,
                "replacement": after,
                "span": [match.start(), match.end()],
            })
        return after

    return pattern.sub(wrapped, text)


class IndustrialNormalizer:
    def __init__(self, config: NormalizerConfig | None = None):
        self.config = config or NormalizerConfig()

    def normalize(self, text: str) -> dict:
        correction_log: list[dict] = []
        normalized = text or ""

        if self.config.normalize_spoken_error_codes:
            normalized = self._normalize_spoken_error_codes(normalized, correction_log)

        if self.config.convert_chinese_numbers:
            normalized = self._normalize_chinese_numbers(normalized, correction_log)

        if self.config.normalize_unit_spacing:
            normalized = self._normalize_unit_spacing(normalized, correction_log)

        if self.config.normalize_workstations:
            normalized = self._normalize_workstations(normalized, correction_log)

        if self.config.normalize_model_case:
            normalized = self._normalize_model_case(normalized, correction_log)

        return {"text": normalized, "correction_log": correction_log}

    def _normalize_spoken_error_codes(self, text: str, log: list[dict]) -> str:
        context = r"(报错|错误码|错误提示|故障码)"
        digit_chars = "".join(SPOKEN_DIGITS)
        pattern = re.compile(rf"({context})(负)?([{digit_chars}](?:\s*[{digit_chars}]){{5,}})")

        def repl(match: re.Match) -> str:
            context_text = match.group(1)
            sign = "-" if match.group(3) else ""
            spoken_code = match.group(4)
            digits = "".join(
                SPOKEN_DIGITS[char]
                for char in spoken_code
                if char in SPOKEN_DIGITS
            )
            if len(digits) < 6:
                return match.group(0)

            source = f"{match.group(3) or ''}{spoken_code}"
            replacement = f"{sign}{digits}"
            log.append({
                "rule": "spoken_error_code",
                "source": source,
                "replacement": replacement,
                "span": [match.start(3) if match.group(3) else match.start(4), match.end(4)],
            })
            return f"{context_text}{replacement}"

        return pattern.sub(repl, text)

    def _normalize_chinese_numbers(self, text: str, log: list[dict]) -> str:
        unit_pattern = "|".join(re.escape(unit) for unit in self.config.units)
        pattern = re.compile(rf"([零〇一二两三四五六七八九十百千万]+)\s*({unit_pattern})")

        def repl(match: re.Match) -> str:
            number = _cn_to_int(match.group(1))
            if number is None:
                return match.group(0)
            return f"{number}{match.group(2)}"

        return _replace_with_log(text, pattern, repl, "chinese_number_to_digit", log)

    def _normalize_unit_spacing(self, text: str, log: list[dict]) -> str:
        unit_pattern = "|".join(re.escape(unit) for unit in self.config.units)
        pattern = re.compile(rf"(?<![-\d])(\d+)\s+({unit_pattern})")
        return _replace_with_log(
            text,
            pattern,
            lambda match: f"{match.group(1)}{match.group(2)}",
            "unit_spacing",
            log,
        )

    def _normalize_workstations(self, text: str, log: list[dict]) -> str:
        pattern = re.compile(r"(?<![A-Za-z])([a-zA-Z])\s*工位")
        return _replace_with_log(
            text,
            pattern,
            lambda match: f"{match.group(1).upper()}工位",
            "workstation_case",
            log,
        )

    def _normalize_model_case(self, text: str, log: list[dict]) -> str:
        pattern = re.compile(r"\bacopos\s*d\s*1\b", flags=re.IGNORECASE)
        return _replace_with_log(
            text,
            pattern,
            lambda match: "ACOPOSD1",
            "model_case",
            log,
        )


def normalize_text(text: str, config: NormalizerConfig | None = None) -> dict:
    return IndustrialNormalizer(config).normalize(text)
