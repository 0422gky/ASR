"""
Industrial term correction based on configs/industrial_terms.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_TERMS_PATH = Path(__file__).resolve().parent / "configs" / "industrial_terms.yaml"


@dataclass(frozen=True)
class Term:
    canonical: str
    aliases: tuple[str, ...]
    category: str
    priority: int


def _fallback_parse_terms(path: Path) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_aliases = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "terms:":
            continue
        if stripped.startswith("- canonical:"):
            if current:
                terms.append(current)
            current = {"canonical": stripped.split(":", 1)[1].strip(), "aliases": []}
            in_aliases = False
            continue
        if current is None:
            continue
        if stripped == "aliases:":
            in_aliases = True
            continue
        if in_aliases and stripped.startswith("- "):
            current["aliases"].append(stripped[2:].strip())
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip()
            current[key.strip()] = int(value) if key.strip() == "priority" else value
            in_aliases = False

    if current:
        terms.append(current)
    return {"terms": terms}


def load_terms(config_path: str | Path | None = None) -> list[Term]:
    path = Path(config_path) if config_path else DEFAULT_TERMS_PATH
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = _fallback_parse_terms(path)

    terms = []
    for item in data.get("terms", []):
        canonical = str(item["canonical"])
        aliases = tuple(str(alias) for alias in item.get("aliases", []))
        category = str(item.get("category", "product"))
        priority = int(item.get("priority", 0))
        terms.append(Term(canonical, aliases, category, priority))
    return terms


def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias.strip())
    escaped = re.sub(r"\\\s+", r"\\s+", escaped)
    return re.compile(escaped, flags=re.IGNORECASE)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


class TermCorrector:
    def __init__(
        self,
        config_path: str | Path | None = None,
        enable_fuzzy: bool = False,
        fuzzy_threshold: float = 0.92,
    ):
        self.terms = load_terms(config_path)
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

    def correct(self, text: str) -> dict:
        correction_log: list[dict] = []
        fixed_terms: list[dict] = []
        corrected = text or ""

        replacements = []
        for term in self.terms:
            for alias in term.aliases:
                if alias and alias != term.canonical:
                    replacements.append((term.priority, len(alias), alias, term))
        replacements.sort(key=lambda item: (item[0], item[1]), reverse=True)

        for _, _, alias, term in replacements:
            pattern = _alias_pattern(alias)

            def repl(match: re.Match, alias: str = alias, term: Term = term) -> str:
                source = match.group(0)
                if source == term.canonical:
                    return source
                entry = {
                    "rule": "term_alias",
                    "source": source,
                    "replacement": term.canonical,
                    "canonical": term.canonical,
                    "alias": alias,
                    "category": term.category,
                    "priority": term.priority,
                    "span": [match.start(), match.end()],
                }
                correction_log.append(entry)
                fixed_terms.append(entry)
                return term.canonical

            corrected = pattern.sub(repl, corrected)

        if self.enable_fuzzy:
            corrected = self._fuzzy_correct(corrected, correction_log, fixed_terms)

        return {
            "text": corrected,
            "fixed_terms": fixed_terms,
            "correction_log": correction_log,
        }

    def _fuzzy_correct(self, text: str, log: list[dict], fixed_terms: list[dict]) -> str:
        corrected = text
        tokens = re.findall(r"[A-Za-z0-9 ]{4,}|[\u4e00-\u9fffA-Za-z0-9]+", corrected)
        for token in tokens:
            compact_token = _compact(token)
            if not compact_token:
                continue
            best: tuple[float, Term] | None = None
            for term in self.terms:
                candidates = (term.canonical, *term.aliases)
                score = max(SequenceMatcher(None, compact_token, _compact(c)).ratio() for c in candidates)
                if score >= self.fuzzy_threshold and (best is None or score > best[0]):
                    best = (score, term)
            if best is None or token == best[1].canonical:
                continue
            entry = {
                "rule": "term_fuzzy",
                "source": token,
                "replacement": best[1].canonical,
                "canonical": best[1].canonical,
                "category": best[1].category,
                "priority": best[1].priority,
                "score": round(best[0], 4),
            }
            corrected = corrected.replace(token, best[1].canonical, 1)
            log.append(entry)
            fixed_terms.append(entry)
        return corrected


def correct_terms(
    text: str,
    config_path: str | Path | None = None,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 0.92,
) -> dict:
    return TermCorrector(config_path, enable_fuzzy, fuzzy_threshold).correct(text)
