"""Fuzzy matcher for Whisper mishears of trigger phrases.

Whisper transcribes "come here" as "come hear" / "cone here" / "come ear"
often enough that exact substring matching drops real detections. This
module normalizes transcripts, checks exact substring first, then slides
a word window over the transcript and scores each window against each
trigger with ``difflib.SequenceMatcher``. Pure-Python, no extra deps.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class PhraseMatch:
    phrase: str   # canonical trigger matched (e.g. "come here")
    heard: str    # normalized transcript span that matched
    ratio: float  # 0.0–1.0 similarity (1.0 = exact substring)


def match_trigger(
    transcript: str,
    triggers: Iterable[str],
    ratio_threshold: float = 0.80,
) -> Optional[PhraseMatch]:
    """Best trigger match above ``ratio_threshold``, or ``None``.

    Strategy:
      1. Normalize transcript and each trigger.
      2. If a trigger is a substring of the transcript, return ratio=1.0.
      3. Otherwise, slide a word window of width ``n-1``, ``n``, ``n+1`` for
         each trigger of ``n`` words and score each window with
         ``SequenceMatcher.ratio``. Widths ±1 absorb single insert/delete.
      4. Keep the highest-scoring window ≥ threshold across all triggers.
    """
    norm_transcript = normalize(transcript)
    if not norm_transcript:
        return None
    words = norm_transcript.split()

    best: Optional[PhraseMatch] = None

    for raw_trigger in triggers:
        trigger = normalize(raw_trigger)
        if not trigger:
            continue

        if trigger in norm_transcript:
            return PhraseMatch(phrase=trigger, heard=trigger, ratio=1.0)

        trig_words = trigger.split()
        n = len(trig_words)
        for width in (n - 1, n, n + 1):
            if width <= 0 or width > len(words):
                continue
            for i in range(len(words) - width + 1):
                window = " ".join(words[i:i + width])
                ratio = SequenceMatcher(None, window, trigger).ratio()
                if ratio >= ratio_threshold and (
                    best is None or ratio > best.ratio
                ):
                    best = PhraseMatch(
                        phrase=trigger, heard=window, ratio=ratio
                    )

    return best
