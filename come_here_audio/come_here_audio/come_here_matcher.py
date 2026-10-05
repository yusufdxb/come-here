"""Bounded "come here" wake matcher: whole tokens, never substrings.

Used by WhisperPhraseDetector when its trigger set is exactly {"come here"};
phrase_matcher.match_trigger stays in place for any other trigger set.

WHY NOT SUBSTRING + DIFFLIB
---------------------------
match_trigger returns ratio 1.0 whenever "come here" occurs anywhere inside the
transcript, and otherwise accepts any word window scoring 0.80 with difflib. On
a 57-sentence negative stress set it fired 11 times: "welcome here everyone",
"become here", "income here", "outcome here", "comes here", "coming here",
"came here", "home here", "some here", "come there", "welcome here".

EVIDENCE (lab 2026-09-15)
-------------------------
The 09-14 robot-mic recordings (57 spoken "come here", 20 negative utterances
including "go here", "over here, come on", "come along", ordinary lab speech)
were replayed through the node's segmenter, Whisper settings and gating, and
every Whisper transcript in the 09-14 lab logs (50) was re-scored:

    substring + ratio 0.80   49/57 hits  0 false wakes  stress negatives 11/57
    this matcher             49/57 hits  0 false wakes  stress negatives  2/57
    lab-log transcripts      identical decisions on all 50

The two stress negatives it still fires on are the exact whole utterances
"coming here" and "welcome here". Those are how Whisper transcribed a real
"come here" in the recordings, and they are accepted ONLY as the entire
utterance, so "they are coming here" and "welcome here everyone" do not wake.
The aliases come from the same recordings they are scored on (one example
each), so their benefit is in-sample.

DELIBERATELY NOT MATCHED
------------------------
"from here", "go here", "up here", "from me": real mishears of "come here" in
the recordings, and also ordinary phrases, so recovering them trades false
wakes for hits. "come over here": a 09-14 false wake ("go over here").
"calm here", "comb here", "cum here": real words with no recorded evidence.
Misses caused by the confidence gate or Whisper's VAD are not matcher problems.
"""

from __future__ import annotations

import re
from typing import List, Optional

from come_here_audio.phrase_matcher import PhraseMatch

CANONICAL = 'come here'

#: Spellings of "come" accepted as the first token. "come" plus non-word
#: phonetic spellings of /k^m/, /kom/, /kaem/ (accented or clipped delivery).
COME_FORMS = frozenset({'come', 'kum', 'kom', 'kam'})
#: Spellings of "here" accepted as the second token. "hear" is the homophone
#: Whisper produces; "hier" is the German-accented spelling.
HERE_FORMS = frozenset({'here', 'hear', 'heer', 'heere', 'hier'})
#: Transcripts that are "come here" run together.
JOINED_FORMS = frozenset({'comehere'})
#: Accepted only when the ENTIRE normalized utterance equals one of these.
#: "coming here" / "welcome here": observed transcripts of a real "come here"
#: (09-14 recordings). "cone here": a measured Whisper mishear, but "put the cone
#: here" is lab speech, so never inside a sentence.
WHOLE_UTTERANCE_ALIASES = frozenset({'coming here', 'welcome here', 'cone here'})

_NON_WORD = re.compile(r"[^a-z' ]+")


def tokens(text: str) -> List[str]:
    """Lowercase words; hyphens and punctuation split, apostrophes kept."""
    text = (text or '').lower().replace('-', ' ')
    return _NON_WORD.sub(' ', text).split()


def match_come_here(transcript: str) -> Optional[PhraseMatch]:
    """PhraseMatch for "come here" or None.

    ratio: 1.0 for the exact words, 0.9 for an accepted spelling variant,
    0.85 for a whole-utterance alias. Diagnostics only; nothing gates on it.
    """
    words = tokens(transcript)
    if not words:
        return None
    for first, second in zip(words, words[1:]):
        if first in COME_FORMS and second in HERE_FORMS:
            heard = f'{first} {second}'
            return PhraseMatch(phrase=CANONICAL, heard=heard,
                               ratio=1.0 if heard == CANONICAL else 0.9)
    for word in words:
        if word in JOINED_FORMS:
            return PhraseMatch(phrase=CANONICAL, heard=word, ratio=0.9)
    utterance = ' '.join(words)
    if utterance in WHOLE_UTTERANCE_ALIASES:
        return PhraseMatch(phrase=CANONICAL, heard=utterance, ratio=0.85)
    return None


# -- "good boy": praise that stands a seated robot back up -------------------
#
# Same rules as "come here": whole tokens only, never a substring, so
# "goodbye", "good boyfriend" and "a good buoy" do not match. No lab
# recordings of "good boy" exist yet: the variants below are spellings, not
# measured mishears. Re-score against robot-mic recordings before adding more.

PRAISE = 'good boy'

GOOD_FORMS = frozenset({'good'})
BOY_FORMS = frozenset({'boy', 'boi'})
PRAISE_JOINED_FORMS = frozenset({'goodboy', 'goodboi'})


def match_good_boy(transcript: str) -> Optional[PhraseMatch]:
    """PhraseMatch for "good boy" or None (1.0 exact, 0.9 spelling variant)."""
    words = tokens(transcript)
    for first, second in zip(words, words[1:]):
        if first in GOOD_FORMS and second in BOY_FORMS:
            heard = f'{first} {second}'
            return PhraseMatch(phrase=PRAISE, heard=heard,
                               ratio=1.0 if heard == PRAISE else 0.9)
    for word in words:
        if word in PRAISE_JOINED_FORMS:
            return PhraseMatch(phrase=PRAISE, heard=word, ratio=0.9)
    return None
