"""The bounded "come here" matcher: what must wake, what must not.

Negatives are the known mishears / near phrases, substring traps and ordinary
lab speech (including the robot's own spoken replies picked up by its mic).
Positives are transcripts Whisper actually produced for a real "come here" on
the robot mic, plus the spelling variants the matcher accepts on purpose.
"""
from types import SimpleNamespace

import pytest

from come_here_audio.come_here_matcher import match_come_here

NEGATIVES = [
    # known near phrases and mishears that are ordinary speech
    'from here', 'go here', 'Go here!', 'come over here', 'up here', 'from me',
    'over here, come on!', 'go over here, come on, come in.', 'come along', 'coming',
    # substring traps
    'welcome here everyone', 'become here', 'income here', 'outcome here',
    'comes here', 'came here', 'they are coming here', 'home here', 'some here',
    # near words
    'come there', 'come on', 'come in', 'come back', 'come with me', 'right here',
    'over here', 'get over here', 'calm down', 'stay calm here', 'the comb is here',
    'put the cone here', 'come ear',
    # ordinary lab speech and the robot's own replies
    'hey guys!', 'is it recording', 'okay one more time', 'good boy!', 'thanks!',
    'here i am.', 'there you are.', 'i see you.', 'coming closer', "i'm here",
    'hey fetch', 'hey robot', 'you', 'here', 'come', '', '   ',
]

OBSERVED_POSITIVES = [
    'come here', 'come here.', 'Come here!', 'hey guys, come here!', 'come here with me',
    'come here, come here, come here.', 'coming here', 'Welcome here',
]

VARIANT_POSITIVES = [
    'come hear', 'kum here', 'kom here', 'kam here', 'come heere', 'come hier',
    'comehere', 'come-here', 'robot, come here please', 'cone here',
]


@pytest.mark.parametrize('text', NEGATIVES)
def test_negatives_do_not_wake(text):
    assert match_come_here(text) is None


@pytest.mark.parametrize('text', OBSERVED_POSITIVES + VARIANT_POSITIVES)
def test_positives_wake_as_come_here(text):
    match = match_come_here(text)
    assert match is not None
    assert match.phrase == 'come here'


def test_exact_words_report_full_ratio_and_variants_less():
    assert match_come_here('hey, come here!').ratio == 1.0
    assert match_come_here('come hear').ratio == pytest.approx(0.9)
    assert match_come_here('coming here').ratio == pytest.approx(0.85)


def test_aliases_only_match_the_whole_utterance():
    assert match_come_here('coming here') is not None
    assert match_come_here('is he coming here') is None
    assert match_come_here('welcome here') is not None
    assert match_come_here('welcome here everyone') is None
    assert match_come_here('cone here') is not None
    assert match_come_here('put the cone here') is None


def _detector(segments, triggers=None):
    from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector

    detector = WhisperPhraseDetector.__new__(WhisperPhraseDetector)
    detector._ct2_model = SimpleNamespace(transcribe=lambda audio, **kw: (iter(segments), None))
    detector._whisper_vad_filter = True
    detector._no_speech_threshold = 0.75
    detector._confidence_threshold = 0.20
    detector._phrase_ratio_threshold = 0.80
    if triggers is not None:
        detector.TRIGGER_PHRASES = triggers
    return detector


def _segment(text, avg_logprob=-0.3, no_speech_prob=0.1):
    return SimpleNamespace(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def test_detector_uses_the_bounded_matcher_for_come_here():
    assert _detector([_segment(' Become here.')])._transcribe_ct2(None) is None
    detection = _detector([_segment(' Hey guys, come here!')])._transcribe_ct2(None)
    assert detection is not None
    assert detection.phrase == 'come here'


def test_detector_gates_still_run_before_matching():
    low_conf = _detector([_segment(' come here', avg_logprob=-0.89)])
    assert low_conf._transcribe_ct2(None) is None
    no_speech = _detector([_segment(' come here', no_speech_prob=0.9)])
    assert no_speech._transcribe_ct2(None) is None


def test_other_trigger_sets_keep_the_generic_matcher():
    detector = _detector([_segment(' stop now')], triggers={'stop now'})
    detection = detector._transcribe_ct2(None)
    assert detection is not None
    assert detection.phrase == 'stop now'
