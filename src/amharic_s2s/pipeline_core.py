"""Small dependency-free helpers for the recorded-question integration test."""
import re

VERSION = 'pipeline-v1-recorded-questions'
MODES = ('sequential', 'clauses')


class ClauseBuffer:
    """Conserve generated text; prefer punctuation, otherwise bound long text by words.

    Boundaries are candidates for speech, NOT semantic or factual approval.
    Greeting-only fragments are held for the following substantive text.
    """
    def __init__(self, max_words=24):
        self.pending = ''
        self.max_words = max_words

    @staticmethod
    def substantive(text):
        words = re.sub(r'[^\w\s]', ' ', text).split()
        return bool(re.search(r'\d', text)) or len(words) >= 3

    def feed(self, text='', final=False):
        self.pending += text
        out = []
        while self.pending:
            cut, reason = None, None
            for match in re.finditer(r'[።፧!?]|\.(?=\s)|\.(?=$)', self.pending):
                # A final dot may still be the start of a decimal in the next delta.
                if match.group() == '.' and match.end() == len(self.pending) and not final:
                    continue
                if self.substantive(self.pending[:match.end()]):
                    cut, reason = match.end(), 'punctuation'
                    break
            if cut is None:
                # Only split at an observed word boundary, never inside a partial word.
                words = list(re.finditer(r'\S+\s+', self.pending))
                if len(words) >= self.max_words:
                    cut, reason = words[self.max_words-1].end(), 'word_fallback'
            if cut is None and final:
                cut, reason = len(self.pending), 'eos_remainder'
            if cut is None:
                break
            piece, self.pending = self.pending[:cut], self.pending[cut:]
            if piece.strip():
                out.append((piece, reason))
            elif out:
                old, why = out[-1]
                out[-1] = (old+piece, why)
        return out


def quality_flags(text):
    from .llm_backend import diagnostics
    d = diagnostics(text)
    flags = []
    if d['empty_answer']: flags.append('empty')
    if d['reasoning_marker']: flags.append('reasoning_marker')
    if d['ethiopic_letter_ratio'] < .8: flags.append('non_amharic_letters')
    if d['repeated_trigram_fraction'] > .35: flags.append('repetition')
    return flags


def accepted(record, review):
    return (record.get('status') == 'success'
            and record.get('stop_reason') == 'eos'
            and not record.get('quality_flags')
            and all(review.get(k) is True for k in
                    ('asr_correct','answer_correct','first_chunk_meaningful','tts_faithful'))
            and type(review.get('naturalness')) is int
            and review['naturalness'] >= 3)


def metrics(record):
    e, chunks = record.get('events', {}), record.get('chunks', [])
    result = {}
    origin = e.get('asr_inference_start')
    def delta(name, end, start):
        if end is not None and start is not None: result[name] = end-start
    delta('asr_total_s', e.get('asr_done'), e.get('asr_prepare_start'))
    delta('asr_inference_and_decode_s', e.get('asr_done'), origin)
    delta('llm_full_answer_s', e.get('llm_done'), e.get('llm_start'))
    delta('full_completion_from_asr_s', e.get('complete'), origin)
    if chunks:
        delta('candidate_tts_start_from_asr_s', chunks[0]['tts_start'], origin)
        delta('first_audio_ready_from_asr_s', chunks[0]['audio_ready'], origin)
        result['first_chunk_tts_s'] = chunks[0]['audio_ready']-chunks[0]['tts_start']
        # Hypothetical immediate playback of server-ready waveforms. No transport/browser.
        cursor, gaps = chunks[0]['audio_ready'], 0.0
        for c in chunks:
            gaps += max(0.0, c['audio_ready']-cursor)
            cursor = max(cursor,c['audio_ready'])+c['audio_duration_s']
        result['simulated_playback_gap_s'] = gaps
        result['audio_duration_s'] = sum(c['audio_duration_s'] for c in chunks)
    return result
