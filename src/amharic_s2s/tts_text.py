"""Cached preparation for the pinned Amharic VITS tokenizer (Transformers 5.17).

Preserves the upstream preparation order; only Uroman's lifecycle changes.
No number expansion or spelling changes are added to this comparison.
"""
from functools import partial


def cached_prepare(tokenizer, romanizer, text, is_split_into_words=False, normalize=None, **kwargs):
    if tokenizer.phonemize or tokenizer.language != 'amh':
        raise ValueError('Cached preparation supports non-phonemizing Amharic VITS only.')
    normalize = tokenizer.normalize if normalize is None else normalize
    if normalize:
        text = tokenizer.normalize_text(text)
    text = tokenizer._preprocess_char(text)
    if tokenizer.is_uroman and any(ord(c) > 127 for c in text):
        text = romanizer.romanize_string(text)
    if normalize:
        text = ''.join(c for c in text if c in tokenizer.encoder).strip()
    return text, kwargs


def install_cached_romanizer(tokenizer):
    import uroman
    if tokenizer.phonemize or tokenizer.language != 'amh' or not tokenizer.is_uroman:
        raise ValueError('Unexpected MMS tokenizer configuration.')
    romanizer = uroman.Uroman()
    tokenizer.prepare_for_tokenization = partial(cached_prepare, tokenizer, romanizer)
    return romanizer
