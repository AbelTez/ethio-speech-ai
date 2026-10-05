"""Versioned scoring: preserve spelling; never hide substitutions by folding letters."""
import re
import unicodedata

NORMALIZER = "amharic-nfc-punctuation-leading-amh-v1"


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    # Strip only the known metadata tag at the beginning, not arbitrary words/tags.
    text = re.sub(r"^\s*\[AMH\]\s*", "", text, flags=re.IGNORECASE)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split()).lower()


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def score(reference, hypothesis):
    result = {}
    for prefix, ref, hyp in (("raw", " ".join(reference.split()), " ".join(hypothesis.split())),
                             ("normalized", normalize(reference), normalize(hypothesis))):
        words, chars = ref.split(), "".join(ref.split())
        if not words or not chars:
            raise ValueError("Reference must contain scoreable text.")
        we = distance(words, hyp.split())
        ce = distance(chars, "".join(hyp.split()))
        result.update({f"{prefix}_word_errors": we, f"{prefix}_ref_words": len(words),
                       f"{prefix}_char_errors": ce, f"{prefix}_ref_chars": len(chars),
                       f"{prefix}_wer": we / len(words), f"{prefix}_cer": ce / len(chars)})
    return result


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def aggregate(records):
    """Corpus error rates; never average utterance WER to get corpus WER."""
    good = [r["metrics"] for r in records if r["status"] == "success"]
    out = {"successful": len(good)}
    if not good:
        return out
    for prefix in ("raw", "normalized"):
        for unit, metric in (("word", "wer"), ("char", "cer")):
            out[f"{prefix}_{metric}"] = sum(m[f"{prefix}_{unit}_errors"] for m in good) / sum(m[f"{prefix}_ref_{unit}s"] for m in good)
    for metric in ("asr_latency_s", "inference_s", "rtf"):
        for q in (50, 90):
            out[f"{metric}_p{q}"] = percentile([m[metric] for m in good], q / 100)
    out["peak_vram_gib"] = max(m["peak_vram_gib"] for m in good)
    return out
