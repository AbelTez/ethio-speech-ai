"""Amharic-aware long-text splitting; preserve every character in order.

Punctuation is preferred; whitespace is the fallback for oversized sentences.
The character target is soft only for an indivisible word longer than the target.
This is native synthesis chunking, not streaming playback.
"""
import re

SENTENCE_BOUNDARY = re.compile(r'[።፣፤፥፦፧.!?;:,。！？；，：](?=\s|$)\s*')


def split_speech_text(text, chunk_len, min_chunk_len=None):
    if not isinstance(text,str):raise TypeError('Expected one text string.')
    if chunk_len < 1:raise ValueError('Chunk character target must be positive.')
    if not text.strip():return []
    units=[];start=0
    for match in SENTENCE_BOUNDARY.finditer(text):
        units.append(text[start:match.end()]);start=match.end()
    if start<len(text):units.append(text[start:])
    pieces=[]
    for unit in units:
        if len(unit)<=chunk_len:
            pieces.append(unit);continue
        # Retain whitespace, so joining the chunks reconstructs the exact input.
        current=''
        for token in re.findall(r'\s+|\S+\s*',unit):
            if current and len(current)+len(token)>chunk_len:
                pieces.append(current);current=''
            current+=token
        if current:pieces.append(current)
    chunks=[];current=''
    for piece in pieces:
        if current and len(current)+len(piece)>chunk_len:
            chunks.append(current);current=''
        current+=piece
    if current:chunks.append(current)
    # Merge a short tail only when that keeps the target; never drop it.
    if min_chunk_len and len(chunks)>1 and len(chunks[-1])<min_chunk_len:
        if len(chunks[-2])+len(chunks[-1])<=chunk_len:
            chunks[-2:]=[chunks[-2]+chunks[-1]]
    if ''.join(chunks)!=text:raise RuntimeError('Chunking changed the input text.')
    return chunks


def install_omni_splitter(on_plan):
    """Bind the splitter where OmniVoice 0.2.1 calls it, in this subprocess only."""
    import importlib.metadata
    if importlib.metadata.version('omnivoice')!='0.2.1':
        raise RuntimeError('The tested chunking adapter requires omnivoice==0.2.1.')
    from omnivoice.models import omnivoice as implementation
    def split(text,chunk_len,min_chunk_len=None):
        chunks=split_speech_text(text,chunk_len,min_chunk_len)
        on_plan({'target_characters':chunk_len,'chunk_characters':[len(c) for c in chunks],
                 'chunk_words':[len(c.split()) for c in chunks],'chunks':chunks,
                 'text_preserved':''.join(chunks)==text})
        return chunks
    implementation.chunk_text_punctuation=split
