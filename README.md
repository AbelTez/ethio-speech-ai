# Ethio Speech AI

A self-hosted Amharic voice assistant that answers spoken questions using an
organization's reference documents, then speaks the answer. This repository is
the clean team handoff of the tested bank demo.

**Start here:** [Colab setup and feedback guide](COLAB_GUIDE.md). For server setup,
prerequisites and the HTTP API, read [ETHIO-SPEECH-AI-DOC.md](ETHIO-SPEECH-AI-DOC.md).

## Share and run

Send teammates **`dist/ethio-speech-ai-colab.zip`** and `COLAB_GUIDE.md`.
The ZIP also contains both guides and the notebook.

1. Extract the ZIP locally and upload `notebooks/ethio_speech_ai_colab.ipynb` to Colab.
2. Upload the **original ZIP** to your own Drive at
   `MyDrive/ethio_speech_ai/inputs/ethio-speech-ai-colab.zip`.
3. Choose a T4 GPU, enable your `HF_TOKEN` Colab secret, and run cells 1–5 in order.
4. Speak the eight test questions, review answers and audio, then export feedback
   with cell 6. Run cell 7 and disconnect/delete the runtime when finished.

Each tester supplies their own name in cell 1. Results go to their own Drive under
`ethio_speech_ai/results/<tester>/`. Previous experiments or recordings are not needed.

## Selected pipeline

```text
Microphone → VAD (800 ms silence) → Hohe ASR → local document search
           → Gemma 4 E4B / llama.cpp → OmniVoice → spoken answer
```

| Part | Configuration |
|---|---|
| ASR | Hohe Amharic, FP16 |
| LLM | Gemma 4 E4B instruction model, pinned Q4_K_M GGUF, thinking disabled |
| TTS | Amharic OmniVoice, 16 steps, selected mixed precision, whole-answer synthesis |
| Documents | 41 DOCX sources compiled into 1,590 records; CPU lexical retrieval |
| Interface | Browser microphone and automatic endpointing; FastAPI service |

Models stay loaded and warm on one GPU. Listening pauses while the assistant
processes and speaks. The public API receives complete WAV recordings and returns
complete synthesized WAV responses. It is not a full-duplex streaming API.

The latency target measures **ASR decoding start → TTS invocation**. Time until
the user hears audio is measured separately. Results depend on hardware and input;
this handoff does not claim a guaranteed response time or complete bank coverage.

## Repository contents

| Path | Purpose |
|---|---|
| `notebooks/ethio_speech_ai_colab.ipynb` | The one teammate demo notebook |
| `COLAB_GUIDE.md` | Setup, questions, evaluation, export and troubleshooting |
| `ETHIO-SPEECH-AI-DOC.md` | Architecture, server installation and API reference |
| `TEAM_GUIDE.md` | Link to the renamed full team guide |
| `src/amharic_s2s/` | Runtime, UI, retrieval and required shared helpers |
| `configs/`, `requirements/` | Pinned model/tool settings and dependencies |
| `knowledge/bank_v2/` | Prepared bank index and source audit, included in the private ZIP |
| `data/` | Three fixed startup warmup recordings and their manifest |
| `tests/` | CPU contract and browser behavior checks |
| `scripts/package_colab.py` | Rebuild and verify the handoff ZIP without models or a GPU |
| `dist/` | The final shareable ZIP and its SHA-256 checksum |

Some shared Python modules retain experiment-era names because the selected
runtime imports their helpers. Their code is preserved from the working demo;
historical experiment notebooks, reports and patch scripts are omitted.
The supported entry points are this notebook, the API, document builder and tool
setup described in `TEAM_GUIDE.md`.

## Maintain the package

From this directory:

```bash
python3 scripts/package_colab.py
```

Packaging uses an explicit inventory, validates local imports and notebook syntax,
and includes a per-file `BUNDLE_MANIFEST.json`. The ZIP has files at its root so
the notebook can extract it directly. It contains no nested ZIP, Git metadata,
model weights, installed environments, past results or credentials.

Run focused CPU checks after code changes:

```bash
python -m pip install -e '.[api,api-test]'
python -m unittest discover -s tests -p 'test_*.py'
node tests/test_empty_speech.mjs
node tests/test_endpoint.mjs
node tests/test_capture.mjs
```

These checks do not establish GPU performance or Amharic speech quality. Review
the team's exported audio and answers before changing the selected configuration.

## Sharing and contribution

This is currently a **private team handoff** containing bank reference text and
warmup recordings. Those assets and the final ZIP are ignored by Git but included
in the local distribution. A Git clone alone needs the private package assets.
Do not assume these materials are cleared for public redistribution.

Before an open-source release, choose a project license and replace or obtain
redistribution permission for the bank documents and recordings. Model weights
are downloaded separately under their own terms. Vendored VAD/ONNX notices remain
in `src/amharic_s2s/web/vendor/`. Submit focused contributions with tests and, for
model changes, recorded answer quality, listening feedback and hardware details.
