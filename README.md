# Ethio Speech AI

A self-hosted Amharic voice assistant that answers spoken questions using reference
documents and speaks the response. It combines speech recognition, local document
retrieval, language generation, and speech synthesis in a browser interface backed
by a Python API.

The project provides a practical reference for building document-grounded voice
assistants for low-resource languages. The implemented pipeline targets Amharic;
adapting it to another language requires suitable models, language-specific text
handling, and evaluation by speakers of that language.

## Features

- Spoken questions and answers, with automatic end-of-speech detection.
- Document-grounded responses with source passages available for review.
- Local inference with models kept loaded on one GPU.
- A browser interface, HTTP API, and Google Colab notebook.
- Separate measurements for recognition, retrieval, generation, and synthesis.
- Reproducible model configurations, package checksums, and CPU/browser tests.

## How it works

```text
Microphone → voice activity detection → Hohe ASR → local document search
           → Gemma / llama.cpp → OmniVoice → spoken answer
```

| Component | Current implementation |
|---|---|
| Speech recognition | Hohe Amharic, FP16 |
| Document retrieval | CPU BM25 and character TF-IDF, with FAQ title weighting |
| Language generation | Gemma 4 E4B instruction model, pinned Q4_K_M GGUF via llama.cpp |
| Speech synthesis | Amharic OmniVoice, 16 steps, mixed precision |
| Service | FastAPI and a browser microphone interface |

The browser submits a recording after 800 ms of silence and pauses listening while
the assistant processes and speaks. The API accepts complete WAV recordings and
returns complete synthesized WAV responses. Model weights are downloaded during
setup; inference runs on the host.

## Getting started

### Preview the interface on a CPU

From the repository root, using Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[api]'
export AMHARIC_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
printenv AMHARIC_API_TOKEN
python -m amharic_s2s.api --repo . --demo --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` and connect using the API token printed in the terminal.
This preview uses fixed text and a tone; it requires no model weights, document
index, or warmup recordings.

### Run the speech models

Follow the [setup and API guide](ETHIO-SPEECH-AI-DOC.md) for the GPU service, or the
[Colab guide](COLAB_GUIDE.md) for the notebook workflow.

The selected GPU setup uses Linux x86-64 with glibc 2.39+, Python 3.13, an NVIDIA
CUDA GPU, and at least 30 GiB of free disk before provisioning. The pipeline has
run on a T4 with approximately 15 GiB usable VRAM. You also need Hugging Face model
access, the pinned dependencies, and local data assets described in the guide.

**A source checkout does not include `data/`, `knowledge/`, or `dist/`.** Warmup
recordings, the example bank index, and the Colab ZIP are local assets excluded
from Git. Prepare the required assets before starting the real models or building
a Colab bundle. The bank documents are an example domain, not a required public
dataset or a download supplied by this repository.

## Adapt the project

For another document collection, use the DOCX builder and update the domain
instructions and evaluation questions. See
[using another corpus](ETHIO-SPEECH-AI-DOC.md#use-another-organization-corpus).

For another low-resource language, review the ASR and TTS backends, language model
support, retrieval normalization, prompts, and answer checks. Replacing model IDs
alone is insufficient. The
[language adaptation guide](ETHIO-SPEECH-AI-DOC.md#adapting-to-other-low-resource-languages)
identifies the relevant modules and evaluation steps.

## Limitations

- The service handles one inference request at a time on one GPU.
- Audio is processed by turns; full-duplex streaming and interruption during
  playback are not implemented.
- The current context window is 512 tokens, with 128 reserved for the answer.
  Evidence that cannot fit the input budget results in an insufficient-evidence reply.
- The example domain is bank reference documents. The assistant cannot access
  accounts or execute transactions.
- Recognition, retrieval, answer quality, and speech quality need separate
  evaluation. Existing pilot observations do not establish broad language coverage
  or production readiness.
- The under-three-second target measures ASR decoding start to TTS invocation,
  not time until audible playback. Hardware and input affect latency.

See the [technical guide](ETHIO-SPEECH-AI-DOC.md) for deployment constraints and
recorded performance observations.

## Repository structure

| Path | Purpose |
|---|---|
| `ETHIO-SPEECH-AI-DOC.md` | Architecture, prerequisites, GPU setup, API, and adaptation |
| `COLAB_GUIDE.md` | Notebook setup, evaluation, export, and troubleshooting |
| `notebooks/ethio_speech_ai_colab.ipynb` | Colab workflow for a prepared bundle |
| `src/amharic_s2s/` | Runtime, API, browser interface, retrieval, and evaluation helpers |
| `configs/`, `requirements/` | Model settings and dependency pins |
| `docs/bank-demo-test-cases.md` | Example domain evaluation questions |
| `tests/` | CPU contract and browser behavior checks |
| `scripts/package_colab.py` | Build and verify a Colab bundle from source and local assets |

## Development

Install the API and test dependencies, then run the existing checks:

```bash
python -m pip install -e '.[api,api-test]'
python -m unittest discover -s tests -p 'test_*.py'
node tests/test_empty_speech.mjs
node tests/test_endpoint.mjs
node tests/test_capture.mjs
```

Browser checks require Node.js. These tests exercise contracts and behavior without
loading GPU models; they do not establish speech quality or model performance.

To build a Colab ZIP after preparing the local assets:

```bash
python3 scripts/package_colab.py
```

The packager validates imports, notebook syntax, and audio checksums, then writes
`dist/ethio-speech-ai-colab.zip` and its SHA-256 checksum. The ZIP contains an
explicit file inventory and a per-file `BUNDLE_MANIFEST.json`; model weights,
environments, credentials, and previous results are excluded.

## Contributing

Issues and focused pull requests are welcome. Include reproduction steps for bugs
and relevant tests for behavior changes. For model or language changes, record the
hardware, model revisions, configuration, and recognition/answer/listening results.
Describe known limitations alongside improvements.

## License and attribution

A project license has not yet been selected. Third-party components retain their
own licenses; notices for bundled browser dependencies are in
[`src/amharic_s2s/web/vendor/`](src/amharic_s2s/web/vendor/). Model weights are
obtained separately and remain subject to their respective terms.
