# Amharic Speech Assistant Setup and API Guide

A self-hosted Amharic voice assistant for organization documents. A client sends a
spoken question and receives the recognized text, an answer, supporting document
passages and a WAV response. ASR, retrieval, language generation and speech
synthesis run on the host; no external inference API is required.

This guide describes the implemented single-GPU service and its HTTP contract.
It is intended for teammates integrating the API and contributors adapting it to
another organization. The current bank corpus is an example deployment, not a
required public dataset. Host deployment and wider answer coverage still need
acceptance on the target server.

## Architecture

```text
Browser microphone → local VAD → WAV upload → Hohe ASR
    → local document retrieval → Gemma answer → OmniVoice WAV → browser playback
```

| Component | Selected implementation |
|---|---|
| Speech recognition | `snapwre/hohe-asr-amharic`, FP16 |
| Document search | CPU BM25 plus character TF-IDF, with FAQ title weighting |
| Language model | Gemma 4 E4B instruction model, pinned Q4_K_M GGUF, llama.cpp |
| Speech synthesis | `gheero-Leyu/amharic-omnivoice-tts`, 16 steps, selected mixed precision |
| Service and UI | FastAPI/Uvicorn and the included browser interface |

Models load and warm up once. The browser submits after **800 ms of silence** and
pauses microphone capture during inference/playback. The HTTP API accepts complete
recordings; it does not stream microphone frames or synthesized audio. The complete
checked answer is synthesized before its WAV becomes available. Thinking is off.

## Documents and storage

The current corpus contains **41 DOCX sources**, represented by **675 FAQ records
and 915 section records**. Each record retains its source filename, document hash
and extracted block range. The serving JSON is about **2 MiB**. Search structures
are built once in CPU memory at startup; each question sends only the complete top
passage to Gemma, not the entire corpus. No vector database or embedding GPU is needed.

Deduplication has a precise scope: 41 same-name PDFs are excluded under a DOCX-first
policy, and three exact duplicate records were merged with their origins retained.
PDF text equivalence, overlapping topics and conflicting policies are not resolved
by that policy. `source_audit.json` preserves extracted blocks and excluded-file
hashes for review; it is not loaded for every request.

The context window is **512 tokens total**, reserving **128 for the answer**.
Question, instruction, evidence and retained history must fit the remaining 384.
The runtime drops older history pairs first. A top passage that still cannot fit
causes `retrieval_budget_abstention`; it is not silently shortened. Search matches
are evidence candidates, not proof that the question is answerable.

## Prerequisites

| Requirement | Current deployment path |
|---|---|
| Host | Linux x86-64, glibc **2.39 or newer**, such as Ubuntu 24.04; required by the pinned llama.cpp binary |
| Python | **3.13** with pip and venv reproduces the Colab Python baseline |
| GPU | NVIDIA CUDA GPU; the selected stack has run together on a T4 with about 15 GiB usable VRAM |
| Driver | Compatible with the chosen PyTorch CUDA wheel and the bundled CUDA 12.8 llama.cpp runtime |
| Disk | At least **30 GiB free** before first binary/model setup; keep model caches on persistent storage |
| Network | Package/model downloads during provisioning; fully populated caches are required before offline startup |
| Browser | Microphone permission; HTTPS when accessed remotely, or localhost for development |
| Files | Source/config files, the organization index, and the three warmup WAV files described below |

Use GPU 0 for this configuration. The llama.cpp helper selects GPU 0; changing
only `CUDA_VISIBLE_DEVICES` is not a supported way to remap the entire stack.
CPU-only mode is available for an interface preview, not the selected models.
Minimum host RAM and simultaneous-user capacity have not been established.

Two different secrets are used:

- **`HF_TOKEN`**: read access for model downloads, with any required model access already granted.
- **`AMHARIC_API_TOKEN`**: your own random API secret, at least 32 ASCII characters. Clients and the browser use this value, not the Hugging Face token.

## Set up the GPU service

Run commands from the repository root. The private handoff bundle includes data
assets; a source checkout alone may not, because `data/` and `knowledge/` are
ignored by Git. Required warmup files are:

```text
data/pipeline_public/questions_v2.json
data/pipeline_v2_audio/everyday.wav
data/pipeline_v2_audio/fee.wav
data/pipeline_v2_audio/unknown.wav
```

### Install the environment

Provision the NVIDIA driver and Python first. Then:

```bash
cd /path/to/ethio-speech-ai
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/integrated-colab.txt -r requirements/browser-api.txt
python -m pip install --no-deps omnivoice==0.2.1
python -m pip install -e .
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The CUDA wheel command follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
The OmniVoice install follows our pinned headless runtime rather than installing
its separate Gradio application. Keep these versions together; `pip install -e .`
alone does not install the model dependencies. This server recipe follows the
working Colab setup but has not been executed on the organization's host.

### Configure and provision

Supply `HF_TOKEN` from your team's secret store. Generate a separate API secret
and configure the **same value** in the server and clients; do not commit it.
The example below generates one for this shell:

```bash
export AMHARIC_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export AMHARIC_LLM_ENGINE=llama_cpp
export HF_HOME="$HOME/.cache/amharic-s2s/huggingface"
export AMHARIC_CPP_WORK="$HOME/.cache/amharic-s2s/llama"
export AMHARIC_CPP_LOG_DIR="$PWD/runs/server"
export AMHARIC_KNOWLEDGE_INDEX="$PWD/knowledge/bank_v2/bank_index.json"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m amharic_s2s.llm_engine_compare setup --repo . --work "$AMHARIC_CPP_WORK"
python -m amharic_s2s.api --repo . --host 127.0.0.1 --port 8000
```

Explicitly set `AMHARIC_LLM_ENGINE=llama_cpp`; the legacy default is a different
Transformers backend. The setup command verifies pinned binary checksums. API
startup downloads/reuses pinned weights, verifies GGUF integrity and GPU layer
offload, and performs six warmup turns. Allow several minutes on first startup.
Keep the server running; use a second terminal with the same API secret for clients.

Open `http://localhost:8000`, paste **AMHARIC_API_TOKEN**, click **Connect**, wait
for **Ready** and **Bank document search ready**, then **Start conversation**.
With the bank index loaded, leave `context` empty; automatic retrieval is enforced.
Stop the service with Ctrl+C and allow its worker and llama.cpp child to exit.

For a CPU interface preview, run the API command with `--demo`. It returns fixed
text and a tone, and displays a preview banner; it does not exercise the models.

### Colab alternative

Follow [COLAB_GUIDE.md](COLAB_GUIDE.md) for the complete teammate workflow.

Upload `dist/ethio-speech-ai-colab.zip` to
`MyDrive/ethio_speech_ai/inputs/`, open
`notebooks/ethio_speech_ai_colab.ipynb`, choose T4 and enable `HF_TOKEN`.
Run cells 1–5, review/export in 6, and stop in 7. The notebook fills the API key
automatically. It uses a notebook callback bridge to localhost, not a public API
URL. Colab processes data on Google's runtime and writes demo archives to Drive.

## API reference

Base URL in the example deployment: `http://127.0.0.1:8000`.
All `/ready` and `/v1/turns` routes require:

```http
Authorization: Bearer <AMHARIC_API_TOKEN>
```

| Method | Path | Response and purpose |
|---|---|---|
| GET | `/` | Public browser interface |
| GET | `/health` | Public liveness, `200 {"status":"alive"}`; does not imply models are ready |
| GET | `/ready` | `200` when initialized; `503` while loading or failed |
| POST | `/v1/turns` | Submit WAV, `202` with turn ID and `Location` header |
| GET | `/v1/turns/{id}` | `200` with status/result, or `404` if absent |
| GET | `/v1/turns/{id}/audio` | Completed PCM16 `audio/wav`; `409` before completion |
| POST | `/v1/turns/{id}/cancel` | Request cancellation; poll until terminal |
| DELETE | `/v1/turns/{id}` | `204` after deleting a finished turn; `409` while active |

Static UI assets are public under `/static`. Swagger, ReDoc and OpenAPI endpoints
are currently disabled. No text-only chat, document-upload, WebSocket or SSE
endpoint is exposed by this API.

### Readiness and admission

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -H "Authorization: Bearer $AMHARIC_API_TOKEN" http://127.0.0.1:8000/ready
```

Check both `ready` and `accepting_requests`. A warmed but busy server can return
200 with `accepting_requests:false`. The service accepts **one upload/inference
at a time**; another submission gets **409**, with no request queue. Use one API
worker. Adding Uvicorn workers would duplicate model loading and break admission
coordination.

### Submit a recording

Use `Content-Type: application/json`:

```json
{
  "audio_base64": "<base64 of complete WAV bytes, no data URL prefix>",
  "context": "",
  "grounded": true,
  "history": []
}
```

| Field | Contract |
|---|---|
| `audio_base64` | Required. WAV, 1–30 seconds, mono/stereo, 8–96 kHz; finite, audible samples |
| `context` | Optional string, maximum 4096 characters; must be empty in the current bank-document mode |
| `grounded` | Optional boolean. Bank mode uses documents regardless; without an index this selects the supplied-context prompt |
| `history` | Optional array, maximum two complete user/assistant pairs; each message has exactly `role` and `content`, up to 2000 characters |

WebM/Opus uploads are unsupported. Maximum total JSON body is **16 MiB**, with a
**30-second upload deadline**. A successful submission returns:

```json
{"id":"<turn_id>","status":"running","result":null,"error":null}
```

History is supplied per request, not stored as a global chat session. Send the last
two successful exchanges when continuity is needed; send `[]` for a new topic.
Invalid roles or incomplete pairs are rejected. Do not automatically resubmit a
POST after an ambiguous network timeout: the original turn may already be running.

### Poll and download

Poll approximately every 300 ms. Status is `running`, `cancelling`, `completed`,
`cancelled` or `failed`. On completion, `result` contains:

| Field | Meaning |
|---|---|
| `transcript`, `answer` | Recognized question and final response text |
| `sample_rate`, `audio_url` | Output sample rate and relative WAV endpoint |
| `timings` | ASR, retrieval, LLM, TTS and total processing measurements |
| `details.grounding` | Route, prompt information and context metadata |
| `details.retrieval.selected` | Exact supplied passages and source locations when retrieval was used |
| `compilation` | Backend/timing metadata; retained field name also used with llama.cpp |

Common grounding routes are `documents`, `social_template`, `no_retrieval_match`
and `retrieval_budget_abstention`. Templates and abstentions bypass LLM generation
but still use TTS. A completed response is not a human quality approval.

Fetch audio with the authorization header; browser clients should play the returned
blob. Do not put the API secret in the audio URL. Cancellation is cooperative;
keep polling until it finishes. Cancel and wait before deleting an active turn.

### Minimal Python client

Save this example as `client.py`. Set the same `AMHARIC_API_TOKEN` in the client
shell, then run `python client.py question.wav`. It uses only Python's standard library.

```python
import base64, json, os, sys, time
from pathlib import Path
from urllib.request import Request, urlopen

base = os.environ.get("AMHARIC_API_URL", "http://127.0.0.1:8000").rstrip("/")
headers = {"Authorization": "Bearer " + os.environ["AMHARIC_API_TOKEN"]}

def call(path, payload=None, method=None, audio=False):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=body, method=method,
                  headers={**headers, "Content-Type": "application/json"})
    with urlopen(req, timeout=40) as response:
        data = response.read()
    return data if audio else json.loads(data)

ready = call("/ready")
if not ready["accepting_requests"]:
    raise SystemExit("Server is busy; wait before submitting.")
turn = call("/v1/turns", {
    "audio_base64": base64.b64encode(Path(sys.argv[1]).read_bytes()).decode(),
    "context": "", "grounded": True, "history": []
})
path = "/v1/turns/" + turn["id"]
print("Turn:", turn["id"])
deadline = time.monotonic() + 180
while turn["status"] in ("running", "cancelling"):
    if time.monotonic() > deadline:
        call(path + "/cancel", {}, method="POST")
        raise SystemExit("Cancellation requested; poll this turn until terminal.")
    time.sleep(0.3)
    turn = call(path)
if turn["status"] != "completed":
    raise SystemExit(f"{turn['status']}: {turn.get('error')}")
print("Heard:", turn["result"]["transcript"])
print("Answer:", turn["result"]["answer"])
print("Timings:", turn["result"]["timings"])
Path("answer.wav").write_bytes(call(path + "/audio", audio=True))
print("Saved answer.wav")
```

HTTP errors raise `urllib.error.HTTPError`. Integrations should inspect the status
and body and present a useful message rather than blindly retrying submissions.

### Errors and retention

| HTTP status | Meaning |
|---|---|
| 401 | Missing/incorrect API secret |
| 404 | Unknown, expired or deleted turn |
| 408 / 413 | Upload timeout / body too large |
| 409 | Busy, audio not ready, or active turn cannot be deleted |
| 415 / 422 | Wrong content type / invalid audio or request fields |
| 503 | Runtime loading, unavailable or shutting down |

After acceptance, polling normally returns HTTP 200 even for `status:"failed"`.
Read `error`: `no_speech` means repeat the question; `answer_rejected` means an
unfinished or text-check failure; `invalid_input` can indicate excess prompt size
or prohibited manual context. `busy` and `not_ready` describe runtime admission.
`inference_failed` requires inspection of server logs and may require a restart.
Empty ASR does not unload models; the included automatic UI resumes listening.

The API retains at most **eight turn records**, expiring finished records after
**ten minutes**. Earlier records may be evicted sooner at capacity. Output WAV is
limited to **8 MiB**. Fetch results promptly. The HTTP adapter keeps recordings,
results and generated audio in memory, with no automatic conversation file storage.
The Colab bridge separately archives turns/audio to Drive. Operator diagnostics
can contain model exception details; protect logs alongside organization data.

## Use another organization corpus

The current builder expects DOCX files in these two group directories:

```text
/path/to/reference/Amharic FAQ/*.docx
/path/to/reference/amharic text/*.docx
```

These are builder conventions; replace their contents with your own approved FAQ
and general documents. Both directories must contain DOCX files. A standalone PDF
requires conversion/review first; there is no OCR or general PDF-ingestion endpoint.

```bash
python -m amharic_s2s.bank_corpus --source /path/to/reference --output knowledge/organization
export AMHARIC_KNOWLEDGE_INDEX="$PWD/knowledge/organization/bank_index.json"
```

Build a new index directory, inspect `build_report.json` and `source_audit.json`,
then restart the service to load it. There is no live index reload. Adapt the bank
instruction, social responses and test cases in `knowledge.py` and
`configs/bank_demo_cases.json` when serving a different domain. Test relevant,
unsupported and compound questions against the displayed evidence.

## Deployment and contribution

Keep the API on a private interface or behind an HTTPS reverse proxy. The current
shared token grants access to all retained turns; it is not per-user isolation.
The browser uses the same origin and no cross-origin API policy is enabled.
Multi-tenant authentication, queuing, rate limits and multi-GPU scheduling are
outside the current implementation. Supervise the API and its llama.cpp child as
one service group, and persist caches across restarts.

Provision all artifacts before an offline deployment, then verify startup with
network access disabled. Model downloads during provisioning are separate from
inference traffic. A production process should not require Colab or Drive.

| Area | Main files |
|---|---|
| HTTP and lifecycle | `src/amharic_s2s/api.py`, `runtime.py` |
| Selected GPU backend | `runtime_cpp.py`, `runtime_backend.py` |
| Extraction and search | `bank_corpus.py`, `bank_search.py`, `knowledge.py` |
| Browser and capture | `src/amharic_s2s/web/` |
| Pins and checks | `configs/`, `requirements/`, `tests/` |

Contribute a focused change with relevant CPU tests and, for model behavior,
Amharic answer/listening evidence. Preserve separate timing boundaries and record
hardware, model revision and settings. For example, after installing test dependencies:

```bash
python -m pip install -e '.[api-test]'
python -m unittest discover -s tests -p 'test_runtime*.py'
python -m unittest discover -s tests -p 'test_api.py'
python -m unittest discover -s tests -p 'test_bank_corpus.py'
```


## Current evidence and performance meaning

The 5 October bank pilot retained one greeting and two document answers. Apollo's
definition and the mobile-registration answer agreed with their retrieved text.
One other record was unavailable at export; listening notes were not filled in.
This supports the demonstrated paths, not complete bank coverage or speech approval.

| Retained document turn | Retrieval | ASR start to TTS start | ASR start to full waveform |
|---|---:|---:|---:|
| Apollo definition | 3.8 ms | 2.610 s | 6.033 s |
| Mobile registration | 12.5 ms | 1.593 s | 3.916 s |

The project's **under-three-second target** measures ASR inference/decoding start
through TTS invocation. It is distinct from audible response time: Apollo browser
playback began 9.894 seconds after speech ended in this Colab run. Transport,
endpointing and synthesis contribute additional time. These are individual
observations, not a server latency guarantee. Recognition misses, difficult
compound retrieval and full passage budget limits remain documented limitations.
