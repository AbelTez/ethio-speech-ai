# Run Ethio Speech AI on your own Colab

This workflow requires a prepared **`ethio-speech-ai-colab.zip`** containing
the notebook, document index, three startup recordings, source code, and guides.
Model weights download during setup/startup.

**The ZIP and data assets are not included in a Git checkout.** Prepare the assets
listed in the [setup guide](ETHIO-SPEECH-AI-DOC.md#set-up-the-gpu-service), then run
`python3 scripts/package_colab.py` from the repository root to create
`dist/ethio-speech-ai-colab.zip`. The packager requires every file in its inventory
and fails if an asset is missing. The instructions and questions below describe
the example bank corpus; adapt the domain cases and readiness expectations for
your own index.

## 1. Prepare your files and accounts

1. Extract the ZIP on your computer. Inside, find
   `notebooks/ethio_speech_ai_colab.ipynb`.
2. In **your own Google Drive**, create `ethio_speech_ai`, then its `inputs` folder.
3. Upload the **original ZIP file** into that folder. Its exact location
   must be `MyDrive/ethio_speech_ai/inputs/ethio-speech-ai-colab.zip`.
   Do not upload the extracted folder instead, rename the ZIP, or put it inside
   another ZIP.
4. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload
   notebook**, and select `ethio_speech_ai_colab.ipynb` from the extracted folder.
5. Choose **Runtime → Change runtime type → T4 GPU → Save**. GPU availability and
   quotas depend on your account. A CPU runtime cannot run this selected demo.
6. On Hugging Face, use your own read token and obtain any required access to
   [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it).
7. In Colab's left sidebar, open **Secrets** (key icon). Add a secret named exactly
   `HF_TOKEN`, paste your Hugging Face token, and enable **Notebook access**.

Use headphones, allow microphone access, and test in a quiet room. The reference
documents are test material, not verified current bank policy. Use sample
questions; the assistant cannot read an account balance or perform transactions.

## 2. Run the seven numbered cells

Run cells individually in order. **Do not use Run all**: cells 6 and 7 export the
review and stop the service before you have had a chance to speak.

| Cell | What to do | What to expect |
|---|---|---|
| **1 — Paths** | Run and enter your tester name, such as `hana` or `dawit_1`. Use ASCII letters/numbers, `_` or `-`. | A separate output folder for your name. |
| **2 — Drive/package** | Run and authorize your own Drive. | `Package verified` after extraction and checksum checks. |
| **3 — Setup** | Run once and wait for completion. | CUDA/GPU check, dependency installation, pinned llama.cpp setup, and document retrieval preflight. |
| **4 — Start models** | Run once. | Starts the model server in the background. |
| **5 — Interface** | Run and wait. | Download/load/warmup messages, then `REAL MODELS READY` and the interface. |
| **6 — Feedback/export** | Run **after** testing and editing `QUALITY_NOTES`. | Saves and downloads your result ZIP. |
| **7 — Stop** | Run after confirming the download. | Stops both servers; then disconnect/delete the runtime. |

First-time installation and model downloads can take several minutes or longer
on a slow connection. Watch the logs; restarting an active load wastes work.
Cell 5 allows up to 20 minutes and reports a timeout or failure rather than
silently claiming readiness.

The interface access key is filled automatically. If the interface asks you to
click **Connect**, do so. This uses a generated API key, **not your HF_TOKEN**.
Wait for **Ready** and **Bank document search ready · 41 source files** (for the
example corpus; the count depends on your index) before
starting a conversation. This should be the real model interface, not the yellow
“Interface preview · no AI models” banner.

## 3. Speak and evaluate

Click **Start conversation**, wait for **Listening**, then speak **one question**.
Pause for at least **800 ms** at the end. Submission is automatic; do not manually
stop/send each recording. Wait for the answer and speech to finish before speaking
again. The assistant does not listen while it is processing or playing audio.

Between unrelated topics use **End conversation → New topic → Start conversation**.
Keep manual reference empty; document mode supplies it automatically. Do not delete
turns or clear their stored results before exporting.

For every turn, check these separately:

1. **Recognition:** Does **Heard** match what you said? If wrong, record the miss
   and repeat the question. Do not blame retrieval for a different recognized question.
2. **Evidence:** Does the displayed source actually answer the question?
3. **Answer:** Is the Amharic relevant, correct and faithful to that source? Check
   numbers, conditions and missing information. A completed request is not a quality pass.
4. **Speech:** Is it understandable and natural? Note wrong pronunciation,
   omissions, repetitions, cut-off endings or unwanted voice changes.
5. **Delay:** Note whether the wait feels acceptable. The export records timings;
   distinguish ASR→TTS start from the longer wait until audible playback.

### Eight core questions

| # | Say this | Expected behavior |
|---|---|---|
| 1 | ሰላም | Natural greeting. |
| 2 | አፖሎ ምንድን ነው | Apollo definition from the matching document. |
| 3 | ያለ ካርድ ከኤቲኤም ገንዘብ ማውጣት እችላለሁ | Explain the source-supported cardless option. |
| 4 | ለአቢሲንያ የሞባይል ባንኪንግ ለመመዝገብ ምን ያስፈልገኛል | Keep the existing-account prerequisite and branch-registration instruction from the supplied FAQ. |
| 5 | የኤቲኤም ካርዴ ቢጠፋብኝ ምን ማድረግ አለብኝ | Longer, complete answer with supported blocking methods; no claim that it has blocked the card. |
| 6 | አሁን በግል ሂሳቤ ውስጥ ስንት ገንዘብ አለ | Explain that it cannot see your balance; do not invent an amount. |
| 7 | ሽሮ እንዴት ይሠራል | Decline the unrelated cooking question from bank-only evidence. |
| 8 | እሺ በጣም አመሰግናለሁ | Natural acknowledgment. |

If time permits, ask your own relevant paraphrases and an unsupported question.
Additional number/compound-query examples are in `docs/bank-demo-test-cases.md`.
An insufficient-evidence reply is appropriate for unknown facts; if it occurs for
a supported question, record it as a retrieval/budget limitation, not a correct
answer to that question. The full top passage must fit a 384-token input budget.

## 4. Export results and evaluation notes

1. Click **End conversation** and wait for any active answer to finish.
2. In cell 6, edit the text **inside the triple quotes** assigned to `QUALITY_NOTES`.
   Fill each line with your actual observation, or write `not tested`. Example:

   ```text
   Apollo definition: Heard correct; source relevant; answer correct;
   speech understandable but repeated the last phrase; wait felt long.
   ```

3. Run cell 6. Allow the browser download. The output prints **RESULT ZIP** and
   also keeps a copy in your Drive under:

   ```text
   MyDrive/ethio_speech_ai/results/<your-name>/bank-demo-<timestamp>-<id>.zip
   ```

4. Run cell 7. Then choose **Runtime → Disconnect and delete runtime** to release
   the GPU. Closing the browser tab alone is not the cleanup step.

The export contains recognized text, answers, available generated audio, retrieved
passages, timings, logs, your notes and package identification. It excludes raw
microphone uploads and intentionally does not export credentials. Results include
document passages and conversation text; review those contents before sharing an
export with a bug report or evaluation.

## 5. Troubleshooting

| Problem | Action |
|---|---|
| `CUDA: False`, CPU-only torch or “Choose a GPU runtime” | Select T4 GPU and reconnect; rerun from cell 1 on the GPU session. Do not remove the GPU assertion. |
| Drive credential propagation failed | Reconnect Colab with the Google account holding the ZIP and complete Drive authorization. Retry cell 2 before proceeding. |
| Package not found | Check the exact Drive path and ZIP filename in step 1. |
| Hash mismatch | Replace the uploaded ZIP with the original package and use a fresh runtime. Do not bypass verification. |
| Missing `HF_TOKEN` or model access 403 | Enable secret access for this notebook; verify that the token's account has model access. |
| Model loading is slow | Check cell 5 progress; do not start a second server. Download time is separate from measured response latency. |
| Wrong or empty “Heard” | Check microphone permission/input, use headphones, speak clearly and repeat. Empty ASR should resume listening. |
| `inference_failed`, GPU OOM or server startup failure | Export partial diagnostics with cell 6 if available, stop with cell 7, and send the error/result. Avoid repeated identical reloads. |
| No interface or missing microphone permission | Let cell 5 finish, allow browser microphone access and keep the Colab tab active. |
| ZIP download blocked | Retrieve the printed result ZIP from your Drive manually before stopping the session. |
| Colab quota exhausted/session deleted | Saved result ZIPs remain in Drive. Model processes and local caches do not; a later session starts again from cell 1. |

For GPU server deployment or API integration, use the
[setup and API guide](ETHIO-SPEECH-AI-DOC.md).
