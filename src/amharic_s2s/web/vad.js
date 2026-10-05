import {Endpoint} from './endpoint.js';

const asset = name => window.amharicVADAssets?.[name] || '/static/vendor/' + name;
let modelPromise;
async function loadModel() {
  if (!modelPromise) modelPromise = (async () => {
    if (!window.ort) await new Promise((resolve, reject) => {
      const script = document.createElement('script'); script.src = asset('ort.wasm.min.js');
      script.onload = resolve; script.onerror = () => reject(new Error('Could not load the local speech detector.'));
      document.head.appendChild(script);
    });
    const ort = window.ort;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = {'ort-wasm-simd.wasm': asset('ort-wasm-simd.wasm')};
    const response = await fetch(asset('silero_vad_v5.onnx'));
    if (!response.ok) throw new Error('Speech model asset is unavailable.');
    return ort.InferenceSession.create(await response.arrayBuffer(), {executionProviders: ['wasm']});
  })().catch(exc => { modelPromise = null; throw exc; });
  return modelPromise;
}

// One detector per page, serialized inference; all microphone tracks stop on dispose.
export class SpeechRecorder {
  constructor(onProgress, onDone, onError) {
    this.onProgress = onProgress; this.onDone = onDone; this.onError = onError;
    this.endpoint = new Endpoint({silenceMs: 800}); this.disposed = false;
    this.queue = []; this.processing = false;
  }
  async start() {
    this.context = new AudioContext({sampleRate: 16000});
    await this.context.resume();
    if (this.context.sampleRate !== 16000) throw new Error('This browser cannot capture at 16 kHz. Use Chrome.');
    this.session = await loadModel();
    if (this.disposed) return;
    this.state = new ort.Tensor('float32', new Float32Array(256), [2, 1, 128]);
    this.sr = new ort.Tensor('int64', BigInt64Array.from([16000n]), [1]);
    this.tail = new Float32Array(64);
    this.stream = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}, video: false});
    if (this.disposed) { this.dispose(); return; }
    await this.context.audioWorklet.addModule(window.amharicVADCaptureURL || '/static/vad-capture.js');
    if (this.disposed) return;
    this.clockOrigin = performance.now() - this.context.currentTime * 1000;
    this.node = new AudioWorkletNode(this.context, 'vad-capture');
    this.source = this.context.createMediaStreamSource(this.stream);
    this.node.port.onmessage = ({data}) => {
      if (this.disposed) return;
      if (this.queue.length >= 32) { this.fail(new Error('Speech detection is falling behind. Close other browser tabs and restart the conversation.')); return; }
      this.queue.push(data); this.drain();
    };
    this.node.onprocessorerror = () => this.fail(new Error('Microphone capture failed. Restart the conversation.'));
    for (const track of this.stream.getTracks()) track.onended = () => this.fail(new Error('Microphone disconnected.'));
    this.source.connect(this.node); this.node.connect(this.context.destination);
  }
  async drain() {
    if (this.processing || this.disposed) return;
    this.processing = true;
    try {
      while (this.queue.length && !this.disposed) {
        const {frame, time} = this.queue.shift();
        const input = new Float32Array(576); input.set(this.tail); input.set(frame, 64);
        this.tail = frame.slice(-64);
        const tensor = new ort.Tensor('float32', input, [1, 576]);
        let output;
        try { output = await this.session.run({input: tensor, state: this.state, sr: this.sr}); }
        finally { tensor.dispose(); }
        this.state.dispose(); this.state = output.stateN;
        const score = output.output.data[0]; output.output.dispose();
        if (this.disposed) break;
        let peak = 0; for (const x of frame) peak = Math.max(peak, Math.abs(x));
        const result = this.endpoint.push(frame, score, this.clockOrigin + time * 1000);
        this.onProgress({peak, seconds: this.endpoint.frames.length * .032, speech: this.endpoint.active});
        if (result?.type === 'misfire') this.onProgress({peak: 0, seconds: 0, speech: false, discarded: true});
        if (result?.type === 'too_long') throw new Error('Question exceeded 30 seconds. Please restart and ask a shorter question; no truncated audio was sent.');
        if (result?.type === 'speech') {
          this.dispose(); this.onDone(result.audio, 16000, result.lastVoice); break;
        }
      }
    } catch (exc) { this.fail(exc); }
    finally {
      this.processing = false;
      if (this.disposed) { this.state?.dispose(); this.state = null; this.sr?.dispose(); this.sr = null; }
    }
  }
  fail(exc) { if (this.disposed) return; this.dispose(); this.onError(exc); }
  dispose() {
    this.disposed = true; this.queue = [];
    this.stream?.getTracks().forEach(t => t.stop());
    this.source?.disconnect(); this.node?.disconnect(); this.node?.port.close();
    if (this.context?.state !== 'closed') this.context?.close().catch(() => {});
    if (!this.processing) { this.state?.dispose(); this.state = null; this.sr?.dispose(); this.sr = null; }
  }
}
