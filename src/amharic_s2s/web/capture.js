// Captures PCM at the actual AudioContext rate. Never sends microphone data to a server.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Float32Array(Math.floor(sampleRate * 30));
    this.length = 0;
    this.done = false;
    this.nextProgress = 0;
    this.port.onmessage = ({data}) => { if (data === 'stop') this.finish(); };
  }
  finish() {
    if (this.done) return;
    this.done = true;
    const samples = this.samples.slice(0, this.length);
    this.port.postMessage({type: 'done', samples}, [samples.buffer]);
    this.samples = null;
  }
  process(inputs) {
    if (this.done) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    const count = Math.min(channels[0].length, this.samples.length - this.length);
    let peak = 0;
    for (let i = 0; i < count; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      this.samples[this.length++] = value;
      peak = Math.max(peak, Math.abs(value));
    }
    if (this.length >= this.nextProgress) {
      this.port.postMessage({type: 'progress', seconds: this.length / sampleRate, peak});
      this.nextProgress = this.length + sampleRate / 10;
    }
    if (this.length === this.samples.length) this.finish();
    return !this.done;
  }
}
registerProcessor('pcm-capture', Capture);
