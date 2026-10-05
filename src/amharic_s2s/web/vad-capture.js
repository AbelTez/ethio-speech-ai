class VADCapture extends AudioWorkletProcessor {
  constructor() { super(); this.frame = new Float32Array(512); this.offset = 0; }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      this.frame[this.offset++] = value;
      if (this.offset === 512) {
        this.port.postMessage({frame: this.frame, time: currentTime}, [this.frame.buffer]);
        this.frame = new Float32Array(512); this.offset = 0;
      }
    }
    return true; // output remains silent
  }
}
registerProcessor('vad-capture', VADCapture);
