export function wavPCM16(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, i) => {
    const value = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  });
  return new Blob([buffer], {type: 'audio/wav'});
}

export function base64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = () => reject(new Error('Could not read the recording.'));
    reader.readAsDataURL(blob);
  });
}

export class Recorder {
  constructor(onProgress, onDone, onError) {
    this.onProgress = onProgress; this.onDone = onDone; this.onError = onError;
    this.disposed = false;
  }
  async start() {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
      throw new Error('Microphone access needs HTTPS or localhost in a supported browser.');
    this.context = new AudioContext();
    await this.context.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}, video: false});
    if (this.disposed) { this.dispose(); return; }
    if (this.context.sampleRate < 8000 || this.context.sampleRate > 96000)
      throw new Error('This microphone sample rate is unsupported. Use a rate between 8 and 96 kHz.');
    await this.context.audioWorklet.addModule(window.amharicCaptureURL || '/static/capture.js');
    if (this.disposed) return;
    this.node = new AudioWorkletNode(this.context, 'pcm-capture');
    this.node.onprocessorerror = () => this.fail('Microphone capture failed. Record the question again.');
    this.node.port.onmessage = ({data}) => {
      if (this.disposed) return;
      if (data.type === 'progress') this.onProgress(data);
      if (data.type === 'done') {
        const rate = this.context.sampleRate;
        this.dispose();
        this.onDone(data.samples, rate);
      }
    };
    this.source = this.context.createMediaStreamSource(this.stream);
    this.source.connect(this.node);
    // The processor outputs silence. Connecting it keeps capture active without mic echo.
    this.node.connect(this.context.destination);
    for (const track of this.stream.getTracks())
      track.onended = () => { if (!this.disposed) this.fail('Microphone disconnected. Record again.'); };
  }
  stop() {
    this.node?.port.postMessage('stop');
    this.stopTimer = setTimeout(() => this.fail('Recording stopped unexpectedly. Record again.'), 3000);
  }
  fail(message) {
    if (this.disposed) return;
    this.dispose(); this.onError(new Error(message));
  }
  dispose() {
    this.disposed = true;
    clearTimeout(this.stopTimer);
    this.stream?.getTracks().forEach(track => track.stop());
    this.source?.disconnect(); this.node?.disconnect(); this.node?.port.close();
    if (this.context && this.context.state !== 'closed') this.context.close().catch(() => {});
  }
}
