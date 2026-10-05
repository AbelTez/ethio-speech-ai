// Silero scores arrive in 512-sample (32 ms) frames at 16 kHz.
// No semantic end-of-turn claim: a pause may still be mid-sentence.
export class Endpoint {
  constructor({silenceMs = 800, preRollMs = 320, minSpeechMs = 96, maxMs = 30000} = {}) {
    this.silenceFrames = Math.ceil(silenceMs / 32);
    this.preFrames = Math.ceil(preRollMs / 32);
    this.minFrames = Math.ceil(minSpeechMs / 32);
    this.maxFrames = Math.floor(maxMs / 32);
    this.reset();
  }
  reset() { this.pre = []; this.frames = []; this.active = false; this.voiced = 0; this.quiet = 0; this.lastVoice = null; }
  push(samples, score, capturedAt) {
    if (!Number.isFinite(score)) throw new Error('Speech detector returned an invalid score.');
    const speech = score >= (this.active ? 0.35 : 0.5);
    if (!this.active) {
      this.pre.push(samples);
      if (this.pre.length > this.preFrames) this.pre.shift();
      if (!speech) return null;
      this.active = true; this.frames = this.pre; this.pre = [];
    } else this.frames.push(samples);
    if (speech) { this.voiced++; this.quiet = 0; this.lastVoice = capturedAt; }
    else this.quiet++;
    if (this.frames.length >= this.maxFrames) { this.reset(); return {type: 'too_long'}; }
    if (this.quiet < this.silenceFrames) return null;
    if (this.voiced < this.minFrames) { this.reset(); return {type: 'misfire'}; }
    const audio = new Float32Array(Math.max(16000, this.frames.length * 512));
    this.frames.forEach((f, i) => audio.set(f, i * 512));
    const result = {type: 'speech', audio, lastVoice: this.lastVoice};
    this.reset(); return result;
  }
}
