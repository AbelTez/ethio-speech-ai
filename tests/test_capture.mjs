import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

function capture(rate) {
  let Capture;
  const messages = [];
  class Processor { constructor() { this.port = {postMessage: data => messages.push(data)}; } }
  vm.runInNewContext(readFileSync(new URL('../src/amharic_s2s/web/capture.js', import.meta.url), 'utf8'), {
    AudioWorkletProcessor: Processor, sampleRate: rate, Float32Array,
    registerProcessor: (_, cls) => { Capture = cls; }
  });
  return {node: new Capture(), messages};
}

test('sample-count cap is exactly 30 seconds even if browser timers are delayed', () => {
  for (const rate of [8000, 44100, 48000, 96000]) {
    const {node, messages} = capture(rate);
    const chunk = new Float32Array(128).fill(.25);
    while (node.process([[chunk]])) { /* synchronous render blocks */ }
    const done = messages.filter(m => m.type === 'done');
    assert.equal(done.length, 1);
    assert.equal(done[0].samples.length, rate * 30);
    assert.equal(done[0].samples.at(-1), .25);
    node.port.onmessage({data: 'stop'});
    assert.equal(messages.filter(m => m.type === 'done').length, 1);
  }
});

test('stop preserves captured frames and stereo is downmixed', () => {
  const {node, messages} = capture(8000);
  node.process([[new Float32Array(128).fill(.5), new Float32Array(128).fill(-.25)]]);
  node.port.onmessage({data: 'stop'});
  assert.equal(messages.at(-1).samples.length, 128);
  assert.equal(messages.at(-1).samples[0], .125);
  assert.equal(node.process([]), false);
});
