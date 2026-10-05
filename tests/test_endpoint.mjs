import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Endpoint} from '../src/amharic_s2s/web/endpoint.js';
const frame = () => new Float32Array(512).fill(.1);
test('800 ms endpoint retains onset and does not send before 25 quiet frames', () => {
  const e = new Endpoint();
  for(let i=0;i<20;i++) e.push(frame(),0,i*32);
  for(let i=0;i<10;i++) assert.equal(e.push(frame(),.9,640+i*32),null);
  for(let i=0;i<24;i++) assert.equal(e.push(frame(),0,960+i*32),null);
  const r=e.push(frame(),0,1728);
  assert.equal(r.type,'speech'); assert.equal(r.lastVoice,928);
  assert.equal(r.audio.length,44*512); assert.equal(e.active,false);
});
test('brief pause resets countdown; short noises never submit', () => {
  const e=new Endpoint();
  e.push(frame(),.9,0);
  let r; for(let i=0;i<25;i++) r=e.push(frame(),0,32+i*32);
  assert.equal(r.type,'misfire');
  for(let i=0;i<8;i++) e.push(frame(),.9,i*32);
  for(let i=0;i<10;i++) e.push(frame(),0,256+i*32);
  e.push(frame(),.9,600);
  for(let i=0;i<24;i++) assert.equal(e.push(frame(),0,632+i*32),null);
  assert.equal(e.push(frame(),0,1400).type,'speech');
});
test('long continuous speech is rejected rather than submitted truncated', () => {
  const e=new Endpoint(); let r;
  for(let i=0;i<937;i++) r=e.push(frame(),.9,i*32);
  assert.equal(r.type,'too_long');
});

test('three voiced frames accept a brief word; two remain a false-start candidate', () => {
  for (const [count,expected] of [[3,'speech'],[2,'misfire']]) {
    const e=new Endpoint(); let result;
    for(let i=0;i<count;i++) e.push(frame(),.9,i*32);
    for(let i=0;i<25;i++) result=e.push(frame(),0,(count+i)*32);
    assert.equal(result.type,expected);
  }
});
