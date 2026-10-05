// Exercise the actual browser poll/listening state machine without microphone hardware.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';
const source=readFileSync(new URL('../src/amharic_s2s/web/app.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'');
function setup(automatic=true, failure='no_speech') {
  const elements=new Map(), timers=new Map(); let started=0, next=0;
  const element=id=>{
    if (!elements.has(id)) elements.set(id,{textContent:'',hidden:false,value:'',
      classList:{toggle(){},remove(){}},pause(){},load(){},removeAttribute(){},
      replaceChildren(){},addEventListener(){}});
    return elements.get(id);
  };
  const ctx=vm.createContext({
    window:{amharicTransport:async path=>({ok:true,status:200,json:async()=>path==='/ready'?
      {ready:true,accepting_requests:true}:{status:'failed',error:failure}}),addEventListener(){}},
    document:{getElementById:element},AbortSignal,performance,URL,
    setInterval(){},setTimeout(fn){const id=++next;timers.set(id,fn);return id;},
    clearTimeout(id){timers.delete(id);},
    SpeechRecorder:class {async start(){started++;}dispose(){this.disposed=true;}}
  });
  vm.runInContext(source,ctx);
  vm.runInContext(`token='key'; activeId='turn'; phase='waiting'; conversation=${automatic}; history=[{role:'user',content:'previous'}];`,ctx);
  return {ctx,elements,timers,started:()=>started,read:s=>vm.runInContext(s,ctx)};
}
test('empty ASR resumes listening once, without replaying or submitting the failed recording',async()=>{
  const h=setup(); await h.read('poll()');
  assert.equal(h.read('conversation'),true);
  assert.equal(h.read('activeId'),null);
  assert.equal(h.read('draft'),null);
  assert.equal(h.read('history.length'),1);
  assert.match(h.elements.get('vad-note').textContent,/Please repeat/);
  assert.equal(h.timers.size,1);
  for(const fn of h.timers.values())fn();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.started(),1);
  assert.equal(h.read('phase'),'listening');
});
test('ending conversation before recovery prevents reopening microphone',async()=>{
  const h=setup();await h.read('poll()');
  const queued=[...h.timers.values()];
  await h.elements.get('end-conversation').onclick();
  queued.forEach(fn=>fn());await Promise.resolve();
  assert.equal(h.started(),0);assert.equal(h.read('conversation'),false);
});
test('manual mode waits for a new recording',async()=>{
  const h=setup(false);await h.read('poll()');
  assert.equal(h.timers.size,0);assert.equal(h.read('phase'),'idle');
  assert.match(h.elements.get('vad-note').textContent,/Speech was not recognized/);
});
test('genuine inference failures still stop automatic conversation',async()=>{
  const h=setup(true,'inference_failed');await h.read('poll()');
  assert.equal(h.read('conversation'),false);assert.equal(h.timers.size,0);
  assert.match(h.elements.get('error').textContent,/inference_failed/);
});
