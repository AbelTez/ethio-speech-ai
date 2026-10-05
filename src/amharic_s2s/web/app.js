import {SpeechRecorder} from './vad.js';
import {Recorder, wavPCM16, base64} from './recorder.js';
// Colab supplies a notebook callback transport; normal hosting uses same-origin HTTP.
const request = window.amharicTransport || window.fetch.bind(window);

const $ = id => document.getElementById(id);
let token = '', phase = 'idle', available = false, recorder = null, draft = null;
let activeId = null, lastId = null, cancelRequested = false, polling = false;
let previewURL = null, answerURL = null, stoppedAt = null, submittedAt = null, played = false;
let readyPending = false, latestResult = null, browserTimes = {};
let history = [], discardedSpeech = 0;
let documentMode = false;
let conversation = false, speechEndedAt = null, resumeTimer = null;
const busy = () => ['listening', 'playing', 'opening', 'recording', 'stopping', 'uploading', 'waiting', 'cancelling', 'downloading'].includes(phase);
const status = message => { $('status').textContent = message; };
function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }

function controls() {
  $('reset-history').disabled = conversation || busy() || !!activeId;
  $('start-conversation').disabled = conversation || !token || !available || busy() || !!activeId;
  $('end-conversation').disabled = !conversation;
  const reconnect = !!activeId && !polling && !$('retry').hidden;
  $('record').disabled = conversation || !token || !available || busy() || !!activeId;
  $('stop').disabled = phase !== 'recording';
  $('send').disabled = !draft || !available || busy() || !!activeId;
  $('cancel').disabled = !busy() && !activeId && !draft;
  $('cancel').textContent = draft && !busy() && !activeId ? 'Discard recording' : 'Cancel';
  $('connect').disabled = (busy() || !!activeId) && !reconnect;
  $('token').disabled = (busy() || !!activeId) && !reconnect;
  $('disconnect').disabled = conversation || !token || busy() || !!activeId;
  $('context').disabled = documentMode || conversation || busy() || !!activeId;
  $('grounded').disabled = documentMode || conversation || busy() || !!activeId;
  $('forget').disabled = busy() || !!activeId;
}

async function api(path, options = {}) {
  const response = await request(path, {...options, cache: 'no-store',
    headers: {...options.headers, Authorization: `Bearer ${token}`},
    signal: AbortSignal.timeout(45000)});
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const failure = new Error(response.status === 401 ? 'Access key not accepted. Reconnect with the correct key.' :
      typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
    failure.status = response.status;
    throw failure;
  }
  return response;
}

async function readiness() {
  if (!token || readyPending) return;
  readyPending = true;
  const checkedToken = token;
  try {
    const response = await request('/ready', {headers: {Authorization: `Bearer ${token}`},
      cache: 'no-store', signal: AbortSignal.timeout(10000)});
    const data = await response.json();
    if (token !== checkedToken) return;
    if (response.status === 401) throw new Error('Access key not accepted. Enter the correct key and connect.');
    if (!response.ok && response.status !== 503) throw new Error('Could not check server readiness.');
    available = !!data.accepting_requests;
    $('connection').textContent = data.ready ? (available ? 'Ready' : 'Busy') :
      data.state === 'failed' ? 'Server unavailable' : 'Loading models';
    $('connection').classList.toggle('ready', available);
    $('demo').hidden = !data.demo_mode;
    const kb = data.capacity?.find(row => row.stage === 'knowledge_loaded');
    documentMode = kb?.version === 'bank-lexical-v2';
    if (documentMode) { $('context').value = ''; $('grounded').checked = true; }
    $('knowledge-status').hidden = !kb;
    if (kb) $('knowledge-status').textContent = `Bank document search ready · ${kb.documents} source files. Leave manual reference empty to search them automatically.`;
    if (!busy() && !activeId && !draft) {
      if (data.state === 'failed') status('Model startup or inference failed. The server needs to be restarted.');
      else if (!data.ready) status('Models are loading and warming up. This can take several minutes.');
      else if (!available) status('Another question is being processed. Please wait.');
      else if (!latestResult) status('Ready. Start a conversation in Amharic.');
    }
  } catch (exc) {
    if (token !== checkedToken) return;
    available = false;
    $('connection').textContent = 'Connection unavailable';
    $('connection').classList.remove('ready');
    if (!busy() && !activeId) error(exc.message);
  } finally { readyPending = false; controls(); }
}

function releasePreview() {
  $('preview').pause(); $('preview').removeAttribute('src'); $('preview').load(); $('preview').hidden = true;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null; draft = null;
}
function releaseAnswer() {
  $('playback').pause(); $('playback').removeAttribute('src'); $('playback').load();
  if (answerURL) URL.revokeObjectURL(answerURL);
  answerURL = null; latestResult = null; played = false;
  $('response').hidden = true; $('answer').textContent = ''; $('transcript').textContent = '';
  $('sources').textContent = ''; $('answer-basis').textContent = '';
  $('timings').replaceChildren();
}

function captureDone(samples, rate) {
  recorder = null;
  stoppedAt ??= performance.now();
  let peak = 0;
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
  phase = 'idle'; $('level').value = 0;
  $('duration').textContent = `${(samples.length / rate).toFixed(1)} / 30 s`;
  if (samples.length < rate || !Number.isFinite(peak) || peak < 1e-5) {
    error('Record at least one second of audible speech, then stop.'); controls(); return;
  }
  draft = wavPCM16(samples, rate);
  previewURL = URL.createObjectURL(draft);
  $('preview').src = previewURL; $('preview').hidden = false;
  status('Recording ready. Listen if you like, then send your question.'); controls();
}

$('record').onclick = async () => {
  speechEndedAt = null;
  error(); releasePreview(); $('playback').pause(); stoppedAt = null;
  phase = 'opening'; controls(); status('Allow microphone access to begin recording.');
  const current = new Recorder(({seconds, peak}) => {
    $('duration').textContent = `${seconds.toFixed(1)} / 30 s`; $('level').value = peak;
  }, captureDone, exc => { phase = 'idle'; recorder = null; error(exc.message); controls(); });
  recorder = current;
  try {
    await current.start();
    if (current.disposed) return;
    phase = 'recording'; status('Recording… stop when your question is complete.'); controls();
  } catch (exc) {
    const cancelled = current.disposed;
    current.dispose();
    if (recorder !== current) return;
    recorder = null; phase = 'idle';
    if (!cancelled) error(exc.name === 'NotAllowedError' ?
      'Microphone permission was denied. Allow it in your browser settings, then record again.' : exc.message);
    controls();
  }
};
$('stop').onclick = () => {
  stoppedAt = performance.now(); phase = 'stopping'; controls(); status('Finishing recording…'); recorder.stop();
};

function showTimings() {
  const rows = [
    ['ASR start → TTS start (target <3 s)', latestResult?.timings?.asr_to_tts_start_s],
    ['ASR start → audio ready on server', latestResult?.timings?.asr_to_audio_ready_s],
    ['Last detected speech → playback (includes endpoint pause)', browserTimes.speechEndToPlay],
    ['Document search', latestResult?.timings?.retrieval_search_s],
    ['Send → audio downloaded', browserTimes.download],
    ['Send → playback started', browserTimes.play],
    ['Stop → playback started (includes review time)', browserTimes.stopToPlay]
  ];
  $('timings').replaceChildren();
  for (const [name, value] of rows) {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = name; dd.textContent = Number.isFinite(value) ? `${value.toFixed(3)} s` : 'Not measured';
    $('timings').append(dt, dd);
  }
  if (lastId && window.amharicReportTiming)
    window.amharicReportTiming(lastId, {...browserTimes}).catch(() => {});
}
$('playback').addEventListener('playing', () => {
  if (played || !submittedAt || !latestResult || !answerURL) return;
  played = true;
  const now = performance.now();
  browserTimes.play = (now - submittedAt) / 1000;
  if (speechEndedAt !== null) browserTimes.speechEndToPlay = (now - speechEndedAt) / 1000;
  if (stoppedAt !== null) browserTimes.stopToPlay = (now - stoppedAt) / 1000;
  showTimings(); $('playback-note').textContent = 'Playing the answer. Use the audio controls to pause or replay.';
});

async function cancelServer() {
  if (!activeId) return;
  try { await api(`/v1/turns/${activeId}/cancel`, {method: 'POST'}); }
  catch (exc) {
    if (exc.status !== 404) throw exc;
    activeId = null; phase = 'idle'; status('Turn has expired. You can record another question.'); controls();
  }
}

async function poll() {
  if (polling || !activeId) return;
  let repeatSpeech = false;
  polling = true; $('retry').hidden = true;
  try {
    while (activeId) {
      const key = activeId;
      const turn = await (await api(`/v1/turns/${key}`)).json();
      if (turn.status === 'running' || turn.status === 'cancelling') {
        await new Promise(resolve => setTimeout(resolve, 300)); continue;
      }
      if (turn.status === 'cancelled' || cancelRequested) {
        conversation = false;
        activeId = null; phase = 'idle'; status('Cancelled. No answer will play.'); break;
      }
      if (turn.status === 'failed') {
        activeId = null; phase = 'idle';
        if (turn.error === 'no_speech') {
          releasePreview();
          const message = 'Speech was not recognized. Please repeat after Listening, or record again in manual mode.';
          $('vad-note').textContent = message;
          status(message);
          repeatSpeech = true;
          break;
        }
        throw new Error(`The answer could not be completed (${turn.error || 'inference_failed'}).`);
      }
      if (turn.status !== 'completed') throw new Error('Unrecognized server status. Check again.');
      phase = 'downloading'; status('Answer ready. Downloading audio…'); controls();
      // Construct the same-origin path from our turn ID; never fetch a model-provided URL.
      const blob = await (await api(`/v1/turns/${key}/audio`)).blob();
      if (cancelRequested) { activeId = null; phase = 'idle'; status('Cancelled. No answer will play.'); break; }
      browserTimes.download = (performance.now() - submittedAt) / 1000;
      releaseAnswer(); latestResult = turn.result; lastId = key;
      $('transcript').textContent = latestResult.transcript;
      $('answer').textContent = latestResult.answer;
      const basis=latestResult.details?.grounding?.route;
      $('answer-basis').textContent = basis === 'social_template' ? 'Prepared Amharic greeting or acknowledgment.' :
        basis === 'no_retrieval_match' ? 'No sufficiently matching document passage was found.' :
        basis === 'retrieval_budget_abstention' ? 'The matching passage did not fit the current prompt budget.' :
        basis === 'documents' ? 'Answer generated from the passage shown below. Check its relevance and accuracy.' : 'Answer used the current question and supplied context.';
      $('sources').textContent = (latestResult.details?.retrieval?.selected || []).map(c =>
        c.sources.map(s => s.source + ' · block ' + s.paragraph).join('\n')+'\n'+c.title+'\n'+c.content).join('\n\n');
      history = [...history, {role:'user', content:latestResult.transcript.slice(0,2000)},
        {role:'assistant', content:latestResult.answer.slice(0,2000)}].slice(-4);
      const timingStatus = latestResult.compilation?.timing_status;
      $('compile-note').textContent = timingStatus === 'demo_only' ? 'Preview only. No model timing was measured.' :
        timingStatus === 'includes_compilation' ? 'This turn includes compilation and is not a warmed latency measurement.' :
        'Server timings describe this turn only; answer and speech quality still need human review.';
      answerURL = URL.createObjectURL(blob); $('playback').src = answerURL;
      $('response').hidden = false; showTimings();
      activeId = null; phase = conversation ? 'playing' : 'idle';
      status(conversation ? 'Speaking… microphone paused.' : 'Answer received. You can record another question.');
      $('playback-note').textContent = 'Starting playback…';
      try { await $('playback').play(); }
      catch { $('playback-note').textContent = 'Press Play to hear the answer. Your browser did not start playback automatically.'; }
      break;
    }
  } catch (exc) {
    conversation = false;
    error(exc.message);
    if (exc.status === 404) { activeId = null; phase = 'idle'; status('Turn expired. Record a new question.'); }
    if (activeId) {
      status('Turn status is uncertain. Check again or cancel; no new question has been sent.');
      $('retry').hidden = false;
    }
  } finally {
    polling = false; controls(); await readiness();
    if (repeatSpeech && conversation && available) {
      clearTimeout(resumeTimer);
      resumeTimer = setTimeout(() => {
        if (conversation && available && !activeId && phase === 'idle') listen();
      }, 300);
    }
  }
}

$('send').onclick = async () => {
  if (!draft || activeId) return;
  error(); $('preview').pause(); releaseAnswer(); lastId = null;
  phase = 'uploading'; cancelRequested = false; browserTimes = {discardedSpeech}; discardedSpeech = 0; submittedAt = performance.now();
  status('Sending your question…'); controls();
  try {
    const encoded = await base64(draft);
    if (cancelRequested) { phase = 'idle'; status('Cancelled before sending.'); return; }
    const turn = await (await api('/v1/turns', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({audio_base64: encoded, context: $('context').value, grounded: $('grounded').checked, history})})).json();
    activeId = turn.id; releasePreview();
    phase = cancelRequested ? 'cancelling' : 'waiting';
    status(cancelRequested ? 'Cancelling… waiting for the server to stop.' : 'Processing your question…');
    if (cancelRequested) await cancelServer();
    await poll();
  } catch (exc) {
    conversation = false;
    error(exc.message);
    if (activeId) { $('retry').hidden = false; status('Check this turn again or retry cancellation.'); }
    else { phase = 'idle'; available = false; status('Submission was not confirmed. Check server readiness before trying again.'); }
  } finally { controls(); await readiness(); }
};

$('cancel').onclick = async () => {
  conversation = false; clearTimeout(resumeTimer);
  error(); $('playback').pause();
  if (phase === 'playing' && !activeId) { phase = 'idle'; status('Conversation ended.'); controls(); return; }
  if (recorder) {
    recorder.dispose(); recorder = null; phase = 'idle'; $('level').value = 0;
    status('Recording discarded.'); controls(); return;
  }
  if (!activeId && !busy()) { releasePreview(); status('Recording discarded.'); controls(); return; }
  cancelRequested = true; phase = 'cancelling'; status('Cancelling… waiting for the server to stop.'); controls();
  try { await cancelServer(); if (activeId) await poll(); }
  catch (exc) { error(exc.message); $('retry').hidden = false; }
};
$('retry').onclick = async () => {
  error();
  try { if (cancelRequested) await cancelServer(); await poll(); }
  catch (exc) { error(exc.message); }
};
$('forget').onclick = async () => {
  error();
  try {
    if (lastId) {
      const response = await request(`/v1/turns/${lastId}`, {method: 'DELETE',
        headers: {Authorization: `Bearer ${token}`}, signal: AbortSignal.timeout(10000)});
      if (!response.ok && response.status !== 404) throw new Error('Could not remove the answer from the server. Try again.');
    }
    lastId = null; history = []; releaseAnswer(); status('Answer and recent history cleared.');
  } catch (exc) { error(exc.message); }
};
$('connect-form').onsubmit = async event => {
  event.preventDefault(); error(); token = $('token').value.trim(); $('token').value = '';
  if (!token) return;
  status('Connecting…'); controls(); await readiness();
  if (activeId) {
    try { if (cancelRequested) await cancelServer(); await poll(); }
    catch (exc) { error(exc.message); controls(); }
  }
};
$('disconnect').onclick = () => {
  history = []; token = ''; available = false; releasePreview(); releaseAnswer(); lastId = null;
  $('token').value = ''; $('context').value = ''; $('grounded').checked = false;
  $('connection').textContent = 'Not connected'; $('connection').classList.remove('ready');
  $('demo').hidden = true; error(); status('Disconnected.'); controls();
};
window.addEventListener('pagehide', () => {
  history = []; conversation = false; clearTimeout(resumeTimer);
  recorder?.dispose(); $('playback').pause();
  if (activeId && token) request(`/v1/turns/${activeId}/cancel`, {method: 'POST', keepalive: true,
    headers: {Authorization: `Bearer ${token}`}}).catch(() => {});
  token = ''; activeId = null; available = false; phase = 'idle';
  $('token').value = ''; $('context').value = ''; $('grounded').checked = false;
  releasePreview(); releaseAnswer();
  $('connection').textContent = 'Not connected'; $('connection').classList.remove('ready');
  status('Connect to begin.'); controls();
});
setInterval(readiness, 2500);
controls();

// Automatic half-duplex loop. No microphone capture during inference/playback.
async function listen() {
  if (!conversation) return;
  error(); releasePreview(); stoppedAt = null; speechEndedAt = null;
  phase = 'opening'; controls(); status('Preparing speech detection…');
  const current = new SpeechRecorder(({seconds, peak, speech, discarded}) => {
    $('duration').textContent = `${seconds.toFixed(1)} / 30 s`; $('level').value = peak;
    if (discarded) { discardedSpeech++; $('vad-note').textContent = `Brief sounds discarded since the last submission: ${discardedSpeech}. Please repeat if your word was missed.`; }
    status(speech ? 'Listening… pause when your question is complete.' :
      discarded ? 'Brief sound discarded. Please repeat after Listening.' : 'Listening… ask your question.');
  }, (samples, rate, lastVoice) => {
    if (!conversation || recorder !== current) return;
    recorder = null; stoppedAt = performance.now(); speechEndedAt = lastVoice;
    browserTimes = {}; draft = wavPCM16(samples, rate); phase = 'idle'; $('level').value = 0;
    $('send').onclick();
  }, exc => {
    if (recorder !== current) return;
    recorder = null; conversation = false; phase = 'idle'; error(exc.message);
    status('Conversation stopped. Restart when ready.'); controls();
  });
  recorder = current;
  try {
    await current.start();
    if (current.disposed || !conversation) return;
    phase = 'listening'; status('Listening… ask your question.'); controls();
  } catch (exc) {
    const cancelled = current.disposed; current.dispose();
    if (recorder !== current) return;
    recorder = null; conversation = false; phase = 'idle';
    if (!cancelled) error(exc.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow it, then start again.' : exc.message);
    controls();
  }
}
$('start-conversation').onclick = async () => {
  if (conversation || busy() || activeId || !available) return;
  $('playback').pause(); conversation = true; await listen();
};
$('end-conversation').onclick = async () => {
  conversation = false; clearTimeout(resumeTimer);
  recorder?.dispose(); recorder = null; $('level').value = 0; $('playback').pause();
  releasePreview();
  if (activeId || ['uploading', 'waiting', 'downloading', 'cancelling'].includes(phase)) await $('cancel').onclick();
  else { phase = 'idle'; status('Conversation ended.'); controls(); }
};
$('playback').addEventListener('play', () => {
  if (conversation) { recorder?.dispose(); recorder = null; phase = 'playing'; controls(); }
});
$('playback').addEventListener('ended', () => {
  if (!conversation) return;
  status('Answer finished. Resuming listening…');
  clearTimeout(resumeTimer);
  resumeTimer = setTimeout(() => { if (conversation) listen(); }, 300);
});

$('reset-history').onclick = () => { history = []; status('New topic. Previous exchanges cleared.'); };
$('context').addEventListener('input', () => { history = []; });
$('grounded').addEventListener('change', () => { history = []; });
