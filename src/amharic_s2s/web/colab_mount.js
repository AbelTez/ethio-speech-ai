// Executed by google.colab.output.eval_js inside the notebook output frame.
const root = document.createElement('div'); root.innerHTML = bundle.html;
document.body.appendChild(root);
const style = document.createElement('style'); style.textContent = bundle.css; document.head.appendChild(style);
const notice = document.createElement('p');
notice.textContent = 'Colab notebook connection · real model mode unless the yellow preview banner is shown. Browser timing includes notebook transport. Follow the notebook’s saving instructions and export before stopping the runtime.';
notice.className = 'hint'; root.querySelector('header').after(notice);
window.amharicTransport = async (path, options = {}) => {
  options.signal?.throwIfAborted();
  const result = await google.colab.kernel.invokeFunction(bundle.callback,
    [options.method || 'GET', path, options.body ?? null, options.headers?.Authorization || ''], {});
  options.signal?.throwIfAborted();
  const value = result.data['application/json'];
  const body = value.type === 'audio' ? Uint8Array.from(atob(value.body), c => c.charCodeAt(0)) :
    value.body === null ? null : JSON.stringify(value.body);
  return new Response(body, {status: value.status, headers: {'Content-Type': value.type === 'audio' ? 'audio/wav' : 'application/json'}});
};
window.amharicReportTiming = (key, values) => google.colab.kernel.invokeFunction(bundle.timing_callback, [key, values], {});
const moduleURL = code => URL.createObjectURL(new Blob([code], {type: 'text/javascript'}));
window.amharicCaptureURL = moduleURL(bundle.capture);
window.amharicVADAssets = Object.fromEntries(Object.entries(bundle.assets).map(([name, encoded]) => {
  const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  return [name, URL.createObjectURL(new Blob([bytes], {type: name.endsWith('.js') ? 'text/javascript' : 'application/octet-stream'}))];
}));
window.amharicVADCaptureURL = moduleURL(bundle.vad_capture);
const endpointURL = moduleURL(bundle.endpoint);
const vadURL = moduleURL(bundle.vad.replace("'./endpoint.js'", JSON.stringify(endpointURL)));
const recorderURL = moduleURL(bundle.recorder);
const appURL = moduleURL(bundle.app.replace("'./recorder.js'", JSON.stringify(recorderURL)).replace("'./vad.js'", JSON.stringify(vadURL)));
await import(appURL);
document.getElementById('token').value = bundle.token;
document.getElementById('connect').click();
window.addEventListener('pagehide', () => [window.amharicCaptureURL, window.amharicVADCaptureURL, endpointURL, vadURL, ...Object.values(window.amharicVADAssets), recorderURL, appURL].forEach(URL.revokeObjectURL));
return null;
