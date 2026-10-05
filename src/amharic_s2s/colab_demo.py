"""Notebook-only transport to the unchanged localhost API; no public server tunnel."""
import base64
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import secrets
import urllib.error
import urllib.request
import zipfile


class NotebookBridge:
    def __init__(self, port, token, archive_dir=None):
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError('Invalid local port')
        self.base = f'http://127.0.0.1:{port}'
        self.token = token
        self.turns = []
        self.browser_timings = {}
        self.archive_dir = Path(archive_dir) if archive_dir else None
        self.archive_warning = None
        if self.archive_dir: self.archive_dir.mkdir(parents=True, exist_ok=True)

    def request(self, method, path, body=None, authorization=''):
        if not isinstance(authorization, str) or not secrets.compare_digest(
                authorization.encode(), ('Bearer ' + self.token).encode()):
            return {'status': 401, 'type': 'json', 'body': {'detail': 'Authentication required.'}}
        valid = isinstance(path, str) and re.fullmatch(
            r'/(?:ready|v1/turns(?:/[a-f0-9]{32}(?:/audio|/cancel)?)?)', path)
        if not valid or method not in ('GET', 'POST', 'DELETE'):
            return {'status': 400, 'type': 'json', 'body': {'detail': 'Unsupported notebook request.'}}
        if body is not None and (not isinstance(body, str) or len(body.encode()) > 16 * 1024 * 1024):
            return {'status': 413, 'type': 'json', 'body': {'detail': 'Upload too large.'}}
        req = urllib.request.Request(self.base + path, data=body.encode() if body is not None else None,
                                     headers={'Authorization': authorization, 'Content-Type': 'application/json'}, method=method)
        # Ignore environment proxy settings: traffic must stay on the Colab VM.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open(req, timeout=35)
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError):
            return {'status': 503, 'type': 'json', 'body': {'detail': 'Local model server is unavailable.'}}
        with response:
            data = response.read(17 * 1024 * 1024)
            audio = response.headers.get_content_type() == 'audio/wav'
            result = {'status': response.status, 'type': 'audio' if audio else 'json',
                      'body': base64.b64encode(data).decode() if audio else json.loads(data) if data else None}
        if method == 'POST' and path == '/v1/turns' and result['status'] == 202:
            self.turns.append(result['body']['id'])
        if self.archive_dir and method == 'GET' and result['status'] == 200:
            key = path.split('/')[3] if path.startswith('/v1/turns/') else None
            if key in self.turns:
                if result['type'] == 'audio':
                    self._save(key + '.wav', base64.b64decode(result['body']))
                elif result['body'].get('status') in ('completed', 'failed', 'cancelled'):
                    self._save(key + '.json', json.dumps(result['body'], ensure_ascii=False, indent=2).encode())
        if method == 'DELETE' and result['status'] in (204, 404) and self.archive_dir:
            key = path.split('/')[-1]
            for suffix in ('.json', '.wav', '.timing.json'):
                (self.archive_dir / (key + suffix)).unlink(missing_ok=True)
        return result

    def _save(self, name, content):
        try:
            path = self.archive_dir / name
            temp = path.with_suffix(path.suffix + '.tmp')
            temp.write_bytes(content); temp.replace(path)
        except OSError as exc:
            self.archive_warning = f'Automatic result save failed: {type(exc).__name__}. Export promptly.'
            print(self.archive_warning, flush=True)

    def timing(self, key, values):
        if key not in self.turns or not isinstance(values, dict):
            return False
        clean = {k: v for k, v in values.items() if k in ('download', 'play', 'stopToPlay', 'speechEndToPlay', 'discardedSpeech')
                 and type(v) in (int, float) and math.isfinite(v) and v >= 0}
        self.browser_timings.setdefault(key, {}).update(clean)
        if self.archive_dir:
            self._save(key + '.timing.json', json.dumps(self.browser_timings[key]).encode())
        return True

    def export(self, destination):
        """Explicit user action: save available results/audio, never keys or raw uploads."""
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=False)
        rows = []
        for key in self.turns:
            response = self.request('GET', f'/v1/turns/{key}', authorization='Bearer ' + self.token)
            record = response['body']
            if response['status'] != 200:
                saved = self.archive_dir / (key + '.json') if self.archive_dir else None
                record = json.loads(saved.read_text()) if saved and saved.exists() else {
                    'id': key, 'status': 'unavailable', 'error': 'expired_or_deleted'}
            record['browser_timings'] = self.browser_timings.get(key, {})
            (root / (key + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            result = record.get('result') or {}
            metrics = result.get('timings') or {}
            rows.append({'id': key, 'status': record['status'], 'transcript': result.get('transcript', ''),
                         'error':record.get('error'),
                         'retrieval_search_s':metrics.get('retrieval_search_s'),
                         'answer': result.get('answer', ''), 'asr_to_tts_start_s': metrics.get('asr_to_tts_start_s'),
                         'asr_to_audio_ready_s': metrics.get('asr_to_audio_ready_s'),
                         'asr_total_s': metrics.get('asr_total_s'),
                         'llm_full_answer_s': metrics.get('llm_full_answer_s'),
                         'tts_generation_s': metrics.get('tts_generation_s'),
                         'browser_send_to_play_s': record['browser_timings'].get('play'),
                         'browser_speech_end_to_play_s': record['browser_timings'].get('speechEndToPlay'),
                         'discarded_speech_count': record['browser_timings'].get('discardedSpeech'),
                         'llm_engine': result.get('compilation', {}).get('engine', ''),
                         'grounding_route': result.get('details', {}).get('grounding', {}).get('route'),
                         'context_present': result.get('details', {}).get('grounding', {}).get('context_present'),
                         'timing_status': result.get('compilation', {}).get('timing_status', '')})
            if record['status'] == 'completed':
                audio = self.request('GET', f'/v1/turns/{key}/audio', authorization='Bearer ' + self.token)
                if audio['status'] == 200:
                    (root / (key + '.wav')).write_bytes(base64.b64decode(audio['body']))
                elif self.archive_dir and (self.archive_dir / (key + '.wav')).exists():
                    (root / (key + '.wav')).write_bytes((self.archive_dir / (key + '.wav')).read_bytes())
        if rows:
            with (root / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        status = self.request('GET', '/ready', authorization='Bearer ' + self.token)
        (root / 'session.json').write_text(json.dumps({'exported_at': datetime.now(timezone.utc).isoformat(),
            'archive_warning': self.archive_warning,
            'transport': 'Colab notebook callbacks to localhost API', 'status': status['body'],
            'note': 'Browser timings include Colab bridge overhead. User quality review still required.'}, indent=2))
        archive = root.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()): z.write(file, root.name + '/' + file.name)
        return archive


def mount_script(web, callback, timing_callback, token):
    """Bundle the existing UI in an output frame; secrets never written to a file."""
    web = Path(web)
    html = (web / 'index.html').read_text()
    main = html[html.index('<main>'):html.index('</main>') + len('</main>')]
    data = {'html': main, 'css': (web / 'style.css').read_text(),
            'capture': (web / 'capture.js').read_text(), 'recorder': (web / 'recorder.js').read_text(),
            'app': (web / 'app.js').read_text(), 'callback': callback,
            'timing_callback': timing_callback, 'token': token,
            'vad': (web / 'vad.js').read_text(), 'endpoint': (web / 'endpoint.js').read_text(),
            'vad_capture': (web / 'vad-capture.js').read_text(),
            'assets': {name: base64.b64encode((web / 'vendor' / name).read_bytes()).decode()
                       for name in ('ort.wasm.min.js', 'ort-wasm-simd.wasm', 'silero_vad_v5.onnx')}}
    return '(async (bundle) => {\n' + (web / 'colab_mount.js').read_text() + '\n})(' + json.dumps(data) + ')'


def show(bridge, web):
    from google.colab import output
    from IPython.display import JSON
    callback = 'amharic_demo_' + secrets.token_hex(8)
    timing_callback = callback + '_timing'
    output.register_callback(callback, lambda *args: JSON(bridge.request(*args)))
    output.register_callback(timing_callback, lambda *args: JSON({'saved': bridge.timing(*args)}))
    output.eval_js(mount_script(web, callback, timing_callback, bridge.token), timeout_sec=60)
