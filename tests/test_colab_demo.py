import base64
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from amharic_s2s.colab_demo import NotebookBridge, mount_script


class ColabBridgeTests(unittest.TestCase):
    def test_auth_path_and_upload_bounds_before_http(self):
        bridge = NotebookBridge(8765, 'x' * 40)
        with patch('urllib.request.build_opener') as opener:
            self.assertEqual(bridge.request('GET', '/ready')['status'], 401)
            for path in ('http://example.com', '/../../secret', '/v1/turns/../../secret', '/ready?token=secret'):
                self.assertEqual(bridge.request('GET', path, authorization='Bearer ' + bridge.token)['status'], 400)
            self.assertEqual(bridge.request('POST', '/v1/turns', 'a' * (16 * 1024 * 1024 + 1),
                                           'Bearer ' + bridge.token)['status'], 413)
            opener.assert_not_called()

    def test_explicit_export_audio_timings_and_no_tokens(self):
        bridge = NotebookBridge(8765, 'private-token-never-export')
        key, old = 'a' * 32, 'b' * 32
        bridge.turns = [key, old]
        bridge.timing(key, {'play': 4.2, 'download': float('nan'), 'arbitrary': 'private'})
        self.assertEqual(bridge.browser_timings[key], {'play': 4.2})
        def fake(method, path, **kwargs):
            if path == '/ready': return {'status': 200, 'body': {'ready': True}}
            if old in path: return {'status': 404, 'body': {'detail': 'expired'}}
            if path.endswith('/audio'): return {'status': 200, 'body': base64.b64encode(b'RIFF-test').decode()}
            return {'status': 200, 'body': {'id': key, 'status': 'completed', 'result': {
                'transcript': 'ጥያቄ', 'answer': 'መልስ', 'timings': {'asr_to_tts_start_s': 1.5}}}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(bridge, 'request', side_effect=fake):
            archive = bridge.export(Path(tmp) / 'run')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read(f'run/{key}.wav'), b'RIFF-test')
                all_bytes = b''.join(z.read(n) for n in z.namelist())
                self.assertNotIn(bridge.token.encode(), all_bytes)
                self.assertIn(b'expired_or_deleted', all_bytes)
                self.assertIn(b'4.2', all_bytes)
                rows=list(csv.DictReader(io.StringIO(z.read('run/summary.csv').decode())))
                self.assertEqual(rows[0]['browser_speech_end_to_play_s'],'')
                self.assertEqual(rows[0]['asr_to_tts_start_s'],'1.5')

    def test_automatic_archive_survives_server_expiry(self):
        from email.message import Message
        class Reply:
            status=200
            def __init__(self, data, mime):
                self.data=data;self.headers=Message();self.headers['Content-Type']=mime
            def read(self, n):return self.data[:n]
            def __enter__(self):return self
            def __exit__(self,*args):pass
        key='c'*32
        with tempfile.TemporaryDirectory() as tmp:
            bridge=NotebookBridge(8765,'secret',Path(tmp)/'saved');bridge.turns=[key]
            record={'id':key,'status':'completed','result':{'answer':'answer','timings':{}}}
            with patch('urllib.request.build_opener') as opener:
                opener.return_value.open.return_value=Reply(json.dumps(record).encode(),'application/json')
                bridge.request('GET','/v1/turns/'+key,authorization='Bearer secret')
                opener.return_value.open.return_value=Reply(b'RIFF-saved','audio/wav')
                bridge.request('GET','/v1/turns/'+key+'/audio',authorization='Bearer secret')
            bridge.timing(key,{'play':2.0,'discardedSpeech':1})
            def expired(method,path,**kw):
                return {'status':200,'body':{'ready':True}} if path=='/ready' else {'status':404,'body':{}}
            with patch.object(bridge,'request',side_effect=expired):
                archive=bridge.export(Path(tmp)/'export')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('export/'+key+'.wav'),b'RIFF-saved')
                self.assertEqual(json.loads(z.read('export/'+key+'.json'))['status'],'completed')
            self.assertTrue((Path(tmp)/'saved'/(key+'.timing.json')).exists())

    def test_mount_reuses_existing_ui_and_not_external_assets(self):
        web = Path(__file__).resolve().parents[1] / 'src/amharic_s2s/web'
        script = mount_script(web, 'request_callback', 'timing_callback', 'test-token')
        self.assertIn('google.colab.kernel.invokeFunction', script)
        self.assertIn('window.amharicCaptureURL', script)
        self.assertIn('Record', script)
        self.assertNotIn('ngrok', script)


if __name__ == '__main__': unittest.main()
