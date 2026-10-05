"""HTTP contract and lifecycle tests; never download or run model weights."""
import asyncio
import base64
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time
import unittest

import httpx
import numpy as np
import soundfile as sf

from amharic_s2s.api import Settings, create_app
from amharic_s2s.runtime import Limits, VoiceResponse, VoiceRuntime


def payload(**changes):
    out = BytesIO()
    sf.write(out, np.full(8000, .2), 8000, format='WAV', subtype='PCM_16')
    result = {'audio_base64': base64.b64encode(out.getvalue()).decode()}
    result.update(changes)
    return result


class FakeRuntime:
    def __init__(self):
        self.limits = Limits()
        self.state = 'new'
        self.started = threading.Event()
        self.load = threading.Event()
        self.load.set()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.cancel_calls = 0
        self.calls = []
        self.failure = None
        self.start_failure = False

    def start(self, cases):
        self.state = 'starting'
        self.started.set()
        assert self.load.wait(5)
        if self.start_failure:
            raise RuntimeError('private startup path')
        self.state = 'ready'

    def status(self):
        return {'state': self.state, 'ready': self.state == 'ready', 'busy': False}

    def respond(self, request):
        self.calls.append(request)
        self.entered.set()
        assert self.release.wait(5)
        if self.failure:
            raise self.failure
        return VoiceResponse('runtime-id', 'ጥያቄ', 'መልስ።', np.full(8000, .2), 8000,
                             {'asr_to_tts_start_s': 1.7}, {'timing_status': 'warmed'}, {})

    def cancel(self):
        self.cancel_calls += 1

    def close(self):
        self.state = 'closed'


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(Path(self.tmp.name), 'x' * 40)
        self.runtime = FakeRuntime()
        self.headers = {'Authorization': 'Bearer ' + self.settings.token}
        self.sessions = []

    async def open(self, settings=None, runtime=None, wait=True):
        r = runtime or self.runtime
        app = create_app(settings or self.settings, runtime_factory=lambda _: r,
                         warmup_factory=lambda _: [])
        life = app.router.lifespan_context(app)
        await life.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                   headers=self.headers)
        self.sessions.append((life, client, r))
        if wait:
            await self.until(lambda: r.status()['ready'])
        return app, client

    async def asyncTearDown(self):
        for life, client, runtime in reversed(self.sessions):
            if isinstance(runtime, FakeRuntime):
                runtime.release.set()
                runtime.load.set()
            await client.aclose()
            await life.__aexit__(None, None, None)

    async def until(self, predicate):
        for _ in range(300):
            if predicate():
                return
            await asyncio.sleep(.01)
        self.fail('Timed out waiting for test condition')

    async def terminal(self, client, key):
        for _ in range(300):
            data = (await client.get('/v1/turns/' + key)).json()
            if data['status'] in ('completed', 'failed', 'cancelled'):
                return data
            await asyncio.sleep(.01)
        self.fail('Turn did not complete')

    async def submit(self, client):
        response = await client.post('/v1/turns', json=payload())
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    async def test_health_while_loading_and_authentication(self):
        self.runtime.load.clear()
        app, client = await self.open(wait=False)
        self.assertEqual((await client.get('/health')).status_code, 200)
        self.assertEqual((await client.get('/ready')).status_code, 503)
        self.assertEqual((await client.post('/v1/turns', json=payload())).status_code, 503)
        for method, path in [('GET', '/ready'), ('POST', '/v1/turns'),
                             ('GET', '/v1/turns/guess'), ('GET', '/v1/turns/guess/audio'),
                             ('POST', '/v1/turns/guess/cancel'), ('DELETE', '/v1/turns/guess')]:
            response = await client.request(method, path, headers={'Authorization': 'Bearer wrong'})
            self.assertEqual(response.status_code, 401)
        self.runtime.load.set()
        await self.until(lambda: self.runtime.state == 'ready')
        self.assertEqual((await client.get('/ready')).status_code, 200)

    async def test_audio_roundtrip_context_and_no_files(self):
        app, client = await self.open()
        response = await client.post('/v1/turns', json=payload(context='117 ብር', grounded=True))
        self.assertEqual(response.status_code, 202)
        key = response.json()['id']
        result = await self.terminal(client, key)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['result']['answer'], 'መልስ።')
        self.assertEqual(result['result']['timings']['asr_to_tts_start_s'], 1.7)
        self.assertEqual(self.runtime.calls[0].context, '117 ብር')
        self.assertTrue(self.runtime.calls[0].grounded)
        wav = await client.get(result['result']['audio_url'])
        self.assertEqual(wav.headers['cache-control'], 'no-store')
        samples, rate = sf.read(BytesIO(wav.content))
        self.assertEqual(rate, 8000)
        np.testing.assert_allclose(samples, .2, atol=1 / 32768)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])
        self.assertEqual((await client.delete('/v1/turns/' + key)).status_code, 204)
        self.assertEqual((await client.get('/v1/turns/' + key)).status_code, 404)

    async def test_busy_cancel_discards_late_result_and_stale_cancel_is_harmless(self):
        self.runtime.release.clear()
        app, client = await self.open()
        key = await self.submit(client)
        await self.until(self.runtime.entered.is_set)
        self.assertEqual((await client.post('/v1/turns', json=payload())).status_code, 409)
        self.assertEqual((await client.get(f'/v1/turns/{key}/audio')).status_code, 409)
        self.assertEqual((await client.delete(f'/v1/turns/{key}')).status_code, 409)
        self.assertEqual((await client.post(f'/v1/turns/{key}/cancel')).json()['status'], 'cancelling')
        self.assertEqual((await client.post('/v1/turns', json=payload())).status_code, 409)
        self.runtime.release.set()
        self.assertEqual((await self.terminal(client, key))['status'], 'cancelled')
        self.runtime.release.clear()
        key2 = await self.submit(client)
        count = self.runtime.cancel_calls
        await client.post(f'/v1/turns/{key}/cancel')
        self.assertEqual(self.runtime.cancel_calls, count)
        self.runtime.release.set()
        self.assertEqual((await self.terminal(client, key2))['status'], 'completed')

    async def test_bad_input_and_size_limits_release_admission(self):
        app, client = await self.open(replace(self.settings, max_body_bytes=30000))
        for data in (payload(audio_base64='not base64'), payload(context='x' * 4097),
                     payload(grounded='yes'), {'extra': 1}, ['not an object']):
            self.assertEqual((await client.post('/v1/turns', json=data)).status_code, 422)
        oversized = await client.post('/v1/turns', content=b'x' * 30001,
                                      headers={'Content-Type': 'application/json'})
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual((await client.post('/v1/turns', content=b'x')).status_code, 415)
        self.assertEqual(self.runtime.calls, [])
        await self.terminal(client, await self.submit(client))

    async def test_chunked_upload_cap_and_timeout(self):
        app, client = await self.open(replace(self.settings, max_body_bytes=50, upload_timeout_s=.02))
        async def chunks():
            yield b'x' * 30
            yield b'y' * 30
        self.assertEqual((await client.post('/v1/turns', content=chunks(),
                         headers={'Content-Type': 'application/json'})).status_code, 413)
        async def slow():
            await asyncio.sleep(.1)
            yield b'{}'
        self.assertEqual((await client.post('/v1/turns', content=slow(),
                         headers={'Content-Type': 'application/json'})).status_code, 408)
        self.assertFalse(app.state.service.uploading)

    async def test_retention_and_capacity(self):
        app, client = await self.open(replace(self.settings, retained_turns=1))
        key = await self.submit(client)
        await self.terminal(client, key)
        key2 = await self.submit(client)
        await self.terminal(client, key2)
        self.assertEqual((await client.get('/v1/turns/' + key)).status_code, 404)
        app.state.service.turns[key2]['finished'] = time.monotonic() - 601
        self.assertEqual((await client.get('/v1/turns/' + key2)).status_code, 404)

    async def test_startup_failure_never_ready(self):
        self.runtime.start_failure = True
        app, client = await self.open(wait=False)
        await self.until(lambda: app.state.service.startup_error)
        response = await client.get('/ready')
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private', response.text)
        self.assertEqual(response.json()['state'], 'failed')

    async def test_model_error_is_sanitized(self):
        self.runtime.failure = RuntimeError('private input /secret/path')
        app, client = await self.open()
        result = await self.terminal(client, await self.submit(client))
        self.assertEqual(result['error'], 'inference_failed')
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('private', str(result))

    async def test_output_size_cap(self):
        app, client = await self.open(replace(self.settings, max_audio_bytes=100))
        result = await self.terminal(client, await self.submit(client))
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['result'])

    async def test_upload_reservation_blocks_parallel_uploads(self):
        import json
        app, client = await self.open()
        entered, release = asyncio.Event(), asyncio.Event()
        async def chunks():
            entered.set()
            await release.wait()
            yield json.dumps(payload()).encode()
        upload = asyncio.create_task(client.post('/v1/turns', content=chunks(),
                                     headers={'Content-Type': 'application/json'}))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            self.assertEqual((await client.post('/v1/turns', json=payload())).status_code, 409)
            self.assertFalse((await client.get('/ready')).json()['accepting_requests'])
        finally:
            release.set()
        response = await upload
        self.assertEqual(response.status_code, 202)
        await self.terminal(client, response.json()['id'])

    async def test_shutdown_waits_for_active_inference_and_clears_results(self):
        self.runtime.release.clear()
        app, client = await self.open()
        await self.submit(client)
        await self.until(self.runtime.entered.is_set)
        life, _, _ = self.sessions.pop()
        shutdown = asyncio.create_task(life.__aexit__(None, None, None))
        try:
            await self.until(lambda: app.state.service.closing)
            self.assertFalse(shutdown.done())
            self.assertEqual((await client.post('/v1/turns', json=payload())).status_code, 503)
        finally:
            self.runtime.release.set()
            await shutdown
            await client.aclose()
        self.assertEqual(self.runtime.state, 'closed')
        self.assertEqual(app.state.service.turns, {})

    async def test_real_runtime_lifecycle_with_fake_models(self):
        from test_runtime import Backend, request
        runtime = VoiceRuntime(self.settings.repo, backend_factory=Backend)
        app = create_app(self.settings, runtime_factory=lambda _: runtime,
                         warmup_factory=lambda _: [request()])
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url='http://test', headers=self.headers) as client:
                await self.until(lambda: runtime.status()['ready'])
                result = await self.terminal(client, await self.submit(client))
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['result']['compilation']['timing_status'], 'warmed')
        self.assertEqual(runtime.status()['state'], 'closed')
        self.assertEqual(app.state.service.turns, {})


if __name__ == '__main__':
    unittest.main()
