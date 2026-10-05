"""Single-operator HTTP adapter. No model imports or conversation files at import time."""
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
import json
import logging
from pathlib import Path
import secrets
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .runtime import (Busy, Cancelled, InvalidInput, NotReady, NoSpeech, RejectedAnswer,
                      VoiceRequest, VoiceRuntime, recorded_warmup, validate_request)


@dataclass(frozen=True)
class Settings:
    repo: Path
    token: str = field(repr=False)
    max_body_bytes: int = 16 * 1024 * 1024
    max_audio_bytes: int = 8 * 1024 * 1024
    retained_turns: int = 8
    retention_s: float = 600
    upload_timeout_s: float = 30
    demo_mode: bool = False

    def __post_init__(self):
        if not isinstance(self.token, str) or len(self.token) < 32 or not self.token.isascii():
            raise ValueError('Set a random ASCII API token of at least 32 characters.')
        if min(self.max_body_bytes, self.max_audio_bytes, self.retained_turns,
               self.retention_s, self.upload_timeout_s) <= 0:
            raise ValueError('API limits must be positive.')


def decode_request(body, limits):
    """Inspect WAV metadata before allocating decoded samples; no temporary files."""
    import soundfile as sf
    try:
        data = json.loads(body)
        if not isinstance(data, dict) or set(data) - {'audio_base64', 'context', 'grounded', 'history'}:
            raise ValueError()
        encoded = data['audio_base64']
        context, grounded = data.get('context', ''), data.get('grounded', False)
        if (not isinstance(encoded, str) or not isinstance(context, str)
                or len(context) > limits.max_context_chars or type(grounded) is not bool):
            raise ValueError()
        wav = base64.b64decode(encoded, validate=True)
        with sf.SoundFile(BytesIO(wav)) as f:
            if (f.format != 'WAV' or f.channels not in (1, 2)
                    or not limits.min_sample_rate <= f.samplerate <= limits.max_sample_rate
                    or not limits.min_audio_s <= f.frames / f.samplerate <= limits.max_audio_s):
                raise ValueError()
            rate = f.samplerate
            audio = f.read(dtype='float32')
        return validate_request(VoiceRequest(audio, rate, context, grounded, data.get("history", [])), limits)
    except (ValueError, TypeError, KeyError, RuntimeError, RecursionError, binascii.Error, InvalidInput):
        raise InvalidInput('Provide valid base64 WAV audio, 1–30 seconds, mono/stereo, '
                           '8–96 kHz, with optional context and boolean grounded.') from None


def encode_response(result, max_bytes):
    import numpy as np
    import soundfile as sf
    # Browser-compatible PCM16; refuse clipping rather than altering loudness.
    if len(result.audio) * 2 + 128 > max_bytes or np.max(np.abs(result.audio)) > 1:
        raise ValueError('Output cannot be represented within audio bounds.')
    out = BytesIO()
    sf.write(out, result.audio, result.sample_rate, format='WAV', subtype='PCM_16')
    return out.getvalue()


class TurnService:
    """State is owned by the event loop. A reservation covers upload through completion."""
    def __init__(self, settings, runtime):
        self.settings, self.runtime = settings, runtime
        self.turns = {}
        self.active = None
        self.uploading = False
        self.closing = False
        self.startup_error = False
        self.task = None

    def prune(self):
        now = time.monotonic()
        for key, turn in list(self.turns.items()):
            if turn['finished'] is not None and now - turn['finished'] >= self.settings.retention_s:
                del self.turns[key]

    def get(self, key):
        self.prune()
        if key not in self.turns:
            raise HTTPException(404, 'Turn not found or expired.')
        return self.turns[key]

    def reserve(self):
        if self.closing or self.startup_error or not self.runtime.status()['ready']:
            raise HTTPException(503, 'Runtime is not ready.', headers={'Retry-After': '5'})
        if self.uploading or self.active is not None:
            raise HTTPException(409, 'Another turn is active; no request was queued.')
        self.uploading = True

    def submit(self, request):
        self.prune()
        while len(self.turns) >= self.settings.retained_turns:
            del self.turns[next(iter(self.turns))]
        key = secrets.token_hex(16)
        self.turns[key] = {'id': key, 'status': 'running', 'finished': None,
                           'cancelled': False, 'result': None, 'audio': None, 'error': None}
        self.active = key
        self.task = asyncio.create_task(self.run(key, request))
        return key

    async def run(self, key, request):
        turn = self.turns[key]
        try:
            if turn['cancelled']:
                raise Cancelled()
            result = await asyncio.to_thread(self.runtime.respond, request)
            if turn['cancelled']:
                raise Cancelled()
            audio = await asyncio.to_thread(encode_response, result, self.settings.max_audio_bytes)
            if turn['cancelled']:
                raise Cancelled()
            turn.update(status='completed', audio=audio, result={
                'transcript': result.transcript, 'answer': result.answer,
                'sample_rate': result.sample_rate, 'timings': result.timings,
                'compilation': result.compilation, 'details': result.details,
                'audio_url': f'/v1/turns/{key}/audio'})
        except Cancelled:
            turn['status'] = 'cancelled'
        except Exception as exc:
            # Never expose model tracebacks, filesystem paths or prompt contents.
            code = ('invalid_input' if isinstance(exc, InvalidInput) else
                    'no_speech' if isinstance(exc, NoSpeech) else
                    'answer_rejected' if isinstance(exc, RejectedAnswer) else
                    'busy' if isinstance(exc, Busy) else
                    'not_ready' if isinstance(exc, NotReady) else 'inference_failed')
            turn.update(status='failed', error=code)
        finally:
            if turn['cancelled']:
                turn.update(status='cancelled', audio=None, result=None, error=None)
            turn['finished'] = time.monotonic()
            self.active = None

    def cancel(self, key):
        turn = self.get(key)
        if self.active == key:
            turn.update(cancelled=True, status='cancelling')
            self.runtime.cancel()
        return turn


def create_app(settings, *, runtime_factory=VoiceRuntime, warmup_factory=recorded_warmup):
    @asynccontextmanager
    async def lifespan(app):
        runtime = runtime_factory(settings.repo)
        service = TurnService(settings, runtime)
        app.state.service = service

        async def startup():
            try:
                cases = await asyncio.to_thread(warmup_factory, settings.repo)
                if not service.closing:
                    await asyncio.to_thread(runtime.start, cases)
            except Exception as exc:
                service.startup_error = True
                # The resident runtime sanitizes its public error. Retain the underlying
                # startup cause only in server stderr for the operator's local diagnostics.
                cause = exc.__context__ or exc
                logging.getLogger(__name__).error('Resident startup failed',
                    exc_info=(type(cause), cause, cause.__traceback__))

        async def cleanup():
            while True:
                await asyncio.sleep(min(30, settings.retention_s))
                service.prune()

        loading = asyncio.create_task(startup())
        janitor = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            service.closing = True
            if service.active:
                service.cancel(service.active)
            runtime.cancel()
            await loading
            if service.task:
                await service.task
            await asyncio.to_thread(runtime.close)
            janitor.cancel()
            await asyncio.gather(janitor, return_exceptions=True)
            service.turns.clear()

    app = FastAPI(title='Amharic speech API', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    assets = Path(__file__).parent / 'web'
    app.mount('/static', StaticFiles(directory=assets), name='static')

    @app.get('/', include_in_schema=False)
    async def browser():
        return FileResponse(assets / 'index.html')

    @app.middleware('http')
    async def private_responses(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Permissions-Policy'] = 'microphone=(self), camera=()'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; connect-src 'self'; "
            "media-src 'self' blob:; img-src 'self' data:; worker-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        return response

    def authorized(request):
        value = request.headers.get('authorization', '')
        expected = 'Bearer ' + settings.token
        if not secrets.compare_digest(value.encode(), expected.encode()):
            raise HTTPException(401, 'Authentication required.',
                                headers={'WWW-Authenticate': 'Bearer'})

    def service(request):
        authorized(request)
        return request.app.state.service

    def view(turn):
        return {key: turn[key] for key in ('id', 'status', 'result', 'error')}

    @app.get('/health')
    async def health():
        return {'status': 'alive'}

    @app.get('/ready')
    async def ready(request: Request):
        s = service(request)
        status = s.runtime.status()
        ok = status['ready'] and not s.closing and not s.startup_error
        status.update(ready=ok, accepting_requests=ok and not s.uploading and s.active is None,
                      demo_mode=settings.demo_mode)
        if s.startup_error:
            status.update(state='failed', last_error='startup_failed')
        return JSONResponse(status, status_code=200 if ok else 503)

    @app.post('/v1/turns', status_code=202)
    async def submit(request: Request):
        s = service(request)
        if request.headers.get('content-type', '').split(';')[0].strip() != 'application/json':
            raise HTTPException(415, 'Use application/json with base64 WAV audio.')
        s.reserve()
        try:
            body = bytearray()

            async def read_body():
                async for chunk in request.stream():
                    if len(body) + len(chunk) > settings.max_body_bytes:
                        raise HTTPException(413, 'Request body exceeds the upload limit.')
                    body.extend(chunk)

            try:
                await asyncio.wait_for(read_body(), settings.upload_timeout_s)
            except asyncio.TimeoutError:
                raise HTTPException(408, 'Upload timed out.') from None
            try:
                voice = await asyncio.to_thread(decode_request, body, s.runtime.limits)
            except InvalidInput as exc:
                raise HTTPException(422, str(exc)) from None
            key = s.submit(voice)
            return JSONResponse(view(s.get(key)), status_code=202,
                                headers={'Location': f'/v1/turns/{key}'})
        finally:
            s.uploading = False

    @app.get('/v1/turns/{key}')
    async def result(key: str, request: Request):
        return view(service(request).get(key))

    @app.get('/v1/turns/{key}/audio')
    async def audio(key: str, request: Request):
        turn = service(request).get(key)
        if turn['status'] != 'completed':
            raise HTTPException(409, 'Audio is not available for this turn.')
        return Response(turn['audio'], media_type='audio/wav')

    @app.post('/v1/turns/{key}/cancel')
    async def cancel(key: str, request: Request):
        return view(service(request).cancel(key))

    @app.delete('/v1/turns/{key}', status_code=204)
    async def forget(key: str, request: Request):
        s = service(request)
        s.get(key)
        if s.active == key:
            raise HTTPException(409, 'Cancel and wait for completion before deleting the turn.')
        del s.turns[key]
        return Response(status_code=204)

    return app


def main():
    import argparse
    import os
    import uvicorn
    parser = argparse.ArgumentParser(description='Serve one resident pipeline on one GPU.')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--demo', action='store_true', help='CPU interface preview with fixed text and a tone; no models.')
    args = parser.parse_args()
    try:
        settings = Settings(args.repo.resolve(), os.environ.get('AMHARIC_API_TOKEN', ''), demo_mode=args.demo)
    except ValueError as exc:
        parser.error(str(exc))
    if args.demo:
        from .demo_runtime import DemoRuntime
        app = create_app(settings, runtime_factory=DemoRuntime, warmup_factory=lambda _: [])
    else:
        app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
