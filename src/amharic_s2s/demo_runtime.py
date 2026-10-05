"""Explicit CPU-only interface preview. Fixed text/tone, no recognition or synthesis."""
import threading
from uuid import uuid4

from .runtime import Cancelled, Limits, VoiceResponse


class DemoRuntime:
    def __init__(self, repo):
        self.limits = Limits()
        self.state = 'new'
        self.cancelled = threading.Event()

    def start(self, cases):
        self.state = 'ready'

    def status(self):
        return {'state': self.state, 'ready': self.state == 'ready', 'busy': False}

    def respond(self, request):
        import numpy as np
        self.cancelled.clear()
        if self.cancelled.wait(1):
            raise Cancelled()
        rate = 24000
        t = np.arange(rate, dtype=np.float32) / rate
        wave = .12 * np.sin(2 * np.pi * 440 * t) * np.sin(np.pi * t) ** 2
        return VoiceResponse(uuid4().hex, 'Preview recording received — speech was not transcribed.',
                             'ይህ የማሳያ ሙከራ ነው።', wave, rate, {},
                             {'timing_status': 'demo_only'}, {'demo_only': True, 'quality_reviewed': False})

    def cancel(self):
        self.cancelled.set()

    def close(self):
        self.cancel()
        self.state = 'closed'
