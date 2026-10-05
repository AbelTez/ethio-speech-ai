"""Bounded Gemma experiment: eager dynamic cache versus compiled static-cache decode."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
from functools import wraps
import signal
from statistics import median
import time
import traceback
from pathlib import Path
from uuid import uuid4

from .artifacts import read_json, write_json, object_digest, source_snapshot, environment, safe_name, now, export_run
from .answer_quality import review, quality_pass
from .llm_backend import prompt_messages
from .llm_next import NextLLM, NEXT_PROTOCOL
from .llm_timing import generate_answer

MODES = ('current', 'compiled')
PROTOCOL = {**NEXT_PROTOCOL, 'version': 'gemma-compile-v2', 'repeats': 3, 'seed': 42,
    'cache_length': 512, 'warmup_budget_s': 480,
    'compile': {'fullgraph': True, 'dynamic': None, 'backend': 'inductor', 'mode': 'default'},
    'fix': 'allow automatic generalization of the changing sliding-cache Python integer; keep recompile limit unchanged',
    'method': 'explicit decode-only forward compilation; eager prefill; auto-compilation disabled in both modes',
    'scope': 'LLM alone; same accepted prompts, NF4/FP16 and score guards; no integrated latency claim'}


def schedule(data):
    if len(data['samples']) not in (1, 3):
        raise ValueError('Expected one smoke question or three accepted questions.')
    rows = []
    for repeat in range(3):
        for i, s in enumerate(data['samples']):
            safe_name(s['id'])
            for mode in (MODES if (repeat + i) % 2 == 0 else MODES[::-1]):
                rows.append({**s, 'question_id': s['id'], 'variant': mode, 'repeat': repeat + 1,
                    'id': f'{s["id"]}__{mode}__{repeat+1}',
                    'messages': prompt_messages({'instruction': s['instruction']}, s)})
    if len({r['id'] for r in rows}) != 6 * len(data['samples']):
        raise ValueError('Duplicate question IDs.')
    return rows


@contextmanager
def policy(backend, variant, warm=False):
    """Only cache/compilation policy changes; warmup alone has no generation max_time."""
    original = backend.generation
    cfg = deepcopy(original)
    cfg.disable_compile = True  # We explicitly compile decode, never rely on HF auto-selection.
    cfg.compile_config = None
    cfg.cache_implementation = 'static' if variant == 'compiled' else 'dynamic'
    cfg.max_cache_len = PROTOCOL['cache_length'] if variant == 'compiled' else None
    if warm:
        cfg.max_time = None  # The enclosing warmup deadline includes initial compilation.
    backend.generation = cfg
    try:
        yield
    finally:
        backend.generation = original


class DecodeCompiler:
    """Dispatch outside the compiled graph; no profiling hooks inside model.forward."""
    def __init__(self, backend):
        self.backend = backend
        self.variant = 'current'
        self.calls = 0
        self.cache_types = set()
        self.compiled = None

    @contextmanager
    def installed(self):
        model = self.backend.model
        original = model.forward
        had_local = 'forward' in model.__dict__
        local = model.__dict__.get('forward')
        @wraps(original)
        def forward(*args, **kwargs):
            ids = kwargs.get('input_ids')
            # Prefill has the complete prompt. Subsequent generation forwards have one token.
            if self.variant == 'compiled' and ids is not None and ids.shape[-1] == 1:
                cache = kwargs.get('past_key_values')
                from transformers.cache_utils import StaticCache
                if not isinstance(cache, StaticCache) or not cache.is_compileable:
                    raise RuntimeError('Compiled decode did not receive a compilable StaticCache.')
                if self.compiled is None:
                    self.compiled = self.backend.torch.compile(original, **PROTOCOL['compile'])
                result = self.compiled(*args, **kwargs)
                self.calls += 1
                self.cache_types.add(type(cache).__name__)
                return result
            return original(*args, **kwargs)
        model.forward = forward
        try:
            yield self
        finally:
            if had_local:
                model.forward = local
            else:
                delattr(model, 'forward')


@contextmanager
def deadline(seconds):
    def timeout(signum, frame):
        raise TimeoutError(f'Compile/warmup exceeded {seconds}s. Stop and export; no automatic retry.')
    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def graphs():
    from torch._dynamo.utils import counters
    return int(counters['stats']['unique_graphs'])


def check_evidence(variant, calls, new_graphs, warm):
    if variant == 'compiled' and calls <= 0:
        raise RuntimeError('No compiled decoding calls observed; refusing to label this a compiled result.')
    if not warm and new_graphs:
        raise RuntimeError('A new graph compiled during measurement; this is not warmed timing. Export diagnostics.')


def measure(backend, stream, dispatch, case, warm=False):
    dispatch.variant = case['variant']
    before_calls, before_graphs = dispatch.calls, graphs()
    backend.torch.manual_seed(42)
    backend.torch.cuda.reset_peak_memory_stats()
    with policy(backend, case['variant'], warm=warm):
        m, d = generate_answer(backend, stream, case['messages'], profiled=False, cap=128)
    if d['rendered_prompt'] != case['rendered_prompt']:
        raise RuntimeError('Prompt changed during generation.')
    calls, created = dispatch.calls - before_calls, graphs() - before_graphs
    m.update(compiled_decode_calls=calls, new_graphs=created,
        peak_allocated_gib=backend.torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=backend.torch.cuda.max_memory_reserved()/2**30)
    # Preserve a failed measurement, including its timing, instead of silently discarding it.
    record = {**case, 'status': 'success', 'metrics': m, 'details': d,
              'answer_matches_pipeline': d['answer'] == case['accepted_answer']}
    try:
        check_evidence(case['variant'], calls, created, warm)
        if d['stop_reason'] != 'eos':
            raise RuntimeError('Answer did not finish at EOS; stop this experiment and review.')
    except RuntimeError as exc:
        record.update(status='failed', error=str(exc))
    return record


def prepare(repo, output, owner, env=None, question=None):
    data = read_json(repo/'data/llm_public/latency_prompts.json')
    if question is not None:
        data['samples'] = [s for s in data['samples'] if s['id'] == question]
    schedule(data)
    cfg = {'dataset': data, 'candidate': read_json(repo/'configs/llm_next_candidates.json')['candidates']['gemma4_e4b'],
           'protocol': PROTOCOL, 'environment': environment() if env is None else env, 'source': source_snapshot(repo)}
    name = 'llm-compile-' + object_digest(cfg)[:20]
    root = output/safe_name(owner)/'runs'/name
    if not (root/'run.json').exists():
        write_json(root/'run.json', {'run_id': name, 'created_at': now(), 'owner': owner, **cfg})
    write_json(output/owner/'active.json', {'run_id': name})
    return root, cfg


def run(root, cfg):
    # One attempt. Do not combine measurements from different sessions or repeat failed compilation.
    if (root/'attempt.json').exists():
        print('This experiment was already attempted. Reusing saved results; review/export cells 5–6.', flush=True)
        return
    session = uuid4().hex
    write_json(root/'attempt.json', {'session': session, 'started_at': now()})
    stage = 'loading'
    session_info = {'session': session, 'status': 'running'}
    started = time.perf_counter()
    try:
        print('Loading the selected Gemma model once...', flush=True)
        backend = NextLLM(cfg['candidate'], 'nf4', PROTOCOL)
        torch = backend.torch
        import torch._dynamo
        torch._dynamo.config.suppress_errors = False
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        session_info.update(load_s=time.perf_counter()-started,
            generation_config=backend.generation.to_dict(), loading_info=backend.loading_info,
            attention_implementation=backend.model.config._attn_implementation,
            auto_compile_quantizer_capability=bool(getattr(backend.model.hf_quantizer, 'is_compileable', False)),
            compile_route=PROTOCOL['method'], score_checks_enabled=True,
            gpu_name=torch.cuda.get_device_name(0), total_vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
        stage = 'verify_prompts'
        for c in schedule(cfg['dataset']):
            rendered = backend.tokenizer.apply_chat_template(c['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            if rendered != c['rendered_prompt']:
                raise RuntimeError('Rendered prompt differs: ' + c['id'])
            tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
            if len(tokens) + 128 > PROTOCOL['cache_length']:
                raise RuntimeError('Prompt exceeds the fixed cache budget.')
        with DecodeCompiler(backend).installed() as dispatch:
            stage = 'compile_and_warmup'
            warm_start, start_graphs = time.perf_counter(), graphs()
            print('Warmup/compilation: up to 8 minutes, saved separately from measured latency.', flush=True)
            with deadline(PROTOCOL['warmup_budget_s']):
                # Two full warmup passes expose first-token and steady-decode graph variants.
                for pass_number in (1, 2):
                    for c in schedule(cfg['dataset'])[:2 * len(cfg['dataset']['samples'])]:
                        label = f'warm{pass_number}__{c["id"]}'
                        print(label, flush=True)
                        r = measure(backend, stream, dispatch, c, warm=True)
                        r['session'] = session
                        write_json(root/'warmup'/f'{label}.json', r)
                        if r['status'] != 'success':
                            raise RuntimeError(r['error'])
                if graphs() <= start_graphs or dispatch.calls == 0:
                    raise RuntimeError('No compiled graph confirmed; stopping.')
            session_info.update(warmup_and_compile_s=time.perf_counter()-warm_start,
                warmup_new_graphs=graphs()-start_graphs, observed_cache_types=sorted(dispatch.cache_types))
            write_json(root/'sessions'/f'{session}.json', session_info)
            for c in schedule(cfg['dataset']):
                stage = c['id']
                print('Measuring', stage, flush=True)
                r = measure(backend, stream, dispatch, c)
                r['session'] = session
                write_json(root/'trials'/f'{c["id"]}.json', r)
                print(f'{stage}: {r["metrics"]["full_answer_s"]:.3f}s', flush=True)
                if r['status'] != 'success':
                    raise RuntimeError(r['error'])
        session_info['status'] = 'complete'
    except BaseException as exc:
        session_info.update(status='failed', failed_stage=stage)
        if stage == 'compile_and_warmup':
            session_info['warmup_and_compile_s'] = time.perf_counter()-warm_start
        write_json(root/'failures'/f'{session}.json', {'stage': stage, 'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc()})
        raise
    finally:
        session_info['attempt_wall_s'] = time.perf_counter()-started
        write_json(root/'sessions'/f'{session}.json', session_info)


def aggregate(root):
    """Review six groups, with all repeated answers visible and bound to their trial data."""
    data = read_json(root/'run.json')['dataset']
    for s in data['samples']:
        for mode in MODES:
            trials = [read_json(p) for p in sorted((root/'trials').glob(f'{s["id"]}__{mode}__*.json'))]
            if not trials:
                continue
            complete = (len(trials) == 3 and {r['repeat'] for r in trials} == {1,2,3}
                and len({r['session'] for r in trials}) == 1
                and all(r['status'] == 'success' and r['details']['stop_reason'] == 'eos' for r in trials))
            answers = list(dict.fromkeys(r.get('details', {}).get('answer','') for r in trials))
            times = [r['metrics']['full_answer_s'] for r in trials]
            write_json(root/'results'/f'{s["id"]}__{mode}.json', {
                'question_id': s['id'], 'variant': mode, 'status': 'success' if complete else 'incomplete',
                'trials': trials, 'unique_answers': answers, 'same_answer_across_repeats': len(answers) == 1,
                'all_match_pipeline': all(r.get('answer_matches_pipeline') for r in trials),
                'session': trials[0]['session'], 'saved': len(trials),
                'details': {'answer': '\n\n'.join(answers), 'stop_reason': 'eos' if complete else 'incomplete'},
                'metrics': {'full_answer_s': median(times) if complete else None,
                    'min_s': min(times), 'max_s': max(times),
                    'peak_allocated_gib': max(r['metrics']['peak_allocated_gib'] for r in trials),
                    'peak_reserved_gib': max(r['metrics']['peak_reserved_gib'] for r in trials)}})


def report(output, owner):
    base = output/safe_name(owner)
    root = base/'runs'/safe_name(read_json(base/'active.json')['run_id'])
    aggregate(root)
    rows = []
    for path in sorted((root/'results').glob('*.json')):
        r, rating = read_json(path), review(root, path)
        rows.append({k:r[k] for k in ('question_id','variant','status','session','saved','same_answer_across_repeats','all_match_pipeline')}
            | r['metrics'] | {'quality_pass': quality_pass(r,rating), 'reviewed': rating.get('decision') in ('pass','fail'),
                'answers': r['details']['answer'], 'notes': rating.get('notes','')})
    pairs = []
    for s in read_json(root/'run.json')['dataset']['samples']:
        by = {r['variant']:r for r in rows if r['question_id'] == s['id']}
        a,b = by.get('current',{}),by.get('compiled',{})
        complete = all(r.get('status') == 'success' for r in (a,b)) and a['session'] == b['session']
        identical = complete and all(r['same_answer_across_repeats'] for r in (a,b)) and a['answers'] == b['answers']
        pairs.append({'question_id': s['id'], 'complete_same_session': complete, 'identical_answers': identical,
            'both_quality_pass': all(r.get('quality_pass',False) for r in (a,b)),
            'median_speedup_if_identical': a['full_answer_s']/b['full_answer_s'] if identical else None})
    dest = base/'reports';dest.mkdir(parents=True,exist_ok=True)
    for name, records in [('comparison', rows), ('pairs', pairs)]:
        write_json(dest/(name+'.json'), records)
        with (dest/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(records[0]) if records else ['status']);w.writeheader();w.writerows(records)
    lines = ['# Gemma compilation experiment','',
        'Three warmed repeats per question and mode. Gemma alone, unchanged prompts and numerical guards.',
        'Compilation/warmup is separate. A new graph during measurement stops the run. No automatic retries.',
        'This tests static cache plus compiled decoding together; it does not isolate the contribution of each.',
        'Explicit torch.compile decode path: automatic HF compilation is disabled. Fullgraph failures are not suppressed.',
        'Three repeats are a small screening sample. No integrated latency or ASR/TTS memory-fit claim.', '',
        '|Question|Mode|Saved|Median s|Range s|Peak allocated GiB|Quality|',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        mid = f'{r["full_answer_s"]:.3f}' if r['full_answer_s'] is not None else '—'
        q = ('pass' if r['quality_pass'] else 'fail') if r['reviewed'] else 'unreviewed'
        lines.append(f'|{r["question_id"]}|{r["variant"]}|{r["saved"]}/3|{mid}|{r["min_s"]:.3f}–{r["max_s"]:.3f}|{r["peak_allocated_gib"]:.2f}|{q}|')
    for p in sorted((root/'sessions').glob('*.json')):
        s = read_json(p)
        lines += ['',f'Session status: {s["status"]}. Load: {s.get("load_s", "unavailable")} s. '
            f'Compilation + warmup: {s.get("warmup_and_compile_s", "unavailable")} s.']
    for p in sorted((root/'failures').glob('*.json')):
        f = read_json(p);lines += ['',f'**Stopped at {f["stage"]}:** {f["error"]}']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--owner',default='abel')
    p.add_argument('--question',choices=['everyday','fee','unknown'],help='Run only this question for a bounded smoke experiment.')
    a = p.parse_args();safe_name(a.owner)
    if a.action == 'run':
        root,cfg = prepare(a.repo,a.output,a.owner,question=a.question)
        try:run(root,cfg)
        finally:report(a.output,a.owner)
    else:
        report(a.output,a.owner)
        print('RESULT ZIP:',export_run(a.output/a.owner,a.output/f'llm-compile-{a.owner}-{uuid4().hex[:10]}.zip'),flush=True)


if __name__ == '__main__':
    main()
