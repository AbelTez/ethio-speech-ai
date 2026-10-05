"""Resumable LLM screening; no ML imports for reports or export."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import os
from pathlib import Path
import re
import traceback
from uuid import uuid4

from .artifacts import (completed_keys, digest, environment, export_run, now,
                        object_digest, read_json, safe_name, save_result, write_json)
from .asr_scoring import percentile
from .llm_backend import prompt_messages

PROTOCOL = {'version': 'llm-screen-v2-explicit-greedy', 'max_new_tokens': 128, 'max_input_tokens': 2048,
            'max_time_s': 60, 'batch_size': 1, 'seed': 42, 'planned_repeats': 1,
            'decoding': 'greedy, non-thinking where switch supported',
            'attention': 'sdpa', 'compute_dtype': 'recorded per preset', 'offload': False,
            'use_model_defaults': False, 'score_validation': 'every generated step; overhead included',
            'warmup': 'separate greeting, up to 8 tokens each model load',
            'timing': 'synchronized wall clock; streamer decode included; full_answer_s includes input prep',
            'clause': 'first punctuation-ended streamed text, or remaining text at generation end; unvalidated'}
REVIEW_FIELDS = ['clarity', 'relevance', 'supported', 'complete_answer', 'no_repetition', 'notes']


def validate_dataset(dataset):
    if not dataset.get('instruction', '').strip() or not dataset.get('samples'):
        raise ValueError('Missing instruction or prompts.')
    ids = set()
    for sample in dataset['samples']:
        key = safe_name(sample['id'])
        if key in ids or not sample['question'].strip() or not sample['expected'].strip():
            raise ValueError('Duplicate ID or empty prompt/review requirement.')
        if sample['slice'] not in ('everyday', 'grounded', 'guards'):
            raise ValueError('Unknown slice.')
        ids.add(key)
    return ids


def llm_source(repo):
    paths = ['src/amharic_s2s/artifacts.py', 'src/amharic_s2s/asr_scoring.py',
             'src/amharic_s2s/llm_backend.py', 'src/amharic_s2s/llm_benchmark.py',
             'scripts/llm_compare.py', 'requirements/llm-colab.txt']
    paths += [p for p in ('src/amharic_s2s/llm_next.py', 'scripts/llm_next.py',
                         'requirements/llm-next-colab.txt') if (Path(repo)/p).is_file()]
    hashes = {p: digest(Path(repo)/p) for p in paths}
    return {'sha256': object_digest(hashes), 'files': hashes}


def llm_environment():
    env = environment()
    for name in ('protobuf',):
        try:
            env['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env['packages'][name] = None
    import torch
    env.update(cuda_build=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
               gpu_capability=list(torch.cuda.get_device_capability(0)),
               gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               cudnn_version=torch.backends.cudnn.version())
    return env


def prepare(repo, output, owner, model_key, preset='default', env=None,
            catalog_path=None, dataset_path=None, protocol=None):
    safe_name(owner); safe_name(model_key)
    repo, output = Path(repo), Path(output)
    catalog = read_json(catalog_path or repo/'configs/llm_candidates.json')['candidates']
    if model_key not in catalog:
        raise ValueError(f'Unknown model key {model_key!r}; choose: {", ".join(catalog)}')
    candidate = catalog[model_key]
    if not re.fullmatch(r'[a-f0-9]{40}', candidate['revision']):
        raise ValueError('Candidate revision must be immutable.')
    preset = candidate['default_preset'] if preset == 'default' else preset
    if preset not in ('fp16', 'nf4', 'nf4_fp32'):
        raise ValueError('Unknown numerical preset.')
    dataset = read_json(dataset_path or repo/'data/llm_public/prompts.json')
    validate_dataset(dataset)
    config = {'stage': 'llm', 'planned_repeats': 1, 'model_key': model_key,
              'candidate': candidate, 'preset': preset,
              'compute_dtype': 'float32' if preset == 'nf4_fp32' else 'float16',
              'protocol': protocol or PROTOCOL}
    env = llm_environment() if env is None else env
    source = llm_source(repo)
    identity = {'config': object_digest(config), 'dataset': object_digest(dataset),
                'source': source['sha256'], 'environment': object_digest(env)}
    name = f"llm-{model_key}-{preset}-{object_digest(identity)[:20]}"
    directory = output/owner/'runs'/name
    if (directory/'run.json').exists():
        previous = read_json(directory/'run.json')
        if previous['identity'] != identity or previous['owner'] != owner:
            raise ValueError('Run identity/owner mismatch; refusing to overwrite.')
        if read_json(directory/'config.json') != config or read_json(directory/'dataset.json') != dataset:
            raise ValueError('Saved run configuration or dataset changed; refusing resume.')
    else:
        if directory.exists() and any(directory.glob('results/*.json')):
            raise ValueError('Run metadata missing but results exist; preserve and inspect this directory.')
        write_json(directory/'config.json', config)
        write_json(directory/'dataset.json', dataset)
        write_json(directory/'run.json', {'run_id': name, 'owner': owner, 'created_at': now(),
                   'identity': identity, 'environment': env, 'source': source, 'stage': 'llm'})
    write_json(directory/'sessions'/f'{uuid4().hex}.json', {'started_at': now(), 'environment': env})
    write_json(output/owner/'active'/f'{model_key}.json', {'run_name': name, 'preset': preset})
    return directory, config, dataset


def measured_samples(directory, dataset, backend, sample_limit):
    done = completed_keys(directory)
    samples = dataset['samples'][:sample_limit] if sample_limit else dataset['samples']
    for sample in samples:
        if (sample['id'], 1) in done:
            continue
        try:
            metrics, details = backend.generate(prompt_messages(dataset, sample))
        except Exception as exc:
            save_result(directory, sample['id'], 1, {}, 'failed', f'{type(exc).__name__}: {exc}',
                        details={'question': sample['question'], 'expected': sample['expected']})
            raise
        details.update(question=sample['question'], context=sample['context'],
                       expected=sample['expected'], slice=sample['slice'])
        save_result(directory, sample['id'], 1, metrics, details=details)
        print(f"SAVED {sample['id']} | {metrics['full_answer_s']:.3f}s | "
              f"{details['stop_reason']}\n{details['answer']}\n", flush=True)


def preflight(repo, model_key):
    from .llm_backend import text_only_imports
    text_only_imports()
    from transformers import AutoTokenizer, GenerationConfig
    from huggingface_hub import hf_hub_download
    catalog = read_json(Path(repo)/'configs/llm_candidates.json')['candidates']
    selected = catalog if model_key == 'all' else {model_key: catalog[model_key]}
    dataset = read_json(Path(repo)/'data/llm_public/prompts.json')
    validate_dataset(dataset)
    errors = []
    for key, candidate in selected.items():
        try:
            config_path = hf_hub_download(candidate['model_id'], 'config.json', revision=candidate['revision'])
            config = read_json(config_path)
            expected = {'gemma2_2b':'gemma2', 'gemma3_4b':'gemma3', 'qwen3_4b':'qwen3',
                        'qwen3_8b':'qwen3', 'gemma3n_e4b':'gemma3n'}[key]
            if config['model_type'] != expected:
                raise ValueError('Unexpected checkpoint architecture.')
            tokenizer = AutoTokenizer.from_pretrained(candidate['model_id'], revision=candidate['revision'])
            kwargs = {'enable_thinking': False} if 'enable_thinking' in candidate else {}
            lengths = []
            for s in dataset['samples']:
                text = tokenizer.apply_chat_template(prompt_messages(dataset, s), tokenize=False,
                                                     add_generation_prompt=True, **kwargs)
                lengths.append(len(tokenizer(text, add_special_tokens=False)['input_ids']))
            if max(lengths) > PROTOCOL['max_input_tokens']:
                raise ValueError('Prompt exceeds input limit; revise the shared protocol before running.')
            GenerationConfig.from_pretrained(candidate['model_id'], revision=candidate['revision'])
            print(f"OK {key} | max prompt {max(lengths)} tokens | {candidate['default_preset']} | {candidate['revision']}", flush=True)
        except Exception as exc:
            errors.append(key)
            print(f'ACCESS/CONFIG ERROR {key}: {type(exc).__name__}: {exc}', flush=True)
    if errors:
        raise RuntimeError('Preflight failed for '+', '.join(errors)+'. Accept Gemma terms using the HF_TOKEN account and rerun. No weights were loaded.')


def run(repo, output, owner, model_key, preset, sample_limit, reviewed_run=None,
        backend_class=None, catalog_path=None, dataset_path=None, protocol=None):
    if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':
        raise RuntimeError('CUDA_LAUNCH_BLOCKING=1 changes timing. Use a standalone diagnostic process, then remove it before benchmarking.')
    import time
    import torch
    from .llm_backend import TransformersLLM
    if sample_limit not in (0, 3):
        raise ValueError('Use 3 for smoke or 0 for the full set.')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA GPU runtime. No model downloaded.')
    directory, config, dataset = prepare(repo, output, owner, model_key, preset,
        catalog_path=catalog_path, dataset_path=dataset_path, protocol=protocol)
    active_protocol = config['protocol']
    print(f'Run: {directory.name}\nResults: {directory}', flush=True)
    if sample_limit == 0:
        if reviewed_run != directory.name:
            raise ValueError('Full run requires --reviewed-run with the current run name after smoke review.')
        if not all((s['id'], 1) in completed_keys(directory) for s in dataset['samples'][:3]):
            raise ValueError('Finish the three smoke cases before the full run.')
    selected = dataset['samples'][:sample_limit] if sample_limit else dataset['samples']
    if all((s['id'], 1) in completed_keys(directory) for s in selected):
        print('Selected prompts already saved; no model loaded.', flush=True)
        report(output, owner)
        return
    session = uuid4().hex
    backend = None
    stage = 'model_load'
    try:
        torch.manual_seed(active_protocol['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        started = time.perf_counter()
        backend = (backend_class or TransformersLLM)(config['candidate'], config['preset'], active_protocol)
        torch.cuda.synchronize()
        load_s = time.perf_counter()-started
        stage = 'warmup'
        started = time.perf_counter()
        backend.generate([{'role':'user', 'content':'በአማርኛ ሰላም በል።'}], max_new_tokens=8)
        write_json(directory/'loads'/f'{session}.json', {'at': now(), 'load_including_download_s': load_s,
                   'warmup_s':time.perf_counter()-started, 'generation_config':backend.generation.to_dict(),
                   'device_map': getattr(backend.model, 'hf_device_map', None),
                   'loading_info': backend.loading_info, 'preset':config['preset']})
        stage = 'measured_prompts'
        measured_samples(directory, dataset, backend, sample_limit)
    except Exception as exc:
        write_json(directory/'session_errors'/f'{session}.json', {
            'at':now(), 'stage':stage, 'error':f'{type(exc).__name__}: {exc}',
            'traceback':traceback.format_exc()})
        raise
    finally:
        del backend
        report(output, owner)


def write_csv(path, rows, fields):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding='utf-8-sig')


def report(output, owner):
    safe_name(owner)
    home = Path(output)/owner
    reports = home/'reports'
    reports.mkdir(parents=True, exist_ok=True)
    # Keep manual edits across report regeneration, including earlier environments.
    review_path = home/'human_review.csv'
    reviews = {}
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig', newline='') as f:
            reviews = {(r['run_id'],r['sample_id']):r for r in csv.DictReader(f)}
    rows, summaries = [], []
    for path in sorted((home/'runs').glob('*/run.json')):
        directory = path.parent
        metadata, config = read_json(path), read_json(directory/'config.json')
        dataset = read_json(directory/'dataset.json')
        group = object_digest({k:v for k,v in metadata['identity'].items() if k != 'config'} | {'protocol':config['protocol']})[:12]
        base = {'run_id':directory.name, 'comparison_group':group, 'model_key':config['model_key'],
                'preset':config['preset'], 'model_id':config['candidate']['model_id'],
                'revision':config['candidate']['revision'], 'gpu':metadata['environment'].get('gpu','')}
        run_rows = []
        for record in sorted((directory/'results').glob('*.json')):
            r = read_json(record)
            d = r.get('details', {})
            row = {**base, 'sample_id':r['sample_id'], 'status':r['status'], 'error':r['error'],
                   **r['metrics'], **{k:d.get(k,'') for k in ('slice','question','context','expected','answer','raw_answer','first_clause','clause_source','stop_reason','capped')},
                   **d.get('diagnostics',{})}
            review = reviews.get((directory.name,r['sample_id']), {})
            row.update({k:review.get(k,'') for k in REVIEW_FIELDS})
            row['human_reviewed'] = all(row[k] in ('pass','fail') for k in REVIEW_FIELDS[:-1])
            row['human_pass'] = row['human_reviewed'] and all(row[k]=='pass' for k in REVIEW_FIELDS[:-1])
            run_rows.append(row)
        rows.extend(run_rows)
        good = [r for r in run_rows if r['status']=='success']
        summary = {**base, 'saved':len(good), 'expected':len(dataset['samples']),
                   'complete':len(good)==len(dataset['samples']),
                   'failed':sum(r['status']=='failed' for r in run_rows),
                   'session_errors':len(list((directory/'session_errors').glob('*.json'))),
                   'capped':sum(bool(r['capped']) for r in good),
                   'empty':sum(bool(r.get('empty_answer')) for r in good),
                   'under_2_s':sum(r['full_answer_s']<2 for r in good),
                   'human_reviewed':sum(r['human_reviewed'] for r in good),
                   'human_pass':sum(r['human_pass'] for r in good)}
        for field in ('full_answer_s','generation_s','first_token_s','first_text_s','first_clause_s','peak_allocated_gib'):
            values = [r[field] for r in good if field in r]
            summary[field+'_n'] = len(values)
            if values:
                summary.update({field+'_p50':percentile(values,.5), field+'_p90':percentile(values,.9), field+'_max':max(values)})
        summaries.append(summary)
    response_fields = list(dict.fromkeys(['run_id','sample_id','model_key','preset','status','answer','expected']+[k for r in rows for k in r]))
    write_csv(reports/'responses.csv', rows, response_fields)
    write_csv(reports/'comparison.csv', summaries, list(dict.fromkeys(['run_id','model_key','preset','saved','expected']+[k for r in summaries for k in r])))
    # Add new prompts without erasing human review fields. Review CSV is editable.
    for row in rows:
        key = (row['run_id'],row['sample_id'])
        reviews[key] = {k:row.get(k,'') for k in ['run_id','sample_id','model_key','preset','question','context','expected','answer']+REVIEW_FIELDS}
    write_csv(review_path, list(reviews.values()), ['run_id','sample_id','model_key','preset','question','context','expected','answer']+REVIEW_FIELDS)
    write_json(reports/'summary.json', summaries)
    lines = ['# LLM screening report', '',
             'Compare complete configurations in the same comparison group. Preset differences remain visible.',
             'One pass per authored development prompt; expected counts are shown below. Descriptive percentiles, not a stability study.',
             'Success means generation completed and was saved, not that answer quality passed.',
             'Latency includes poor, empty and capped outputs when generation returned; failures are counted separately.',
             'Full answer includes input preparation, generation and final decoding. First clause is a punctuation heuristic, not an approved TTS handoff.',
             'Under 2 seconds is an optimization target. No connected pipeline or production claim.',
             'Fill human_review.csv with pass/fail for the five quality fields, then regenerate this report.', '',
             '|Model|Preset|Saved/expected|Reviewed/pass|Capped|Full p50 s|Full p90 s|Clause p50 s|Under 2 s|Group|',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in summaries:
        fmt = lambda k:f'{r[k]:.3f}' if k in r else '—'
        lines.append(f"|{r['model_key']}|{r['preset']}|{r['saved']}/{r['expected']}|{r['human_reviewed']}/{r['human_pass']}|{r['capped']}|{fmt('full_answer_s_p50')}|{fmt('full_answer_s_p90')}|{fmt('first_clause_s_p50')}|{r['under_2_s']}/{r['saved']}|{r['comparison_group']}|")
    (reports/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight','run','report','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path)
    p.add_argument('--owner',default='abel')
    p.add_argument('--model',default='gemma2_2b')
    p.add_argument('--preset',choices=['default','fp16','nf4','nf4_fp32'],default='default')
    p.add_argument('--sample-limit',type=int,choices=[0,3],default=3)
    p.add_argument('--reviewed-run')
    args = p.parse_args()
    if args.action == 'preflight':
        preflight(args.repo,args.model)
        return
    if args.output is None:
        p.error('--output is required for run/report/export')
    if args.action == 'run':
        run(args.repo,args.output,args.owner,args.model,args.preset,args.sample_limit,args.reviewed_run)
    else:
        report(args.output,args.owner)
        if args.action == 'export':
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            print(export_run(args.output/args.owner,args.output/f'llm-results-{safe_name(args.owner)}-{stamp}-{uuid4().hex[:6]}.zip'))


if __name__ == '__main__':
    main()
