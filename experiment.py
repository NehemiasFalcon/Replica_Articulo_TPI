"""Experimental runner for Xu et al. LogSage (ASE 2025) public data.

Modes:
  heuristic = local sanity baseline, NO LLM and NOT the paper's RCA result
  llm       = unofficial reimplementation of LogSage RCA on public cases

The released label.json contains an *abnormal log segment*. Therefore measured
localization is a PROXY and is not asserted to be identical to Table II's
90% key-error-line overlap evaluation, which lacks complete scoring details.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request
import urllib.error
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rca_response import (RCA_FORMAT_VERSION, RCA_JSON_SCHEMA,
                          InvalidRCAResponse, check_rca_json)

from logsage import (
    KEYWORDS, RCA_SYSTEM_PROMPT, Settings, build_rca_input,
    candidate_indices, build_success_matcher, prepare_blocks
)

REPOSITORY = 'https://github.com/ByteLuo1029/dataset.git'
PAPER_F1 = {'gpt-4o': 0.9898, 'claude-3.7-sonnet': 0.9878, 'deepseek-v3': 0.9898}
REQUIRED_COLUMNS = [
    'case_id', 'project', 'pair', 'lines', 'label_start', 'label_end',
    'mode', 'model', 'prediction_count', 'matched_count',
    'precision_citas', 'localizacion_any', 'localizacion_90',
    'prompt_tokens_approx', 'seconds', 'error', 'cited_lines', 'label_ranges',
]


@dataclass
class Case:
    case_id: str
    project: str
    pair: str
    failed_log: str
    success_log: str
    label_start: int
    label_end: int
    label_ranges: tuple[tuple[int, int], ...] = ()


def acquire_dataset(target: Path, url: str = REPOSITORY) -> Path:
    """Download dataset to user's machine. Real GitHub access is required."""
    target = target.resolve()
    if target.exists() and any(target.iterdir()):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['git', 'clone', '--depth', '1', url, str(target)]
    print('Cloning public dataset:', url, flush=True)
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            'No se pudo descargar el dataset. Descarga https://github.com/ByteLuo1029/dataset '
            'manualmente y pasa --dataset-root RUTA. Error original: ' + str(exc)
        ) from exc
    return target


def find_experiment_dir(root: Path) -> Path:
    attempts = [root / 'dataset' / 'experiment_dataset',
                root / 'experiment_dataset',
                root / 'dataset' / 'dataset' / 'experiment_dataset']
    for a in attempts:
        if a.is_dir():
            return a
    raise FileNotFoundError('No se encontró experiment_dataset. Revisar --dataset-root; '
                            'rutas exploradas: ' + ', '.join(map(str, attempts)))


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def discover_cases(root: Path, strict: bool = False) -> list[Case]:
    base = find_experiment_dir(root)
    cases, invalid = [], []
    for project in sorted(p for p in base.iterdir() if p.is_dir()):
        for pair in sorted((p for p in project.iterdir() if p.is_dir() and p.name.startswith('pair_')),
                           key=lambda p: (int(p.name[5:]) if p.name[5:].isdigit() else 10**9, p.name)):
            failed = pair / 'fail' / 'fail.txt'
            success = pair / 'success' / 'success.txt'
            label = pair / 'fail' / 'label.json'
            try:
                if not failed.is_file() or not label.is_file():
                    raise FileNotFoundError('missing fail.txt / label.json')
                raw_label = _read(label).strip()
                try:
                    labels = json.loads(raw_label)
                except json.JSONDecodeError:
                    # Public dataset has two real cases containing JSONL multi-range annotations.
                    labels = [json.loads(row) for row in raw_label.splitlines() if row.strip()]
                if isinstance(labels, dict):
                    labels = [labels]
                if not isinstance(labels, list) or not labels:
                    raise ValueError('invalid/empty label data')
                ranges = tuple(sorted((int(x['start']), int(x['end'])) for x in labels))
                failure_log = _read(failed)
                count = len(failure_log.splitlines())
                if any(a < 1 or b < a or b > count+1 for a, b in ranges):
                    raise ValueError(f'bad label ranges {ranges} for {count} lines')
                cases.append(Case(f'{project.name}/{pair.name}', project.name, pair.name,
                                  failure_log,
                                  _read(success) if success.is_file() else '',
                                  min(a for a, b in ranges), max(b for a, b in ranges),
                                  ranges))
            except (ValueError, KeyError, OSError, TypeError) as exc:
                invalid.append(f'{project.name}/{pair.name}: {exc}')
    if invalid:
        print(f'Cases rejected: {len(invalid)}', file=sys.stderr)
        for item in invalid[:10]:
            print('  ' + item, file=sys.stderr)
        if strict:
            raise RuntimeError(f'{len(invalid)} invalid case(s) in dataset')
    if not cases:
        raise RuntimeError('No se cargó ningún par con fail.txt y label.json')
    return cases


def heuristic_prediction(case: Case, limit: int = 10) -> set[int]:
    """A transparent local baseline: score suspicious lines, NOT a LogSage/LLM diagnosis."""
    lines = case.failed_log.splitlines()
    refs = [case.success_log] if case.success_log else []
    matcher = build_success_matcher(refs, use_drain=False)
    candidates = candidate_indices(lines, matcher, Settings(), bool(refs))
    scored = []
    for i in candidates:
        lower = lines[i].lower()
        score = (10 if re.search(r'(^|\s)(fatal|error|exception|failed|failures?:|traceback)\b', lower) else 0)
        score += (4 if any(k in lower for k in ('assert', 'exit code', 'cannot', 'no such file')) else 0)
        score += (1 if any(k in lower for k in KEYWORDS) else 0)
        if score:
            scored.append((score, i))
    return {i + 1 for _, i in sorted(scored, key=lambda p: (-p[0], -p[1]))[:limit]}


def parse_references(output: str, length: int) -> set[int]:
    """Parse explicit log-line references written in the diagnostic output.

    Accepts 'Line 30', 'Lines 30-32, 40' and excludes numerical values unrelated
    to line citations; cannot parse all possible free-form model answers.
    """
    found: set[int] = set()
    # Support the headings used by Qwen ("Log Line References:") as well as
    # the paper-style "Lines 6-12" and our strict "Error Log Lines: 295, 308".
    pattern = re.compile(
        r'\b(?:ERROR_LINES|(?:error\s+)?log\s+lines?|log\s+line\s+references?|lines?|líneas?)'
        r'\s*:?\s*(\d+(?:\s*(?:[-–—,;&]|\band\b|\by\b|\bto\b)\s*\d+)*)',
        re.I,
    )
    for match in pattern.finditer(output):
        source = match.group(1)
        # parse explicit range parts, then standalone numbers
        for a, b in re.findall(r'(\d+)\s*(?:-|–|—|\bto\b)\s*(\d+)', source):
            x, y = int(a), int(b)
            if 1 <= x <= y <= length and y-x <= 10000:
                found.update(range(x, y+1))
        for n in re.findall(r'\d+', source):
            x = int(n)
            if 1 <= x <= length:
                found.add(x)
    return found


def _eval_prediction(pred: set[int], start: int, end: int,
                     ranges: tuple[tuple[int, int], ...] | None = None) -> dict:
    labels = ranges or ((start, end),)
    overlapping = sum(any(a <= n <= b for a, b in labels) for n in pred)
    # Proxy: cited-line precision against the entire annotated segment.
    return {
        'prediction_count': len(pred), 'matched_count': overlapping,
        'precision_citas': round(overlapping / len(pred), 6) if pred else 0,
        'localizacion_any': int(overlapping > 0),
        'localizacion_90': int(bool(pred) and overlapping/len(pred) >= 0.9),
    }


def run_case(case: Case, args: argparse.Namespace, settings: Settings,
             cache: dict[str, dict]) -> tuple[dict, Optional[dict]]:
    tic = time.monotonic()
    nlines = len(case.failed_log.splitlines())
    row = dict(case_id=case.case_id, project=case.project, pair=case.pair,
               lines=nlines, label_start=case.label_start, label_end=case.label_end,
               label_ranges=json.dumps(case.label_ranges),
               mode=args.mode, model=args.model if args.mode == 'ollama' else '',
               error='', prompt_tokens_approx=0)
    detail = None
    try:
        if args.mode == 'heuristic':
            predicted = heuristic_prediction(case, args.max_citations)
        else:
            if case.case_id in cache:
                detail = cache[case.case_id]
                predicted = set(detail['error_log_lines'])
                row['prompt_tokens_approx'] = detail.get('prompt_tokens_approx', 0)
            else:
                blocks = prepare_blocks(case.failed_log,
                                        [case.success_log] if case.success_log else [],
                                        settings=settings, use_drain=args.drain)
                # Only numbers visible to the model are valid citations.
                allowed_lines = {i for b in blocks for i in range(b.start, b.end + 1)}
                cfg = ''  # separate workflow configuration unavailable in public data
                prompt = build_rca_input(blocks, cfg)
                row['prompt_tokens_approx'] = sum(b.tokens for b in blocks)
                detail = {'case_id': case.case_id, 'model': args.model,
                          'format_version': RCA_FORMAT_VERSION,
                          'prompt_tokens_approx': row['prompt_tokens_approx'],
                          'blocks': [{'start': b.start, 'end': b.end} for b in blocks],
                          'diagnosis': '', 'raw_response': ''}
                if not allowed_lines:
                    raise InvalidRCAResponse('El preprocesamiento no entregó líneas de log')
                # First attempt uses Ollama's JSON schema constraint.
                # If response is malformed/ungrounded, try ONCE more.
                for attempt in range(2):
                    extra = ('\n\nCRITICAL: Return only JSON. error_log_lines must '
                             'contain 1-8 original numbered log lines showing the '
                             'actual failure; do not return generic advice.') if attempt else ''
                    raw = query_ollama(RCA_SYSTEM_PROMPT, prompt + extra,
                                       args.model, args.ollama_url,
                                       response_format=RCA_JSON_SCHEMA)
                    detail['raw_response'] = raw
                    detail['attempts'] = attempt + 1
                    try:
                        structured, predicted, answer = check_rca_json(raw, allowed_lines)
                        detail.update({'structured': structured, 'diagnosis': answer,
                                       'error_log_lines': sorted(predicted)})
                        break
                    except InvalidRCAResponse as exc:
                        detail['diagnosis'] = raw
                        detail['response_validation_error'] = str(exc)
                        if attempt == 1:
                            raise
            # Never interpret an invalid response as '0 citations, OK'.
            if not predicted:
                raise InvalidRCAResponse('Respuesta sin referencias válidas a líneas de log')
        row.update(_eval_prediction(predicted, case.label_start, case.label_end, case.label_ranges))
        row['cited_lines'] = ','.join(map(str, sorted(predicted)))
    except Exception as exc:
        row.update(_eval_prediction(set(), case.label_start, case.label_end, case.label_ranges))
        row['cited_lines'] = ''
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['seconds'] = round(time.monotonic()-tic, 3)
    return row, detail


def summarize(rows: list[dict], args: argparse.Namespace, dataset: str) -> dict:
    valid = [r for r in rows if not r.get('error')]
    n = len(valid)
    return {
        'status': 'completed' if n == len(rows) and n else 'partial',
        'dataset': dataset, 'date_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'model': args.model if args.mode == 'ollama' else None,
        'total_cases': len(rows), 'completed_cases': n,
        'error_cases': len(rows)-n,
        'mean_cited_line_precision_proxy': round(sum(r['precision_citas'] for r in valid)/n, 5) if n else None,
        'localization_any_case_rate': round(sum(r['localizacion_any'] for r in valid)/n, 5) if n else None,
        'localization_90_case_rate': round(sum(r['localizacion_90'] for r in valid)/n, 5) if n else None,
        'note': ('These are localization PROXY metrics using the public label.json abnormal segment. '
                 'NOT the published LogSage RQ1 Precision/Recall/F1 scores. '
                 'Heuristic mode is NOT LLM inference.'),
        'paper_reported_f1_gpt4o': 0.9898,
    }


def query_ollama(system_prompt: str, prompt: str, model: str, url: str,
                 response_format: Optional[dict] = None) -> str:
    """Call a user-installed LOCAL Ollama model, without subscription or cloud API."""
    if not url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise ValueError('For safety this mode accepts only local Ollama URLs')
    body = {
        'model': model, 'stream': False,
        'options': {'temperature': 0.1, 'num_predict': 1024, 'num_ctx': 8192},
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ],
    }
    if response_format is not None:
        body['format'] = response_format
    payload = json.dumps(body).encode('utf8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:400]
        raise RuntimeError(
            f'Ollama respondió HTTP {exc.code}: {detail}. '
            'Si la opción format no es compatible, actualiza Ollama.') from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError('Ollama no disponible o modelo no instalado. ' + str(exc)) from exc
    result = (reply.get('message') or {}).get('content', '')
    if not result:
        raise RuntimeError('Ollama devolvió una respuesta vacía')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description='Ejecutar diagnóstico/evaluación sobre dataset LogSage ASE 2025')
    p.add_argument('--dataset-root', type=Path, default=Path('data/LogSage_original'))
    p.add_argument('--download', action='store_true', help='git clone de datos públicos en --dataset-root')
    p.add_argument('--mode', choices=['heuristic', 'ollama'], default='heuristic')
    p.add_argument('--model', default='qwen2.5-coder:3b')
    p.add_argument('--ollama-url', default='http://127.0.0.1:11434/api/chat')
    p.add_argument('--token-limit', type=int, default=5500,
                   help='Local model: lower than paper 22000 token cap to fit modest RAM')
    p.add_argument('--limit', type=int, default=0, help='0 = todos los casos')
    p.add_argument('--max-citations', type=int, default=10)
    p.add_argument('--drain', action='store_true')
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--strict', action='store_true')
    p.add_argument('--sleep', type=float, default=0.0)
    p.add_argument('--max-errors', type=int, default=3, help='Aborta tras N errores consecutivos')
    args = p.parse_args()
    if args.mode == 'heuristic' and args.drain:
        p.error('--drain se aplica solo a --mode ollama')
    root = acquire_dataset(args.dataset_root) if args.download else args.dataset_root
    cases = discover_cases(root, strict=args.strict)
    if args.limit > 0:
        cases = cases[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    existing = args.out/'details.jsonl'
    cache = {}
    if args.mode == 'ollama' and existing.is_file():
        for line in _read(existing).splitlines():
            try:
                item = json.loads(line)
                if (item.get('model') == args.model and
                    item.get('format_version') == RCA_FORMAT_VERSION and
                    item.get('structured') and item.get('error_log_lines')):
                    cache[item['case_id']] = item
            except (ValueError, KeyError):
                pass
    rows = []
    consecutive_errors = 0
    for i, case in enumerate(cases, 1):
        row, detail = run_case(case, args, Settings(token_limit=args.token_limit), cache)
        rows.append(row)
        consecutive_errors = consecutive_errors + 1 if row.get('error') else 0
        if args.mode == 'ollama' and detail and case.case_id not in cache:
            with existing.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(detail, ensure_ascii=False)+'\n')
        print(f'[{i}/{len(cases)}] {case.case_id}: '
              f"pred={row['prediction_count']} inside={row['matched_count']} "
              f"status={'ERROR: '+row['error'] if row['error'] else 'OK'}", flush=True)
        if consecutive_errors >= args.max_errors:
            print('Aborting: too many consecutive errors', file=sys.stderr, flush=True)
            break
        if args.sleep and i < len(cases):
            time.sleep(args.sleep)
    with (args.out/'cases.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args, str(root))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Output:', args.out.resolve())

if __name__ == '__main__':
    main()
