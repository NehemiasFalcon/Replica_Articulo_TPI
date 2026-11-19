"""Presentar evidencia de un caso real de la demo de remediación."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def view(out: Path, scenario: str) -> str:
    p = out / f"{scenario}_detail.json"
    if not p.is_file():
        raise FileNotFoundError(f"No existe {p}. Ejecuta remediation_demo.py primero.")
    d = json.loads(p.read_text(encoding='utf-8'))
    initial = d.get('initial', {})
    recheck = d.get('recheck') or {}
    build = d.get('build') or {}
    text = [
        '=' * 72,
        f"CASO: {scenario.upper()} | modo: {d.get('mode')} | modelo: {d.get('model')}",
        '1. FALLO INICIAL (log real de tests de este proyecto generado)',
        '\n'.join(initial.get('output', '').strip().splitlines()[-22:]) or '(sin log)',
        f"Estado: {'FAIL' if initial.get('exit_code', 1) != 0 else 'PASS'}",
        '2. DIAGNOSTICO',
        d.get('diagnosis') or '(sin diagnostico disponible)',
        '3. HERRAMIENTA SELECCIONADA',
        d.get('selected_tool', ''),
        '4. CAMBIO REAL EN EL CODIGO (diff)',
        d.get('remediation', {}).get('patch') or '(sin parche aplicado)',
        '5. VALIDACION',
        f"Retest: {'PASS' if recheck.get('exit_code') == 0 else 'NO PASO / NO EJECUTADO'}",
        f"Build: {'PASS' if build.get('exit_code') == 0 else 'NO PASO / NO EJECUTADO'}",
        f"Pipeline recuperado: {'SI' if d.get('success') else 'NO'}",
        '=' * 72,
    ]
    return '\n\n'.join(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('results/remediation-demo'))
    p.add_argument('--scenario', choices=['unit', 'lint', 'unsupported'], default='unit')
    args = p.parse_args()
    print(view(args.out, args.scenario))

if __name__ == '__main__':
    main()
