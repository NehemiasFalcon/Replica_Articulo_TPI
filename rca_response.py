"""Structured, evidence-checked responses from local Ollama for LogSage RCA.

Model citations are validated against the numbered input blocks. This checks
provenance, NOT semantic accuracy or the paper's original F1 score.
"""
from __future__ import annotations

import json
from typing import Any

RCA_FORMAT_VERSION = 'rca-json-v3'

RCA_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'task_type': {'type': 'string'},
        'root_cause': {'type': 'string'},
        'error_log_lines': {
            'type': 'array',
            'items': {'type': 'integer'},
            'minItems': 1,
            'maxItems': 8,
        },
        'evidence': {'type': 'string'},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
    },
    'required': ['task_type', 'root_cause', 'error_log_lines', 'evidence', 'confidence'],
    'additionalProperties': False,
}


class InvalidRCAResponse(ValueError):
    """Model returned unusable/unverifiable citations (not successful RCA)."""


def check_rca_json(response: str, allowed_lines: set[int]) -> tuple[dict[str, Any], set[int], str]:
    """Parse schema response, require grounded citations, and format for humans.

    Does NOT use any labels or ground truth. A citation is acceptable only if
    the corresponding line actually appeared in the sent log blocks.
    """
    try:
        parsed = json.loads(response.strip())
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvalidRCAResponse('Qwen no devolvió JSON válido con líneas de evidencia') from exc

    if not isinstance(parsed, dict):
        raise InvalidRCAResponse('La respuesta de Qwen no es un objeto JSON')

    for field in ('task_type', 'root_cause', 'evidence', 'confidence'):
        if not isinstance(parsed.get(field), str) or not parsed[field].strip():
            raise InvalidRCAResponse(f'Falta el campo de diagnóstico: {field}')
    if parsed['confidence'] not in ('high', 'medium', 'low'):
        raise InvalidRCAResponse('El nivel de confianza no es válido')

    refs = parsed.get('error_log_lines')
    if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
        raise InvalidRCAResponse('Faltan líneas de error (1 a 8 números)')
    if any(type(x) is not int or x < 1 for x in refs):
        raise InvalidRCAResponse('Las líneas de error deben ser números enteros positivos')
    pred = set(refs)
    if len(pred) != len(refs):
        raise InvalidRCAResponse('La respuesta repite números de línea')
    ungrounded = pred - allowed_lines
    if ungrounded:
        raise InvalidRCAResponse('Qwen citó líneas NO enviadas al modelo: ' + str(sorted(ungrounded)))

    readable = ('Diagnosis Process:\n'
                f"Task Type: {parsed['task_type']}\n"
                'Error Log Lines: ' + ', '.join(str(x) for x in refs) + '\n'
                f"Evidence: {parsed['evidence']}\n"
                'Root Cause:\n'
                f"[{parsed['confidence'].title()} Likelihood] {parsed['root_cause']}")
    return parsed, pred, readable
