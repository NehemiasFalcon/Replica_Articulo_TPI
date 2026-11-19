import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remediation_demo import create_workspace, run_gate, run_scenario, write_results, apply_tool, extract_json_object, select_knowledge


def test_unit_gate_before_after(tmp_path):
    ws = create_workspace(tmp_path, 'unit')
    assert run_gate(ws, 'unit')['exit_code'] != 0
    result = apply_tool(ws, 'fix_add_operator', 'unit')
    assert result['applied']
    assert 'return a + b' in (ws/'src/calculator.py').read_text()
    assert run_gate(ws, 'unit')['exit_code'] == 0


def test_lint_gate_before_after(tmp_path):
    ws = create_workspace(tmp_path, 'lint')
    assert run_gate(ws, 'lint')['exit_code'] != 0
    assert apply_tool(ws, 'lint_fix', 'lint')['applied']
    assert run_gate(ws, 'lint')['exit_code'] == 0


def test_reject_tool_on_wrong_scenario(tmp_path):
    ws = create_workspace(tmp_path, 'unsupported')
    assert not apply_tool(ws, 'fix_add_operator', 'unsupported')['applied']
    assert run_gate(ws, 'unsupported')['exit_code'] != 0


def test_rule_experiment_creates_valid_artifacts(tmp_path):
    records = [run_scenario(name, mode='rule', model='none', ollama_url='', apply=True, out=tmp_path)
               for name in ('unit', 'lint', 'unsupported')]
    summary = write_results(tmp_path, records, 'rule')
    assert summary['total_cases'] == 3
    assert summary['initial_failures'] == 3
    assert summary['repairs_applied'] == 2
    assert summary['successful_repairs'] == 2
    assert summary['rerun_success_rate'] == 1.0
    assert summary['end_to_end_recovery_rate'] == 0.6667
    assert (tmp_path/'unit_patch.diff').read_text().strip()
    assert json.loads((tmp_path/'unit_detail.json').read_text())['recheck']['exit_code'] == 0


def test_no_apply_does_not_patch(tmp_path):
    record = run_scenario('unit', mode='rule', model='none', ollama_url='', apply=False, out=tmp_path)
    assert not record['remediation']['applied']
    assert not record['success']
    assert 'return a - b' in (tmp_path/'workspaces/unit/src/calculator.py').read_text()


def test_json_extraction():
    assert extract_json_object('```json\n{"tool":"lint_fix","reason":"whitespace"}\n```')['tool']=='lint_fix'


def test_retrieval_returns_only_relevant_knowledge():
    k=select_knowledge('src/report.py:2: TRAILING_WHITESPACE','def title(x): pass')
    assert len(k)==1 and 'lint' in k[0]['source']


def test_ollama_orchestration_mocked(monkeypatch, tmp_path):
    # Tests Ollama integration logic without pretending to run an LLM in CI.
    import remediation_demo
    replies = iter([
        'UNIT_TESTS fails: add(a,b) subtracts instead of adding.',
        '{"tool":"fix_add_operator","reason":"return a - b conflicts with unit-test expected sum"}',
    ])
    monkeypatch.setattr(remediation_demo, 'query_ollama', lambda *a, **k: next(replies))
    record = run_scenario('unit', mode='ollama', model='qwen2.5-coder:3b',
                          ollama_url='http://127.0.0.1:11434/api/chat', apply=True, out=tmp_path)
    assert record['selected_tool'] == 'fix_add_operator'
    assert record['remediation']['applied']
    assert record['success']
    assert len(record['knowledge']) == 1
