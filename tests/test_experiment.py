import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import discover_cases, find_experiment_dir, heuristic_prediction, parse_references, _eval_prediction


def test_public_dataset_structure():
    root = Path(__file__).resolve().parents[1] / 'fixtures'
    cases = discover_cases(root)
    assert len(cases) == 3
    assert all(c.label_end <= len(c.failed_log.splitlines()) for c in cases)
    assert all(c.success_log for c in cases)


def test_cited_lines_parsing():
    cited = parse_references('Lines 5–7 and line 10, 12; line 500', 100)
    assert cited == {5, 6, 7, 10, 12}


def test_cited_lines_avoid_other_numbers():
    assert parse_references('Error code 500 occurred at step 90', 250) == set()


def test_proxy_metric_correctness():
    result = _eval_prediction({1, 9, 10}, 8, 15)
    assert result['matched_count'] == 2
    assert result['localizacion_any'] == 1
    assert result['localizacion_90'] == 0
    assert abs(result['precision_citas'] - 2/3) < .00001


def test_heuristic_can_find_failure():
    cases = discover_cases(Path(__file__).resolve().parents[1] / 'fixtures')
    assert all(heuristic_prediction(case) for case in cases)


def test_multirange_labels(tmp_path):
    d = tmp_path / 'dataset' / 'experiment_dataset' / 'repo' / 'pair_1'
    (d / 'fail').mkdir(parents=True)
    (d / 'success').mkdir(parents=True)
    (d / 'fail' / 'fail.txt').write_text('line1\nline2\nline3\nline4\n', encoding='utf8')
    (d / 'success' / 'success.txt').write_text('ok\n', encoding='utf8')
    (d / 'fail' / 'label.json').write_text('{"start": 1, "end": 1}\n{"start": 3, "end": 3}\n')
    c = discover_cases(tmp_path)[0]
    assert c.label_ranges == ((1,1),(3,3))
    v = _eval_prediction({1,2,3}, c.label_start,c.label_end,c.label_ranges)
    assert v['matched_count'] == 2


def test_full_public_data_label_count():
    root=Path(__file__).resolve().parents[1] / 'data_original'
    if root.is_dir():
        cases=discover_cases(root, strict=True)
        assert len(cases)==250
        assert len([c for c in cases if len(c.label_ranges)>1])==2


def test_ollama_cannot_target_external_endpoint():
    from experiment import query_ollama
    try:
        query_ollama('sys','case','test-model','https://example.com/api/chat')
    except ValueError:
        pass
    else:
        raise AssertionError('External endpoint must be rejected')
