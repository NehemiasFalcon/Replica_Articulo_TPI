from logsage import Settings, approximate_template, candidate_indices, merge_ranges, prepare_blocks, weights_for

def test_template_normalization():
    assert approximate_template('step 23 passed') == approximate_template('step 55 passed')

def test_candidate_filter_union():
    lines = ['init', 'ok', 'ERROR boom', 'last']
    got = candidate_indices(lines, lambda x: True, Settings(tail_lines=1), True)
    assert got == {2, 3}

def test_merge_expansions():
    assert merge_ranges([(1, 3), (3, 7), (10, 11)]) == [(1, 7), (10, 11)]

def test_weights_failure_marker():
    w = weights_for(['--- FAIL: TestA', 'ok'], {0}, Settings())
    assert w == [10, 0]

def test_full_pipeline_keeps_error_context():
    failed = '\n'.join(['ok']*15 + ['FAILED TestA', 'Traceback a', 'expected 1 actual 2'] + ['ok']*3)
    success = '\n'.join(['ok']*21)
    blocks = prepare_blocks(failed, [success], Settings(tail_lines=1, token_limit=5000))
    assert any('FAILED TestA' in b.content for b in blocks)
    assert all(b.start >= 1 and b.end <= 21 for b in blocks)

def test_budget_respected():
    failed = '\n'.join(f'ERROR issue {i}' for i in range(100))
    blocks = prepare_blocks(failed, settings=Settings(token_limit=10, tail_lines=2))
    assert sum(b.tokens for b in blocks) <= 10
