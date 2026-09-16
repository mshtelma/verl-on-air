#!/usr/bin/env python3
"""Reproduce the completed air/82 attempt-4 findings without GPU/judge calls.
Arithmetic checks expose candidate derivations; they do not replace source semantics.
"""
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys

ARCHIVE = Path(__file__).resolve().parents[1]
REPO = ARCHIVE.parents[1]
CODE = ARCHIVE/'code_snapshot'
os.environ['OQ_REWARD_MODE'] = 'grounded'
os.environ.pop('OQ_REWARD_QUARANTINE', None)
sys.path[:0] = [str(CODE/'scripts'), str(CODE/'scripts/reward')]
from reward.strict_answer import strict_correct
from reward.officeqa_reward import score_answer


def main():
    raw = gzip.decompress((ARCHIVE/'bundles/v21_339.jsonl.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == 'dd0e003d17df301a9aa1a2b6256478c32c35daf7817f619907ded783b8184de4'
    cases = {r['uid']: r for r in map(json.loads, raw.decode().splitlines())}
    root = ARCHIVE/'runs/543029753940588'
    report = json.loads((root/'report.json').read_text())
    score_bytes = (root/'scores.jsonl').read_bytes()
    assert hashlib.sha256(score_bytes).hexdigest() == report['artifacts']['scores']['sha256']
    scores = {r['uid']: r for r in map(json.loads, score_bytes.decode().splitlines())}
    assert len(cases) == len(scores) == 339 and set(cases) == set(scores)
    assert report['positives']['accepted'] == 16
    assert report['negatives']['false_accepts'] == 5 and report['n_unknown'] == 11
    reasons = Counter(scores[u]['reward_reason'] for u in report['positives']['false_rejects'])
    assert reasons['verified quote(s) do not contain the committed value'] == 55
    print('Completed run 543029753940588:', report['generated_at'])
    print('Controls 16/80; labeled-negative accepts 5/242; unknown 11/339; wrong-gate 0/17 nonzero')
    print('Case upper95', report['negatives']['fa_upper_95_case'], 'base-ID upper95', report['negatives']['fa_upper_95_base'])
    print('Control rejection reasons:', dict(reasons))
    for family in sorted({r['expected'] for r in cases.values()}):
        rows = [s for u,s in scores.items() if cases[u]['expected'] == family]
        print(family, 'n', len(rows), 'accepted', sum(s['reward'] > 0 for s in rows),
              'unknown', sum(s.get('reward_status') == 'unknown' for s in rows))
    original = {r['uid']: r for r in map(json.loads,(REPO/'officeqa_pilot_records/officeqa_traces.jsonl').read_text().splitlines())}
    for uid in report['negatives']['false_accept_uids']:
        case, score = cases[uid], scores[uid]
        old = original[case['base_uid']]
        print('\nAccepted labeled negative', uid, 'reward', score['reward'], 'original prediction', repr(old['pred']))
        for quote in score['judge_verdict'].get('supporting_quotes', []):
            hits = []
            for i, step in enumerate(case['trajectory']):
                for j, result in enumerate(step.get('tool_results', [])):
                    if quote in str(result.get('result') or ''):
                        orig = old['trajectory'][i]['tool_results'][j]['result']
                        hits.append({'step': i, 'tool': result['name'], 'unchanged': result['result'] == orig})
            print('quote', quote[:140], 'exact hits', hits)
    r = original['UID0148']
    print('\nUID0148 original strict value/list:', strict_correct(r['gt'],r['pred']),
          'legacy benchmark selector:', score_answer(r['gt'],r['pred']))
    print('UID0204 reported-value subtraction:', Decimal('1.47') - Decimal('1.30'))
    values = [
        (6634663,8877769,148283,38750276,6869302,8733757,146839,37094883),
        (6817884,7595657,132774,34996129,7373916,7920048,126521,36353790),
        (8198962,7879571,119769,37619185,8288205,7757327,118966,37591058),
    ]
    with localcontext() as ctx:
        ctx.prec = 50
        for n_components in (2, 3):
            june = [Decimal(sum(v[:n_components])) / Decimal(v[3]) for v in values]
            sept = [Decimal(sum(v[4:4+n_components])) / Decimal(v[7]) for v in values]
            diff = abs(sum(june)/Decimal(3) - sum(sept)/Decimal(3))*Decimal(100)
            print('UID0122 components', n_components, 'difference pp', diff,
                  'rounded', diff.quantize(Decimal('.001')))
    print('Category membership still requires original-source adjudication; matching gold is not its proof.')


if __name__ == '__main__':
    main()
