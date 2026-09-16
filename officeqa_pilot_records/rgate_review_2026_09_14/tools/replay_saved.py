import collections
import hashlib
import json
import pathlib
import sys

import gzip
import tempfile
import os
OUT = pathlib.Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
CODE = OUT/'code_snapshot'
RESULTS = pathlib.Path(tempfile.mkdtemp(prefix='officeqa-rgate-replay-'))
os.environ['OQ_REWARD_MODE'] = 'grounded'
os.environ.pop('OQ_REWARD_QUARANTINE', None)
sys.path[:0] = [str(CODE/'scripts'), str(CODE/'scripts/reward'), str(CODE/'scripts/officeqa')]
from reward import officeqa_grounded_reward as rw
from reward import grounding
import make_rgate_expanded as gx
import rgate_run


def load(path):
    path = pathlib.Path(path)
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def evidence(rec):
    return '\n'.join(str(r.get('result') or '') for st in rec.get('trajectory', []) for r in st.get('tool_results', []))


records = load(OUT/'bundles/original_317.jsonl.gz')
by = {r['uid']: r for r in records}
current = load(OUT/'bundles/v21_339.jsonl.gz')
current_by = {r['uid']: r for r in current}
scores = load(OUT/'runs/591895497446045/scores.jsonl')
score_by = {s['uid']: s for s in scores}
traces = load(ROOT/'officeqa_pilot_records/officeqa_traces.jsonl')
trace_by = {t['uid']: t for t in traces}

print('=== COUNTS ===')
for label, rs in [('original', records), ('v21', current)]:
    neg = [r for r in rs if r['expected'].startswith('negative_')]
    ps = [r for r in rs if r['expected'].startswith('grounded_positive')]
    print(label, 'cases', len(rs), 'base_ids', len({r['base_uid'] for r in rs}),
          'negative_base_ids', len({r['base_uid'] for r in neg}),
          'positive_base_ids', len({r['base_uid'] for r in ps}),
          'difficulty', dict(collections.Counter(r.get('difficulty') for r in rs)))

print('\n=== SAVED SCORE TAXONOMY ===')
for label, ss in [('v1', load(OUT/'runs/657777974855885/scores.jsonl')), ('v2', scores)]:
    grouped = collections.defaultdict(list)
    for s in ss: grouped[by[s['uid']]['expected']].append(s)
    print(label)
    for k, vv in sorted(grouped.items()):
        print(k, 'n', len(vv), 'positive_reward', sum(s['reward']>0 for s in vv),
              'unknown', sum(s.get('reward_status')=='unknown' for s in vv))

print('\n=== HISTORICAL TOOL-OUTPUT CLIPPING ===')
all_results = [str(r.get('result') or '') for t in traces for st in t.get('trajectory', []) for r in st.get('tool_results', [])]
clipped_t = [t for t in traces if any(len(str(r.get('result') or '')) == 3000 for st in t.get('trajectory', []) for r in st.get('tool_results', []))]
print('n_outputs', len(all_results), 'max_len', max(map(len, all_results)), 'exact3000', sum(len(t)==3000 for t in all_results),
      'affected_traces', len(clipped_t), 'of', len(traces))
print('positive_G_traces_with_exact3000', len([r for r in records if r['mutation']=='G' and r['base_uid'] in {t['uid'] for t in clipped_t}]))

print('\n=== ALT DATE CONTRADICTIONS ===')
for r in records:
    if r['mutation'] != 'A': continue
    files = gx._source_files(r)
    alt = gx._alt_filename(files[0], r['base_uid'])
    year = int(gx.FILE_RE.search(alt).group(1))
    qyears = gx._q_years(r)
    if qyears and year < min(map(int, qyears)):
        print(r['uid'], r['note'], 'asked_years', sorted(qyears),
              'score', score_by[r['uid']]['reward'], 'reason', score_by[r['uid']]['reward_reason'])

print('\n=== FALSE REJECTIONS: ACTUAL BAD-YEAR MATCHES ===')
for s in scores:
    if not by[s['uid']]['expected'].startswith('grounded_positive') or s.get('reward_status')=='unknown' or s['reward']>0: continue
    if 'period label' not in s['reward_reason']: continue
    r = by[s['uid']]
    ev = evidence(r)
    years = set(rw._Q_YEAR_RE.findall(r['question']))
    matched = rw._verify_supporting_quotes(s['judge_verdict'].get('supporting_quotes', []), ev)
    bad=[]
    for line in matched:
        ys = set(rw._Q_YEAR_RE.findall(line))
        for m in rw._BULLETIN_RE.finditer(line): ys.discard(m.group(1))
        if ys and not ys&years: bad.append({'found_years': sorted(ys), 'line': line[:600]})
    print(s['uid'], 'question_year_tokens', sorted(years), json.dumps(bad))

print('\n=== SURVIVING EVIDENCE: SELECTED FALSE ACCEPTS ===')
for uid in ['UID0020_U', 'UID0023_U', 'UID0043_U', 'UID0241_U', 'UID0002_N1', 'UID0081_D', 'UID0123_D', 'UID0232_D', 'UID0050_T']:
    r=by[uid]; s=score_by[uid]
    print(uid, 'gt', r['gt'], 'label',r['expected'], 'note',r['note'])
    qs=(s.get('judge_verdict') or {}).get('supporting_quotes', [])
    for q in qs:
        hits=[]
        for i,st in enumerate(r.get('trajectory', [])):
            for j,tr in enumerate(st.get('tool_results', [])):
                txt=str(tr.get('result') or '')
                if q in txt:
                    calls=st.get('tool_calls', [])
                    c=calls[j] if j<len(calls) else {}
                    orig=trace_by.get(r['base_uid'], {}).get('trajectory', [])
                    oldtxt=orig[i].get('tool_results', [])[j].get('result') if i<len(orig) and j<len(orig[i].get('tool_results', [])) else None
                    hits.append({'step':i,'tool':tr.get('name'),'file':(c.get('args') or {}).get('file_name'),
                                 'unchanged_result':txt==oldtxt})
        print(' QUOTE',q[:250], 'exact_hits', hits)

print('\n=== CURRENT V21 RESCORE WITH FROZEN V2 VERDICTS ===')
# This isolates the deterministic code change. No new judge calls and NOT a new live-gate result.
replayed=[]
for s in scores:
    r=by[s['uid']]
    g=rw._assemble(s['strict_correct'],s['grounding'],s.get('judge_verdict'),
                   mode='grounded',is_composite=bool(s['grounding'].get('is_composite')),
                   question=r['question'],evidence_text=evidence(r),gt=str(r.get('gt') or ''))
    out={**s,'reward':g['score'],'reward_status':g['status'],'reward_reason':g['reward_reason']}
    replayed.append(out)
rep=rgate_run.evaluate(records,replayed)
print('positives',rep['positives'],'negatives', {k:v for k,v in rep['negatives'].items() if 'uids' not in k}, 'unknown',rep['n_unknown'])
new_fr=[]
for o in replayed:
    old=score_by[o['uid']]
    if by[o['uid']]['expected'].startswith('grounded_positive') and old['reward']>0 and not o['reward']>0:
        new_fr.append(o['uid'])
        print('NEW REJECT',o['uid'],'gt',o['gt'],'reason',o['reward_reason'],'q',o['question'])
print('new_positive_rejections',len(new_fr))
(RESULTS/'v21-frozen-v2-verdicts.json').write_text(json.dumps(rep,indent=2))
(RESULTS/'v21-frozen-v2-scores.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in replayed))

print('\n=== V21 BLANKED READS STILL COUNT AS REAL RETRIEVAL ===')
for uid in ['UID0023_U','UID0043_U','UID0072_U']:
    r=current_by[uid]
    rep=grounding.grounding_report(r['trajectory'],{'source_files':r.get('source_files','')})
    print(uid,'files_pulled',rep['files_pulled'], 'read_outputs',[
        str(tr.get('result'))[:90] for st in r['trajectory'] for tr in st.get('tool_results',[]) if tr.get('name')=='read_document'])

print('\nFresh replay outputs:', RESULTS)
assert rep is not None
