"""Recalculate Xu et al. Table II from the CSVs provided with the PUBLIC dataset.
This is a reproduction of PUBLISHED statistics, not fresh LLM model inference.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PAPER = {
    'GPT_4o': {
        ('LogSage', 'default'): (.9798, 1.0000, .9898),
        ('LogPrompt', 'CoT'): (.8580, .4654, .6035),
        ('LogPrompt', 'InContext'): (.6120, .4530, .5206),
        ('LogGPT', '1'): (.7280, .3081, .4330),
        ('LogGPT', '2'): (.7185, .3484, .4693),
    },
    'Claude37': {
        ('LogSage', 'default'): (.9837, .9918, .9878),
        ('LogPrompt', 'CoT'): (.8467, .3361, .4812),
        ('LogPrompt', 'InContext'): (.6864, .4046, .5091),
        ('LogGPT', '1'): (.8140, .3075, .4464),
        ('LogGPT', '2'): (.7715, .3073, .4396),
    },
    'Deepseek_V3': {
        ('LogSage', 'default'): (.9838, .9959, .9898),
        ('LogPrompt', 'CoT'): (.8620, .4477, .5893),
        ('LogPrompt', 'InContext'): (.6923, .5018, .5819),
        ('LogGPT', '1'): (.8006, .3394, .4767),
        ('LogGPT', '2'): (.7380, .4438, .5543),
    },
}


def metric(rows):
    tp,fp,fn = (sum(int(r[k]) for r in rows) for k in
                ('true_positives','false_positives','false_negatives'))
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp,fp,fn,p,r,f1


def load(path):
    with path.open(encoding='utf-8-sig',newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
            writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',type=Path,default=Path('data_original'))
    ap.add_argument('--out',type=Path,default=Path('results/original-published-reanalysis'))
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for model in PAPER:
        path=a.dataset_root/'experiment_result'/'baseline_result'/f'{model}_experiment_results.csv'
        groups=defaultdict(list)
        for row in load(path):
            groups[(row['method'],str(row['prompt_type']))].append(row)
        for key, expected in PAPER[model].items():
            items=groups[key]
            tp,fp,fn,p,r,f1=metric(items)
            observed=(round(p,4),round(r,4),round(f1,4))
            records.append({
                'model':model,'method':key[0],'prompt_type':key[1],
                'cases':len(items),'TP':tp,'FP':fp,'FN':fn,
                'precision':observed[0],'recall':observed[1], 'f1':observed[2],
                'paper_precision':expected[0], 'paper_recall':expected[1],
                'paper_f1':expected[2],
                'matches_table_II':observed==expected,
                'average_total_tokens_original':round(sum(int(x['total_tokens']) for x in items)/len(items),1),
                'average_queries_original':round(sum(int(x['query_count']) for x in items)/len(items),2),
            })
    write_csv(a.out/'paper_table_II_recalculated.csv', records)
    abl=load(a.dataset_root/'experiment_result'/'ablation_result'/'ablation_result.csv')
    ab_groups=defaultdict(list)
    for row in abl:ab_groups[(row['method'],row['variant'])].append(row)
    ab=[]
    for (method,variant),items in sorted(ab_groups.items()):
        tp,fp,fn,p,r,f1=metric(items)
        ab.append({'method':method,'variant':variant,'cases':len(items),'TP':tp,
                   'FP':fp,'FN':fn, 'precision':round(p,4),'recall':round(r,4),
                   'f1':round(f1,4),
                   'avg_tokens':round(sum(int(x['total_tokens']) for x in items)/len(items),1)})
    write_csv(a.out/'ablation_recalculated.csv', ab)
    summary={
        'experiment':'re-analysis of AUTHORS original CSV; not new LLM inference',
        'baseline_comparisons':len(records),
        'all_match_paper_table_II':all(x['matches_table_II'] for x in records),
        'cases_per_group':{f'{r["model"]}/{r["method"]}/{r["prompt_type"]}':r['cases'] for r in records},
        'ablation_groups':len(ab),
        'note':'DeepSeek V3 has incomplete per-method baseline rows (210/211 for non-LogSage methods), as in the source CSV. No fabricated missing cases.',
    }
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
