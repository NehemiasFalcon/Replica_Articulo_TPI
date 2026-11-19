"""Experimento independiente (sin LLM): recuperación de evidencia por el preprocesador.
Los valores no son Precision/Recall/F1 de RCA de Xu et al.
"""
import argparse,csv,json,time
from pathlib import Path
from statistics import mean
from experiment import discover_cases
from logsage import Settings,prepare_blocks


def intersect(a,b,c,d):
    return max(0,min(b,d)-max(a,c)+1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,default=Path('data_original'))
    p.add_argument('--out',type=Path,default=Path('results/preprocessing'))
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--drain',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cases=discover_cases(args.dataset_root,strict=True)
    if args.limit:cases=cases[:args.limit]
    records=[]
    for i,c in enumerate(cases,1):
        t=time.monotonic()
        blocks=prepare_blocks(c.failed_log,[c.success_log],Settings(),use_drain=args.drain)
        merged=[]
        for b in blocks:
            if merged and b.start<=merged[-1][1]+1:
                merged[-1]=(merged[-1][0],max(merged[-1][1],b.end))
            else:merged.append((b.start,b.end))
        # Each annotated interval can be discontiguous; count distinct label lines retained.
        n_gold=sum(b-a+1 for a,b in c.label_ranges)
        n_gold_retained=sum(intersect(a,b,l,r) for a,b in c.label_ranges for l,r in merged)
        n_selected=sum(b-a+1 for a,b in merged)
        records.append({
            'case_id':c.case_id, 'total_log_lines':len(c.failed_log.splitlines()),
            'number_of_labels':len(c.label_ranges), 'annotated_lines':n_gold,
            'selected_blocks':len(blocks),'selected_lines':n_selected,
            'selected_token_count_approx':sum(b.tokens for b in blocks),
            'any_overlap':int(n_gold_retained>0),
            'annotated_line_recall_proxy':round(n_gold_retained/n_gold,6) if n_gold else 0,
            'elapsed_seconds':round(time.monotonic()-t,3),
        })
        if i%10==0 or i==len(cases):
            print(f'Preprocessing {i}/{len(cases)} cases; last={c.case_id}',flush=True)
    with (args.out/'preprocessing_cases.csv').open('w',encoding='utf-8-sig',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary={
        'note':'Preprocessing evidence preservation proxy, NOT LogSage paper RCA F1. No LLM was run.',
        'cases':len(records),'cases_with_any_labeled_evidence':sum(r['any_overlap'] for r in records),
        'any_evidence_rate':round(mean(r['any_overlap'] for r in records),5),
        'mean_annotated_line_recall_proxy':round(mean(r['annotated_line_recall_proxy'] for r in records),5),
        'mean_selected_blocks':round(mean(r['selected_blocks'] for r in records),2),
        'mean_selected_tokens_proxy':round(mean(r['selected_token_count_approx'] for r in records),2),
        'mean_selected_log_line_fraction':round(mean(r['selected_lines']/r['total_log_lines'] for r in records),5),
        'cases_with_zero_blocks':sum(r['selected_blocks']==0 for r in records),
    }
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
