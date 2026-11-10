"""Quick full-B value screen, with explicit reviewed examples; not a failure-rate estimator."""
import argparse
import pyarrow.parquet as pq
import csv
import json
import re
from pathlib import Path
from collections import Counter


WORDS = dict(zip('zero one two three four five six seven eight nine ten twelve sixteen eighteen twenty thirty forty fifty sixty seventy eighty ninety hundred'.split(),
                 [0,1,2,3,4,5,6,7,8,9,10,12,16,18,20,30,40,50,60,70,80,90,100]))
PATTERN = r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)|\b(?:' + '|'.join(WORDS) + r')\b'

def numbers(text):
    return [float(WORDS[m.group()]) if m.group() in WORDS else float(m.group().replace(',', ''))
            for m in re.finditer(PATTERN, text.lower())]

# Human-readable B responses manually inspected after the automated screen.
OLD_USED = {(0,'test:1209',2), (200,'test:407',0), (200,'test:407',1),
            (200,'test:407',3), (200,'test:407',4), (1400,'test:407',0), (2600,'test:407',1)}
WRONG_OTHER = {(0,'test:88',4), (200,'test:88',3), (2600,'test:1209',4)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    OUT=args.run/'analysis/injection_value_screen'
    OUT.mkdir(exist_ok=True, parents=True)
    records=[]
    for batch in pq.ParquetFile(args.run/'matched_rollouts.parquet').iter_batches():
        for r in batch.to_pylist():
            before,after=r['injection'].split(', not ')
            new,old=numbers(before)[-1],numbers(after)[0]
            values=set(numbers(r['b_text']))
            o,n=old in values,new in values
            category='both' if o and n else 'old_only' if o else 'revised_only' if n else 'neither'
            key=(r['step'],r['id'],r['trial'])
            reviewed=('confirmed_old_value_used' if key in OLD_USED else
                      'wrong_third_value_used' if key in WRONG_OTHER else 'not_semantically_reviewed')
            records.append(dict(step=r['step'],id=r['id'],trial=r['trial'],source_issue=r['source_issue'],
                                old_value=old,revised_value=new,mention_category=category,review=reviewed,
                                ended_at_eos=r['b']['ended_at_eos'],injection=r['injection'],b_text=r['b_text']))
    assert len(records)==960
    counts={}
    for step in [0,200,1400,2600]:
        rs=[r for r in records if r['step']==step and not r['source_issue']]
        assert len(rs)==230
        counts[step]=dict(responses=len(rs),mentions=dict(Counter(r['mention_category'] for r in rs)),
                          reviewed=dict(Counter(r['review'] for r in rs)))
    summary=dict(scope='Matched fixed-A OLMo JSD cohort; full saved B, not just scored first 100 tokens',
                 screened=920,excluded_source_issue_responses=40,excluded_question_ids=['test:1309','test:255'],
                 checkpoint_counts=counts,confirmed_old_value_cases=7,
                 interpretation='Confirmed counts are lower bounds, not exhaustive ignoring rates. Both-value and revised-only responses were not comprehensively semantically reviewed. Mentioning the revised value does not establish successful correction adoption.',
                 matching='Exact numeric values with comma/decimal normalization and listed single number words; no complete compound-number language parser')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'response_screen.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
