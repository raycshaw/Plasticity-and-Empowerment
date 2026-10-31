"""Build the frozen 200-question dataset, preserving the original 50 records."""
import ast
import hashlib
import json
import operator
import random
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'data/gsm8k200'
SOURCE_HASH = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
NOTES = {
    304: 'Uses the source convention of four working weeks in April.',
    880: 'Assumes the replacement rope has the same 20-meter length, as in the source solution.',
    1052: 'Uses the source convention that the initial total, including Romeo, is 120.',
    394: 'Uses the source interpretation of two books per grid position plus the stated extra books.',
    279: 'Interprets six times older as six times as old, as in the source solution.',
    182: "Jan is 34 now, so Jan was 32 two years ago; Mark was 21 then and Jean is 25 now.",
    885: 'Friday is twice the common Tuesday/Wednesday daily count; both equal counts are corrected together.',
    796: 'Sums the payroll for all three months, with hiring before each monthly payment.',
    423: 'Source reverses the comparison: jogger jeans originally cost less, not more. Revised signed jogger-minus-tattered difference is -12 dollars; absolute gap is 12 dollars.',
    371: 'Source asks for shoes but counts pairs. Revised reference is 66 pairs, or 132 individual shoes.',
    515: 'Uses the source interpretation of $4 per bag, not per pack.',
    526: 'Source confuses power and energy units. Reference follows its intended 2 additional billed units per day, for seven days.',
}


def arithmetic(expression):
    """Evaluate only literal rational arithmetic, without executing source text."""
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError('Only numeric literals and +, -, *, / are allowed')
    return visit(ast.parse(expression, mode='eval').body)


def build():
    raw = (DIRECTORY / 'source_test.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASH
    source = [json.loads(line) for line in raw.splitlines()]
    original = pq.read_table(ROOT / 'data/gsm8k50/gsm8k50_injections.parquet').to_pylist()
    selection = json.loads((DIRECTORY / 'selection.json').read_text())
    assert selection['existing_ids'] == [r['id'] for r in original]
    excluded = {int(r['id'].split(':')[1]) for r in original}
    expected = random.Random(selection['seed']).sample([i for i in range(len(source)) if i not in excluded], 150)
    assert selection['added_indices'] == expected
    changes = [line.split('\t') for line in (DIRECTORY / 'corrections.tsv').read_text().splitlines()]
    assert [int(c[0]) for c in changes] == expected
    rows = [dict(r, cohort='original_50_length_screened', revised_question=None,
                 replacement_old=None, replacement_new=None) for r in original]
    cases = []
    for index, before, after, expression in changes:
        index = int(index)
        item = source[index]
        question = item['question']
        assert question.count(before) == 1, (index, before, 'replacement must be unique')
        assert before != after
        reference = arithmetic(expression)
        revised = str(reference.numerator) if reference.denominator == 1 else str(float(reference))
        answer = item['answer'].split('####')[-1].strip()
        assert reference != Fraction(answer.replace(',', '')), (index, 'unchanged answer')
        row = dict(id=f'test:{index}', question=question, answer=answer,
                   injection=f'Correction: where the problem says "{before}", it should say "{after}". Keep everything else the same.',
                   injection_after_tokens=20, revised_answer=revised,
                   revised_answer_calculation=expression, source_issue=index in {423, 371, 526},
                   review_note=NOTES.get(index, ''), screening_seed=None,
                   screening_non_eos_tokens=None, screening_complete=None, first_span_token_ids=None,
                   cohort='added_150_random_unscreened', revised_question=question.replace(before, after, 1),
                   replacement_old=before, replacement_new=after)
        rows.append(row)
        cases.append(row)
    assert len(rows) == len({r['id'] for r in rows}) == 200
    for old, new in zip(original, rows):
        assert all(new[k] == v for k, v in old.items())
    metadata = dict(source_url='https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl',
                    source_sha256=SOURCE_HASH, split='test', seed=selection['seed'],
                    selection='Original 50 preserved; 150 uniformly sampled without replacement from remaining rows; no new length screening.',
                    injection_authoring='Manually authored phrase corrections and revised calculations, checked for exact unique matches and arithmetic.',
                    limitations='Mixed selection cohorts; source wording issues flagged. Revised answers are audit fields and are never sent to the model.')
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b'dataset_provenance': json.dumps(metadata, sort_keys=True).encode()})
    pq.write_table(table, DIRECTORY / 'gsm8k200_injections.parquet')
    (DIRECTORY / 'new_cases.json').write_text(json.dumps(dict(metadata=metadata, cases=cases), indent=2, ensure_ascii=False) + '\n')
    print('Verified and wrote 200 unique questions, with 50 unchanged records and 150 new corrections.')


if __name__ == '__main__':
    build()
