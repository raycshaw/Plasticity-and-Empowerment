"""Protocol checks with a tiny deterministic model; no downloads or rollouts on Qwen."""
import unittest
from unittest.mock import patch
from test_sampled_rlvr import Model, Tokenizer
from sampled_rlvr_experiment import rollout, HORIZON, load_reference_a


class Tests(unittest.TestCase):
    def test_opening_shared_excluded_and_reference_a_preserved(self):
        model = Model()
        a = [4, 5] * 10
        opening = [3, 5, 4]
        result = rollout(model, Tokenizer(), 'Q', 'J', 1, fixed_a=a, b_opening_ids=opening)
        self.assertEqual(result['a']['token_ids'], a)
        self.assertEqual(result['b']['token_ids'], [5] * HORIZON)
        self.assertEqual(result['empowerment']['scored_positions'], HORIZON)
        self.assertEqual(result['b']['content_token_count'], HORIZON)
        base = [1,3,4,2]
        for t in range(HORIZON):
            for side, (span, message) in enumerate(((a,[3]),(a,[]),([],[3]))):
                self.assertEqual(model.calls[40 + 3*t + side],
                                 base + span + [4,1] + message + [4,2] + opening + [5]*t)
        uncached = rollout(Model(), Tokenizer(), 'Q', 'J', 1, cached=False,
                           fixed_a=a, b_opening_ids=opening)
        self.assertEqual(result, uncached)

    def test_invalid_fixed_a_rejected(self):
        for a in ([5]*19, [0]+[5]*19):
            with self.assertRaises(ValueError):
                rollout(Model(), Tokenizer(), 'Q', 'J', 1, fixed_a=a)

    def test_reference_identity_and_completeness(self):
        q = dict(id='case', question='Q', injection='J')
        row = dict(q, rollout_index=0, rlvr_step=0, a_token_ids=[5]*20,
                   a_complete=True, a_ended_at_eos=False, seed=42)
        with patch('sampled_rlvr_experiment.pq.read_table') as read:
            read.return_value.to_pylist.return_value = [row]
            self.assertEqual(load_reference_a('unused', [q], 1, 0)[('case',0)]['seed'],42)
            with self.assertRaises(ValueError): load_reference_a('unused', [q], 2, 0)
            with self.assertRaises(ValueError): load_reference_a('unused', [dict(q,injection='different')], 1, 0)
            read.return_value.to_pylist.return_value = [row,row]
            with self.assertRaises(ValueError): load_reference_a('unused', [q], 1, 0)

if __name__ == '__main__':
    unittest.main()
