"""Check reference gating without importing CUDA or starting a training job."""

import ast
from pathlib import Path
import unittest


class ReferenceGateTest(unittest.TestCase):
    def test_experimental_training_does_not_accept_reference(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'reference_training_status')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'runner.py', 'exec'), namespace)
        check = namespace['reference_training_status']
        stages = ('raw_dynamics_audit', 'fitted_dynamics_audit')
        valid = {stage: dict(passed=True, moving_friction_opposes_velocity=True) for stage in stages}
        self.assertEqual(check(valid)['failed_or_unverified_stages'], [])
        for invalid in ({}, {stage: dict(passed=True) for stage in stages},
                        {**valid, stages[0]: dict(passed=False, moving_friction_opposes_velocity=True)}):
            with self.assertRaises(ValueError):
                check(invalid)
            result = check(invalid, experimental=True)
            self.assertEqual(result['mode'], 'experimental')
            self.assertTrue(result['failed_or_unverified_stages'])
            self.assertFalse(result['walking_accepted'])
        self.assertFalse(check(valid)['walking_accepted'])


if __name__ == '__main__':
    unittest.main()
