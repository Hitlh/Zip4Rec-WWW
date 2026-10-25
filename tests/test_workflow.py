import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from workflow import Workflow
from ranking.pipeline import checkpoint_schedule


class WorkflowTests(unittest.TestCase):
    def test_sft_schedule_follows_actual_data_size(self):
        self.assertEqual(checkpoint_schedule(2516), (158, [158, 316, 474]))
        self.assertEqual(checkpoint_schedule(34), (3, [3, 6, 9]))
        with self.assertRaises(ValueError):
            checkpoint_schedule(0)

    def test_completed_stage_can_resume_and_modified_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            output = p / 'file.txt'
            command = [sys.executable, '-c',
                       'from pathlib import Path; import sys; Path(sys.argv[1]).open("x").write("done")', str(output)]
            first = Workflow(p / 'run', {'dataset': 'fixture'})
            first.run('build', command, [output])
            second = Workflow(p / 'run', {'dataset': 'fixture'})
            second.run('build', command, [output])  # Running the command again would fail exclusive creation.
            output.write_text('modified')
            with self.assertRaises(ValueError):
                second.run('build', command, [output])
            with self.assertRaises(ValueError):
                Workflow(p / 'run', {'dataset': 'different'})

    def test_failure_stops_pipeline_and_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            workflow = Workflow(p, {})
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.run('failure', [sys.executable, '-c', 'raise SystemExit(2)'], [p / 'missing'])
            state = json.loads((p / 'workflow_state.json').read_text())
            self.assertEqual(state['stages']['failure']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
