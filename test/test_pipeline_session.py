"""Exercise the optional session mode without importing ROS or robot drivers."""
import ast
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from hri_legible_motion import experiment as E


def session_class():
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'experiment_node.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('Aborted', 'ExperimentNode')]
    scope = dict(E=E, json=json, os=__import__('os'), threading=threading,
                 queue=queue, STATUS_TIMEOUT={'executing': 10., 'home': 40.},
                 require_operator=lambda prompt: True)
    scope['rospy'] = NS(get_time=lambda: 98., sleep=lambda seconds: None,
                        is_shutdown=lambda: False, Time=NS(from_sec=lambda s: s))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    return scope['ExperimentNode'], scope


class PipelineSessionTests(unittest.TestCase):
    def setUp(self):
        cls, self.scope = session_class()
        self.node = cls.__new__(cls)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        n = self.node
        n.out_dir, n.duration, n.participant = self.tmp.name, 4., 'P07'
        n.e = dict(ready_time=1., rating_timeout=1., score_time=0.)
        n.lock, n.statuses = threading.Lock(), queue.Queue()
        n.record, n.accepting = None, False
        n.blocks = [dict(name='B1')]
        n.colors = ['orange']
        n.log = E.SessionLog(n.out_dir)
        n.show, n.prompt, n.go_home = Mock(), Mock(return_value='go'), Mock()
        n.wait_rating = Mock(return_value=5)
        self.proposal = NS(source=NS(trial_id='session-attempt-unique',
                                    config_sha256='config', layout=NS(layout_id='layout')),
                           serialize=lambda stream: stream.write(b'proposal bytes'))
        n.pipeline = Mock()
        n.pipeline.prepare.return_value = self.proposal
        n.pipeline.validate.return_value = NS(validation_id='valid-1')
        n.pipeline.wait_status.side_effect = [('executing', 100.), ('done', 105.)]
        self.trial = dict(trial_id=1, motion='legible', target_id=0,
                          phase='main', block='A', index_in_phase=0)

    def test_preparation_precedes_choices_and_scoring_uses_measured_motion(self):
        def prepare(trial):
            self.assertFalse(self.node.accepting)
            self.assertIsNone(self.node.record)
            return self.proposal
        self.node.pipeline.prepare.side_effect = prepare
        record = self.node.run_trial(self.trial)
        self.assertEqual(record.status, 'done')
        self.assertEqual((record.t_start, record.t_done, record.measured_duration), (100., 105., 5.))
        self.node.pipeline.request_motion.assert_called_once_with('run', operator_confirmed=True)
        self.assertFalse(self.node.accepting)
        self.assertFalse(hasattr(self.node, 'trial_pub'))  # No legacy motion trigger.
        self.node.log_trial(record)
        attempts = [json.loads(line) for line in Path(self.tmp.name, 'pipeline_attempts.jsonl').read_text().splitlines()]
        self.assertEqual([a['event'] for a in attempts], ['prepared', 'result'])
        self.assertEqual(attempts[1]['pipeline_trial_id'], self.proposal.source.trial_id)
        self.assertEqual(attempts[1]['measured_duration'], 5.)
        self.assertEqual(Path(self.tmp.name, 'session-attempt-unique.proposal').read_bytes(), b'proposal bytes')

    def test_validator_failure_never_starts_motion_or_accepts_choices(self):
        self.node.pipeline.validate.side_effect = RuntimeError('no validator')
        with self.assertRaisesRegex(RuntimeError, 'validator'):
            self.node.run_trial(self.trial)
        self.node.pipeline.request_motion.assert_not_called()
        self.assertFalse(self.node.accepting)
        self.assertIsNone(self.node.record)

    def test_operator_declining_does_not_request_motion(self):
        self.node.prompt.return_value = 'no'
        with self.assertRaises(self.scope['Aborted']):
            self.node.run_trial(self.trial)
        self.node.pipeline.request_motion.assert_not_called()

    def test_missing_onset_is_failure_no_score_or_automatic_return(self):
        self.node.pipeline.wait_status.side_effect = [(None, None)]
        record = self.node.run_trial(self.trial)
        self.assertEqual(record.status, 'timeout')
        self.assertIsNone(record.t_start)
        self.node.wait_rating.assert_not_called()
        self.node.go_home.assert_not_called()
        self.node.pipeline.request_motion.assert_called_once()

    def test_service_failure_stops_button_acceptance(self):
        self.node.pipeline.request_motion.side_effect = RuntimeError('state unknown')
        with self.assertRaises(self.scope['Aborted']):
            self.node.run_trial(self.trial)
        self.assertFalse(self.node.accepting)

    def test_explicit_config_and_pipeline_mode_are_optional_and_no_cache_in_pipeline(self):
        # Check constructor branches using a fake package/file, without ROS imports.
        import sys
        import yaml
        from unittest.mock import patch
        cfg = dict(experiment=dict(assignment_seed=2026, colors=['orange', 'red', 'yellow', 'green'],
                                  button_to_block=[3, 2, 1, 0], reps_per_block=4, max_same_in_row=2,
                                  output_dir=self.tmp.name),
                   motion=dict(duration=4), blocks=[dict(id=i) for i in range(4)])
        config = Path(self.tmp.name, 'chosen.yaml')
        config.write_text(yaml.safe_dump(cfg))
        cls, scope = session_class()
        scope.update(rospkg=NS(RosPack=lambda: NS(get_path=lambda name: '/unused')),
                     yaml=yaml, trajectories=Mock(), sys=sys, String=object, Int32=object,
                     Empty=object, ChoiceEvent=object, TrialInfo=object)
        scope['rospy'].Publisher = Mock()
        scope['rospy'].Subscriber = Mock()
        client = NS(ExperimentClient=Mock(return_value=Mock()))
        with patch.dict(sys.modules, {'legibility_ros': NS(), 'legibility_ros.experiment_client': client}):
            with patch.object(cls, 'write_session_yaml'):
                node = cls('P07', 0, str(config), True)
        self.assertEqual(node.cfg, cfg)
        scope['trajectories'].read_meta.assert_not_called()
        published = [c.args[0] for c in scope['rospy'].Publisher.call_args_list]
        subscribed = [c.args[0] for c in scope['rospy'].Subscriber.call_args_list]
        self.assertNotIn('/trial_start', published)
        self.assertNotIn('/motion_home', published)
        self.assertNotIn('/motion_status', subscribed)


if __name__ == '__main__':
    unittest.main()
