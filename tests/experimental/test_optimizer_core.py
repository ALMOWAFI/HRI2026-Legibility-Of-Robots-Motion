"""Report-derived numerical tests plus independent rejection/edge-case checks."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experimental.legibility_optimizer import optimizer_core as core
from experimental.legibility_optimizer.demo import identity_fk, synthetic_case


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q, cls.t, cls.goals = synthetic_case()
        cls.limits = dict(joint_lower=[-2, -2, -1], joint_upper=[2, 2, 1],
                          max_joint_deviation=[0.25, 0.25, 0.1])
        cls.config = core.OptimizerConfig(cost_ratio=1.45, deviation_weight=5e-4,
                                         min_legibility_improvement=1e-7,
                                         maxiter=300, ftol=1e-9)
        cls.actual = core.optimize_joint_path(cls.q, cls.t, cls.goals, 0, identity_fk,
                                             config=cls.config, **cls.limits)

    def run_case(self, *, q=None, t=None, goals=None, index=0, fk=identity_fk, **kwargs):
        options = dict(self.limits, config=self.config)
        options.update(kwargs)
        return core.optimize_joint_path(self.q if q is None else q,
                                        self.t if t is None else t,
                                        self.goals if goals is None else goals,
                                        index, fk, **options)

    def solver_result(self, candidate=None, *, success=True, variables=None, **kwargs):
        z = (self.q if candidate is None else candidate)[1:-1].ravel() if variables is None else variables
        fake = SimpleNamespace(x=z, success=success, message="injected solver status")
        with patch.object(core, "minimize", return_value=fake):
            return self.run_case(**kwargs)

    def test_uniform_prior_at_start(self):
        posterior = core.goal_posterior(self.q[0], self.q[0], self.goals, beta=4)
        np.testing.assert_allclose(posterior, [1 / 3] * 3, atol=1e-12)

    def test_nonuniform_prior_and_normalization(self):
        prior = [0.6, 0.3, 0.1]
        np.testing.assert_allclose(core.goal_posterior(
            self.q[0], self.q[0], self.goals, beta=4, priors=prior), prior)
        for beta in (1e-6, 4, 1e6):
            with self.subTest(beta=beta):
                p = core.goal_posterior(self.q[-1], self.q[0], self.goals, beta=beta)
                self.assertTrue(np.all(np.isfinite(p)))
                self.assertTrue(np.all((p >= 0) & (p <= 1)))
                self.assertAlmostEqual(float(p.sum()), 1.0, places=14)

    def test_invalid_priors_and_temperature_rejected(self):
        for prior in ([0, 0.5, 0.5], [1, 1, 1], [np.nan, 0.5, 0.5], [1]):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                core.goal_posterior(self.q[0], self.q[0], self.goals, beta=4, priors=prior)
        for beta in (0, -1, np.inf, np.nan):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                core.goal_posterior(self.q[0], self.q[0], self.goals, beta=beta)

    def test_real_solver_improves_and_accepts(self):
        self.assertTrue(self.actual.solver_success, self.actual.message)
        self.assertTrue(self.actual.numerical_accepted, self.actual.message)
        self.assertGreater(self.actual.optimized_legibility,
                           self.actual.baseline_legibility + self.config.min_legibility_improvement)

    def test_endpoints_and_timestamps_preserved_exactly(self):
        np.testing.assert_array_equal(self.actual.positions[[0, -1]], self.q[[0, -1]])
        np.testing.assert_array_equal(self.actual.times, self.t)

    def test_joint_budget_and_local_constraints_independently(self):
        q = self.actual.positions
        self.assertTrue(np.all(q >= np.array(self.limits["joint_lower"]) - 1e-7))
        self.assertTrue(np.all(q <= np.array(self.limits["joint_upper"]) + 1e-7))
        self.assertTrue(np.all(np.abs(q - self.q)
                               <= np.array(self.limits["max_joint_deviation"]) + 1e-7))
        # Independently compute the cost rather than trust the result's flags.
        base_cost = sum(float(np.dot(b - a, b - a)) / dt
                        for a, b, dt in zip(self.q[:-1], self.q[1:], np.diff(self.t)))
        cost = sum(float(np.dot(b - a, b - a)) / dt
                   for a, b, dt in zip(q[:-1], q[1:], np.diff(self.t)))
        self.assertLessEqual(cost, self.config.cost_ratio * base_cost + 1e-7)

    def test_solver_success_does_not_mean_physical_validity(self):
        self.assertTrue(self.actual.candidate_ok)
        self.assertIsNone(self.actual.physical_validity)
        self.assertTrue(self.actual.retime_required)
        self.assertFalse(self.actual.execution_ready)
        self.assertIsNone(self.actual.velocity_ok)
        self.assertIsNone(self.actual.acceleration_ok)

    def test_failed_solver_rejected_even_with_good_candidate(self):
        result = self.solver_result(self.actual.positions, success=False)
        self.assertTrue(result.improvement_ok)
        self.assertFalse(result.numerical_accepted)

    def test_no_improvement_rejected(self):
        result = self.solver_result()
        self.assertTrue(result.solver_success)
        self.assertFalse(result.improvement_ok)
        self.assertFalse(result.numerical_accepted)

    def test_nonfinite_or_malformed_solver_output_never_accepts_fallback(self):
        config = replace(self.config, min_legibility_improvement=0)
        for z in (np.full(self.q[1:-1].size, np.nan),
                  np.full(self.q[1:-1].size, np.inf), np.zeros(2)):
            with self.subTest(shape=z.shape):
                result = self.solver_result(variables=z, config=config)
                self.assertFalse(result.solver_output_valid)
                self.assertFalse(result.numerical_accepted)
                np.testing.assert_array_equal(result.positions, self.q)

    def test_postcheck_joint_limit_violation_rejected(self):
        q = self.q.copy()
        q[3, 0] = 2.1
        result = self.solver_result(q)
        self.assertFalse(result.joint_limit_ok)
        self.assertFalse(result.numerical_accepted)

    def test_postcheck_local_deviation_violation_rejected(self):
        q = self.q.copy()
        q[3, 0] += 0.3
        result = self.solver_result(q)
        self.assertTrue(result.joint_limit_ok)
        self.assertFalse(result.local_deviation_ok)
        self.assertFalse(result.numerical_accepted)

    def test_postcheck_motion_budget_violation_rejected(self):
        q = self.q.copy()
        q[1:-1, 2] = np.array([1, -1, 1, -1, 1, -1, 1]) * 0.09
        result = self.solver_result(q, config=replace(self.config, cost_ratio=1.0))
        self.assertTrue(result.local_deviation_ok)
        self.assertTrue(result.joint_limit_ok)
        self.assertFalse(result.cost_budget_ok)
        self.assertFalse(result.numerical_accepted)

    def test_invalid_baseline_and_bound_vectors_rejected(self):
        invalid = [dict(joint_lower=[0, 0, 0], joint_upper=[0.5, 1, 1]),
                   dict(joint_lower=[2, 2, 2]), dict(joint_lower=[np.nan, -2, -1]),
                   dict(joint_upper=[2, 2]), dict(max_joint_deviation=[0, 1, 1]),
                   dict(velocity_limits=[0, 1, 1]), dict(acceleration_limits=[1, np.inf, 1])]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_case(**kwargs)

    def test_invalid_timestamps_rejected(self):
        repeated = self.t.copy()
        repeated[4] = repeated[3]
        for t in (repeated, self.t[::-1], self.t - 1, self.t[:-1],
                  np.full_like(self.t, np.nan), np.full_like(self.t, np.inf)):
            with self.subTest(t=t), self.assertRaises(ValueError):
                self.run_case(t=t)

    def test_nonuniform_time_cost_acceleration_and_offset(self):
        q = np.array([[0.0], [1.0], [5.0]])
        t = np.array([5.0, 6.0, 8.0])
        self.assertEqual(core.joint_motion_cost(q, t), 9.0)
        np.testing.assert_array_equal(core.segment_velocities(q, t), [[1], [2]])
        np.testing.assert_allclose(core.interior_accelerations(q, t), [[2 / 3]])
        shifted = self.solver_result(self.actual.positions, t=self.t + 17)
        np.testing.assert_array_equal(shifted.times, self.t + 17)
        self.assertAlmostEqual(shifted.optimized_legibility, self.actual.optimized_legibility)
        tau, _ = core.true_goal_probability_trace(self.q, self.t + 17, self.goals, 0, identity_fk)
        np.testing.assert_allclose(tau, self.t / 4)

    def test_dynamic_limits_real_solver_and_postchecks(self):
        result = self.run_case(velocity_limits=[0.6, 0.6, 0.3],
                               acceleration_limits=[2, 2, 2])
        self.assertTrue(result.numerical_accepted, result.message)
        self.assertTrue(result.velocity_ok)
        self.assertTrue(result.acceleration_ok)
        q = self.q.copy()
        q[1, 0] += 0.1
        rejected = self.solver_result(q, velocity_limits=[0.3, 0.3, 0.3],
                                      acceleration_limits=[0.1, 0.1, 0.1])
        self.assertFalse(rejected.velocity_ok)
        self.assertFalse(rejected.acceleration_ok)
        self.assertFalse(rejected.numerical_accepted)

    def test_infeasible_dynamic_baseline_rejected(self):
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.run_case(velocity_limits=[0.1, 0.1, 0.1])
        q = self.q.copy()
        q[1, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "acceleration"):
            self.run_case(q=q, acceleration_limits=[0.01, 0.01, 0.01])

    def test_invalid_shapes_goals_and_target_rejected(self):
        for kwargs in (dict(q=self.q[:2], t=self.t[:2]), dict(q=np.zeros((9, 0))),
                       dict(q=np.full_like(self.q, np.nan)), dict(goals=[[0, 0, 0]]),
                       dict(goals=[[0, 0], [1, 1]]), dict(index=-1), dict(index=3),
                       dict(index=0.5), dict(index=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_case(**kwargs)

    def test_zero_motion_cost_rejected(self):
        with self.assertRaisesRegex(ValueError, "near-zero"):
            self.run_case(q=np.zeros_like(self.q))

    def test_invalid_configuration_rejected(self):
        for kwargs in (dict(beta=np.nan), dict(cost_ratio=np.inf), dict(cost_ratio=0.9),
                       dict(deviation_weight=-1), dict(early_weight_power=np.nan),
                       dict(min_legibility_improvement=-1), dict(ftol=0),
                       dict(constraint_tolerance=np.inf), dict(maxiter=0), dict(maxiter=1.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                core.OptimizerConfig(**kwargs)

    def test_bad_fk_and_fk_exceptions_propagate(self):
        for fk in (lambda q: [0, 0], lambda q: [0, np.nan, 0]):
            with self.assertRaisesRegex(ValueError, "FK output"):
                self.run_case(fk=fk)
        def unavailable(q):
            raise NotImplementedError("missing robot model")
        with self.assertRaisesRegex(NotImplementedError, "missing robot model"):
            self.run_case(fk=unavailable)

    def test_legibility_is_computed_through_fk(self):
        result = self.run_case(fk=lambda q: np.zeros(3))
        self.assertAlmostEqual(result.optimized_legibility, 1 / 3)
        self.assertFalse(result.numerical_accepted)

    def test_inputs_are_not_modified_or_aliased(self):
        q, t, goals = synthetic_case()
        saved = [a.copy() for a in (q, t, goals)]
        result = self.solver_result(self.actual.positions, q=q, t=t, goals=goals)
        for actual, expected in zip((q, t, goals), saved):
            np.testing.assert_array_equal(actual, expected)
        result.positions[:] = 99
        result.times[:] = 99
        np.testing.assert_array_equal(q, saved[0])
        np.testing.assert_array_equal(t, saved[1])

    def test_densification_preserves_every_original_waypoint(self):
        q, t = core.densify_joint_path(self.q, self.t + 7, 5)
        np.testing.assert_array_equal(q[::5], self.q)
        np.testing.assert_array_equal(t[::5], self.t + 7)
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertEqual(len(t), (len(self.t) - 1) * 5 + 1)
        for samples in (0, -1, 1.5, True):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                core.densify_joint_path(self.q, self.t, samples)


if __name__ == "__main__":
    unittest.main()
