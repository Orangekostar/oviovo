"""Authoring-time tests: pure numerical/CPU toy gradients, no real FC or GPU."""
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from numerics import unit, residual, softmax, absolute_route, delta_output, r_gate, choose_checkpoint

torch.set_num_threads(1)


class ReferenceTests(unittest.TestCase):
    def test_01_old_within_group_shift_cancels(self):
        q = np.array([.2, -.7, .8])
        np.testing.assert_allclose(softmax(q), softmax(q - 12), atol=1e-14)

    def test_02_null_not_biased_by_group_count(self):
        for groups in (1, 8, 64):
            null, a = absolute_route(np.zeros((4, groups)), np.ones(groups))
            np.testing.assert_allclose(null, .5, atol=1e-14)
            np.testing.assert_allclose(a.sum(1), .5, atol=1e-14)

    def test_03_absolute_group_membership_controls_mass(self):
        e = np.array([[.4, -.2, .1], [-.1, .2, -.7]])
        n1, a1 = absolute_route(e, np.ones(3))
        n2, a2 = absolute_route(e, np.array([.01, 1, 1]))
        self.assertTrue(np.all(a2[:, 0] < a1[:, 0]))
        n3, a3 = absolute_route(e, np.full(3, .01))
        self.assertTrue(np.all(n3 > n1))
        self.assertTrue(np.all(a3.sum(1) < a1.sum(1)))

    def test_04_no_groups_or_all_rejected_returns_base(self):
        base = unit(np.array([1., 2., -1.]))
        for groups in (0, 3):
            null, a = absolute_route(np.zeros((4, groups)), np.zeros(groups))
            np.testing.assert_array_equal(null, np.ones(4))
            out = delta_output(base, np.zeros((groups, 3)), a, np.eye(3))
            np.testing.assert_allclose(out, base, atol=1e-14)

    def test_05_function_subtraction_is_identity(self):
        rng = np.random.default_rng(3)
        t = unit(rng.normal(size=8)); initial = unit(rng.normal(size=8))
        np.testing.assert_allclose(residual(t, initial, initial), t, atol=1e-14)

    def test_06_function_subtraction_has_first_step_gradient(self):
        torch.manual_seed(4)
        head = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4)).double()
        x = torch.tensor([.4, -.8, .9], dtype=torch.float64)
        old = F.normalize(head(x).detach(), dim=0)
        teacher = F.normalize(torch.tensor([1., .2, .3, -.4], dtype=torch.float64), dim=0)
        z = F.normalize(teacher + (F.normalize(head(x), dim=0) - old), dim=0)
        torch.testing.assert_close(z, teacher)
        F.cross_entropy(z[None], torch.tensor([2])).backward()
        self.assertGreater(float(head[0].weight.grad.norm()), 0)

    def test_07_teacher_wrong_preservation_is_zero(self):
        z = torch.tensor([.4, -.3, .2], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([.1, .8, .2], dtype=torch.float64)
        loss = 0. * (1 - F.cosine_similarity(z, t, dim=0))
        loss.backward()
        torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_08_frozen_phi_preserves_input_gradient(self):
        phi = nn.Linear(4, 3).double().requires_grad_(False)
        x = torch.ones(4, dtype=torch.float64, requires_grad=True)
        phi(x).square().sum().backward()
        self.assertGreater(float(x.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in phi.parameters()))

    def test_09_zero_linear_can_learn_then_upstream_learns(self):
        torch.manual_seed(9)
        upstream = nn.Linear(3, 4).double()
        out = nn.Linear(4, 4, bias=False).double(); nn.init.zeros_(out.weight)
        base = F.normalize(torch.tensor([.2, .4, .6, -.1], dtype=torch.float64), dim=0)
        x = torch.ones(3, dtype=torch.float64)
        optimizer = torch.optim.SGD(list(upstream.parameters()) + list(out.parameters()), lr=.1)
        z = F.normalize(base + .5*out(upstream(x)), dim=0)
        torch.testing.assert_close(z, base)
        F.cross_entropy(z[None], torch.tensor([0])).backward()
        self.assertGreater(float(out.weight.grad.norm()), 0)
        optimizer.step(); optimizer.zero_grad()
        z = F.normalize(base + .5*out(upstream(x)), dim=0)
        F.cross_entropy(z[None], torch.tensor([0])).backward()
        self.assertGreater(float(upstream.weight.grad.norm()), 0)

    def test_10_group_order_does_not_change_sum(self):
        rng = np.random.default_rng(6)
        e = rng.normal(size=(4, 7)); rho = rng.uniform(size=7)
        values = rng.normal(size=(7, 5)); base = unit(rng.normal(size=5))
        p = rng.permutation(7)
        _, a = absolute_route(e, rho); _, b = absolute_route(e[:, p], rho[p])
        np.testing.assert_allclose(delta_output(base, values, a, np.eye(5)),
                                   delta_output(base, values[p], b, np.eye(5)), atol=1e-14)

    def test_11_copy_teacher_is_not_qualified_improvement(self):
        t = {'A': .3, 'M': .4, 'C': .3}
        self.assertFalse(r_gate(t, t, 0))
        self.assertTrue(r_gate({'A': .306, 'M': .4, 'C': .3}, t, 2))
        self.assertFalse(r_gate({'A': .306, 'M': .39, 'C': .3}, t, 2))

    def test_12_selection_uses_accuracy_not_loss_or_step_zero(self):
        rows = [dict(step=0, A=1., M=1., C=1., CE=0.),
                dict(step=250, A=.5, M=.4, C=.5, CE=4.),
                dict(step=500, A=.4, M=.4, C=.5, CE=2.),
                dict(step=1000, A=.5, M=.4, C=.5, CE=4.)]
        self.assertEqual(choose_checkpoint(rows)['step'], 250)


if __name__ == '__main__':
    unittest.main(verbosity=2)
