import unittest

import torch

from src.losses import BCEDiceLoss


class BCEDiceTests(unittest.TestCase):
    def test_known_overlap_and_batch_reduction(self):
        logits = torch.zeros(2, 1, 1, 2, requires_grad=True)
        targets = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
        expected = torch.log(torch.tensor(2.)) + ((1 - 2 / 3) + (1 - 1 / 2)) / 2
        loss = BCEDiceLoss()(logits, targets)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 0, 0, 0], 0)
        self.assertGreater(logits.grad[0, 0, 0, 1], 0)

    def test_empty_masks_and_extreme_logits(self):
        criterion = BCEDiceLoss()
        for foreground in (0., 1.):
            targets = torch.full((2, 1, 8, 8), foreground)
            correct = torch.full_like(targets, 100. if foreground else -100., requires_grad=True)
            incorrect = -correct.detach()
            good_loss = criterion(correct, targets)
            bad_loss = criterion(incorrect, targets)
            self.assertTrue(torch.isfinite(good_loss))
            self.assertTrue(torch.isfinite(bad_loss))
            self.assertLess(good_loss.item(), 1e-5)
            self.assertGreater(bad_loss.item(), 99)
            good_loss.backward()
            self.assertTrue(torch.isfinite(correct.grad).all())

    def test_near_correct_prediction_has_lower_loss(self):
        targets = torch.tensor([[[[1., 0.], [0., 1.]]]])
        correct = (2 * targets - 1) * 5
        self.assertLess(BCEDiceLoss()(correct, targets), BCEDiceLoss()(-correct, targets))

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            BCEDiceLoss()(torch.zeros(2, 2, 4, 4), torch.zeros(2, 2, 4, 4))


if __name__ == "__main__":
    unittest.main()
