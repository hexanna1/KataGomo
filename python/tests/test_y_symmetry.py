import unittest
import numpy as np
import torch
import data_processing_pytorch as data

PERMS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0))


def reference(tensor, symm, lens):
    if symm == 0:
        return tensor.clone()
    result = torch.zeros_like(tensor)
    a, b, _ = PERMS[symm]
    for i, n in enumerate(lens):
        for y in range(n):
            for x in range(n - y):
                coords = (x, y, n - 1 - x - y)
                result[i, :, coords[b], coords[a]] = tensor[i, :, y, x]
    return result


class YSymmetryTests(unittest.TestCase):
    def test_mixed_sizes_values_and_gradients(self):
        for width in (11, 15):
            lens = list(range(6, 12))
            for symm in range(6):
                source = torch.randn(6, 3, width, width, requires_grad=True)
                indices = data.y_symmetry_indices(width, symm, lens, source.device)
                actual = data.apply_symmetry(source, symm, 'y', lens, indices)
                expected = reference(source, symm, lens)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                weights = torch.randn_like(actual)
                grad = torch.autograd.grad((actual * weights).sum(), source, retain_graph=True)[0]
                expected_grad = torch.autograd.grad((expected * weights).sum(), source)[0]
                torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)

    def test_policy_and_inferred_sizes(self):
        lens = list(range(6, 12))
        mask = np.zeros((6, 11, 11))
        for i, n in enumerate(lens):
            for y in range(n):
                mask[i, y, :n-y] = 1
        self.assertEqual(data.infer_board_lens_from_mask(mask), lens)
        policy = torch.randn(6, 2, 122)
        for symm in range(6):
            actual = data.apply_symmetry_policy(policy, symm, 11, 'y', lens)
            expected = reference(policy[:, :, :-1].reshape(6, 2, 11, 11), symm, lens)
            torch.testing.assert_close(actual[:, :, :-1], expected.reshape(6, 2, 121), rtol=0, atol=0)
            torch.testing.assert_close(actual[:, :, -1], policy[:, :, -1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
