import unittest
import torch


class LaunchTests(unittest.TestCase):
    def test_no_gpu_never_falls_back_to_cpu_training(self):
        from runner import require_cuda
        if torch.cuda.is_available():
            self.skipTest('This test covers a CUDA-unavailable environment')
        with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            require_cuda()


if __name__ == '__main__':
    unittest.main()
