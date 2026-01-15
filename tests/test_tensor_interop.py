import unittest

import numpy as np
import torch,torchvision
import torchofdreaveler as mytorch


class TensorInteropTest(unittest.TestCase):
    def test_tensor_creation_and_transfer(self):
        np_data = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        cpu_tensor = mytorch.Tensor(np_data, device=mytorch.Device.cpu)
        gpu_tensor = cpu_tensor.gpu()
        back_to_cpu = gpu_tensor.cpu()

        np.testing.assert_allclose(back_to_cpu.numpy, np_data)
