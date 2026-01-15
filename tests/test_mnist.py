import unittest

import numpy as np

from tests.common import datasets, load_mnist_numpy, numpy_to_tensor, tensor_to_numpy


class MNISTInteropTest(unittest.TestCase):
    @unittest.skipIf(datasets is None, "torchvision is required for MNIST tests")
    def test_mnist_numpy_to_tensor(self):
        images, labels = load_mnist_numpy(limit=32)
        image_tensor = numpy_to_tensor(images)
        label_tensor = numpy_to_tensor(labels.astype(np.float32))

        np.testing.assert_allclose(tensor_to_numpy(image_tensor)[:8], images[:8], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(tensor_to_numpy(label_tensor)[:8].astype(np.int64), labels[:8])
        self.assertEqual(image_tensor.shape, [32, 1, 28, 28])
        self.assertEqual(label_tensor.shape, [32])
