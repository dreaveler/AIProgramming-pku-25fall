import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import torchofdreaveler as mytorch

try:
    from torchvision import datasets
except ImportError:  # pragma: no cover
    datasets = None


MNIST_ROOT = os.path.join(os.path.dirname(__file__), "mnist_data")
DEFAULT_DEVICE = mytorch.Device.gpu


def numpy_to_tensor(array: np.ndarray, device=DEFAULT_DEVICE):
    """Convert numpy array to mytorch Tensor on the requested device."""
    contiguous = np.ascontiguousarray(array.astype(np.float32))
    return mytorch.Tensor(contiguous, device=device)


def tensor_to_numpy(t: mytorch.Tensor) -> np.ndarray:
    """Return a standalone numpy copy of a (possibly GPU) Tensor."""
    return np.array(t.cpu().numpy, copy=True)


def random_numpy(shape):
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def load_mnist_numpy(train=True, limit=1024):
    """Download (if needed) MNIST and return normalized numpy arrays."""
    if datasets is None:
        raise RuntimeError("torchvision is required to load MNIST.")
    ds = datasets.MNIST(root=MNIST_ROOT, train=train, download=True)
    limit = len(ds.data) if limit is None else min(limit, len(ds.data))
    images = ds.data[:limit].numpy().astype(np.float32) / 255.0
    images = np.expand_dims(images, 1)  # (N, 1, 28, 28)
    labels = ds.targets[:limit].numpy().astype(np.int64)
    return images, labels


def mnist_tensors(limit=128, device=DEFAULT_DEVICE):
    images, labels = load_mnist_numpy(train=True, limit=limit)
    image_tensor = numpy_to_tensor(images, device=device)
    label_tensor = numpy_to_tensor(labels.astype(np.float32), device=device)
    return image_tensor, label_tensor


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


class TensorInteropTest(unittest.TestCase):
    def test_tensor_creation_and_transfer(self):
        np_data = np.array([[-1., 2.], [-3., 4.]], dtype=np.float32)
        cpu_tensor = mytorch.Tensor(np_data, device=mytorch.Device.cpu)
        gpu_tensor = cpu_tensor.gpu()
        back_to_cpu = gpu_tensor.cpu()

        np.testing.assert_allclose(back_to_cpu.numpy, np_data)


class OperatorParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

    def assert_arrays_close(self, my_tensor, torch_tensor, rtol=1e-4, atol=1e-5):
        np.testing.assert_allclose(tensor_to_numpy(my_tensor), torch_tensor.detach().cpu().numpy(), rtol=rtol, atol=atol)

    def test_relu_matches_torch(self):
        data = random_numpy((4, 5))
        relu = mytorch.Relu()
        my_input = numpy_to_tensor(data)
        my_output = relu.forward(my_input)

        torch_output = torch.from_numpy(data).clamp_min_(0.0)
        self.assert_arrays_close(my_output, torch_output)

    def test_fc_forward_matches_torch(self):
        batch, in_features, out_features = 4, 8, 5
        x = random_numpy((batch, in_features))
        w = random_numpy((in_features, out_features))
        b = random_numpy((1, out_features))

        my_x = numpy_to_tensor(x)
        my_w = numpy_to_tensor(w)
        my_b = numpy_to_tensor(b)
        my_out = mytorch.Tensor.zeros([batch, out_features], DEFAULT_DEVICE)
        mytorch.fc_forward(my_x, my_w, my_b, my_out)

        torch_out = torch.from_numpy(x) @ torch.from_numpy(w) + torch.from_numpy(b)
        self.assert_arrays_close(my_out, torch_out)

    def test_fc_backward_matches_torch(self):
        batch, in_features, out_features = 3, 6, 4
        x = random_numpy((batch, in_features))
        w = random_numpy((in_features, out_features))
        b = random_numpy((1, out_features))
        grad_output = random_numpy((batch, out_features))

        my_x = numpy_to_tensor(x)
        my_w = numpy_to_tensor(w)
        my_b = numpy_to_tensor(b)
        my_out = mytorch.Tensor.zeros([batch, out_features], DEFAULT_DEVICE)
        mytorch.fc_forward(my_x, my_w, my_b, my_out)

        grad_output_t = numpy_to_tensor(grad_output)
        grad_input = mytorch.Tensor.zeros([batch, in_features], DEFAULT_DEVICE)
        grad_weights = mytorch.Tensor.zeros([in_features, out_features], DEFAULT_DEVICE)
        grad_bias = mytorch.Tensor.zeros([1, out_features], DEFAULT_DEVICE)

        mytorch.fc_backward(my_x, my_w, my_b, my_out, grad_output_t, grad_input, grad_weights, grad_bias)

        torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        torch_w = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        torch_b = torch.tensor(b.squeeze(0), dtype=torch.float32, requires_grad=True)
        torch_out = torch_x @ torch_w + torch_b
        torch_out.backward(torch.tensor(grad_output, dtype=torch.float32))

        np.testing.assert_allclose(tensor_to_numpy(grad_input), torch_x.grad.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(tensor_to_numpy(grad_weights), torch_w.grad.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(tensor_to_numpy(grad_bias), torch_b.grad.numpy()[None, :], rtol=1e-4, atol=1e-5)

    def test_convolution_matches_torch(self):
        batch, cin, cout, height, width = 2, 3, 4, 5, 5
        img = random_numpy((batch, cin, height, width))
        kernel = random_numpy((cout, cin, 3, 3))

        my_img = numpy_to_tensor(img)
        my_kernel = numpy_to_tensor(kernel)
        my_out = mytorch.Tensor.zeros([batch, cout, height, width], DEFAULT_DEVICE)
        mytorch.convolve(my_img, my_kernel, my_out, padding=1, stride=1)

        torch_out = F.conv2d(torch.tensor(img), torch.tensor(kernel), padding=1)
        self.assert_arrays_close(my_out, torch_out)

    def test_convolution_backward_matches_torch(self):
        batch, cin, cout, height, width = 2, 3, 4, 5, 5
        img = random_numpy((batch, cin, height, width))
        kernel = random_numpy((cout, cin, 3, 3))
        grad_output = random_numpy((batch, cout, height, width))

        my_img = numpy_to_tensor(img)
        my_kernel = numpy_to_tensor(kernel)
        my_out = mytorch.Tensor.zeros([batch, cout, height, width], DEFAULT_DEVICE)
        mytorch.convolve(my_img, my_kernel, my_out)

        grad_output_t = numpy_to_tensor(grad_output)
        grad_img = mytorch.Tensor.zeros([batch, cin, height, width], DEFAULT_DEVICE)
        grad_kernel = mytorch.Tensor.zeros([cout, cin, 3, 3], DEFAULT_DEVICE)
        mytorch.convolve_backward(my_img, my_kernel, grad_output_t, grad_img, grad_kernel)

        torch_img = torch.tensor(img, dtype=torch.float32, requires_grad=True)
        torch_kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=True)
        torch_out = F.conv2d(torch_img, torch_kernel, padding=1)
        torch_out.backward(torch.tensor(grad_output, dtype=torch.float32))

        np.testing.assert_allclose(tensor_to_numpy(grad_img), torch_img.grad.numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(tensor_to_numpy(grad_kernel), torch_kernel.grad.numpy(), rtol=1e-3, atol=1e-4)

    def test_maxpool_forward_matches_torch(self):
        batch, channels, height, width = 2, 2, 4, 4
        data = random_numpy((batch, channels, height, width))

        my_input = numpy_to_tensor(data)
        my_output = mytorch.Tensor.zeros([batch, channels, height // 2, width // 2], DEFAULT_DEVICE)
        mask = mytorch.Tensor.zeros([batch, channels, height // 2, width // 2], DEFAULT_DEVICE)
        mytorch.maxpooling(my_input, my_output, mask)

        torch_out = F.max_pool2d(torch.tensor(data), kernel_size=2, stride=2)
        self.assert_arrays_close(my_output, torch_out)

    def test_maxpool_backward_matches_torch(self):
        batch, channels, height, width = 2, 2, 4, 4
        data = random_numpy((batch, channels, height, width))
        grad_output = random_numpy((batch, channels, height // 2, width // 2))

        my_input = numpy_to_tensor(data)
        my_output = mytorch.Tensor.zeros([batch, channels, height // 2, width // 2], DEFAULT_DEVICE)
        mask = mytorch.Tensor.zeros([batch, channels, height // 2, width // 2], DEFAULT_DEVICE)
        mytorch.maxpooling(my_input, my_output, mask)

        grad_output_t = numpy_to_tensor(grad_output)
        grad_input = mytorch.Tensor.zeros([batch, channels, height, width], DEFAULT_DEVICE)
        mytorch.maxpooling_backward(grad_output_t, mask, grad_input)

        torch_input = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        torch_out = F.max_pool2d(torch_input, kernel_size=2, stride=2)
        torch_out.backward(torch.tensor(grad_output, dtype=torch.float32))

        np.testing.assert_allclose(tensor_to_numpy(grad_input), torch_input.grad.numpy(), rtol=1e-4, atol=1e-5)

    def test_softmax_matches_torch(self):
        batch, classes = 4, 6
        data = random_numpy((batch, classes))
        my_input = numpy_to_tensor(data)
        my_output = mytorch.Tensor.zeros([batch, classes], DEFAULT_DEVICE)
        mytorch.softmax(my_input, my_output)

        torch_out = F.softmax(torch.tensor(data), dim=1)
        self.assert_arrays_close(my_output, torch_out)

    def test_cross_entropy_matches_manual(self):
        batch, classes = 4, 5
        logits = random_numpy((batch, classes))
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, classes, size=(batch,), dtype=np.int64)

        my_probs = numpy_to_tensor(probs)
        my_labels = numpy_to_tensor(labels.astype(np.float32))
        my_loss = mytorch.Tensor.zeros([batch], DEFAULT_DEVICE)
        mytorch.cross_entropy_loss(my_probs, my_labels, my_loss)

        expected = -np.log(probs[np.arange(batch), labels])
        np.testing.assert_allclose(tensor_to_numpy(my_loss), expected, rtol=1e-4, atol=1e-5)

    def test_softmax_cross_entropy_backward_matches_manual(self):
        batch, classes = 3, 4
        logits = random_numpy((batch, classes))
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, classes, size=(batch,), dtype=np.int64)

        my_probs = numpy_to_tensor(probs)
        my_labels = numpy_to_tensor(labels.astype(np.float32))
        grad = mytorch.Tensor.zeros([batch, classes], DEFAULT_DEVICE)
        mytorch.softmax_cross_entropy_backward(my_probs, my_labels, grad)

        expected = probs.copy()
        expected[np.arange(batch), labels] -= 1.0
        np.testing.assert_allclose(tensor_to_numpy(grad), expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
