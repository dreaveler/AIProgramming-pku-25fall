#include "./tensor.cuh"
#include "./function.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>

namespace py = pybind11;

// Helper function to create a Tensor from a py::array
Tensor tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> np_array, Device device) {
    py::buffer_info buf = np_array.request();
    if (buf.ndim == 0) {
        throw std::runtime_error("0-dimensional numpy arrays are not supported");
    }

    std::vector<int> shape(buf.shape.begin(), buf.shape.end());
    std::vector<float> data(static_cast<float*>(buf.ptr), static_cast<float*>(buf.ptr) + buf.size);

    return Tensor(shape, data, device);
}

PYBIND11_MODULE(core, m) {
    m.doc() = "A simple torch-like library with CUDA backend";

    // 1. Bind Device enum
    py::enum_<Device>(m, "Device")
        .value("cpu", Device::cpu)
        .value("gpu", Device::gpu)
        .export_values();

    // 2. Bind Tensor class
    py::class_<Tensor,std::shared_ptr<Tensor>>(m, "Tensor", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<std::vector<int>, Device>(), py::arg("shape"), py::arg("device"))
        .def(py::init(&tensor_from_numpy), py::arg("numpy_array"), py::arg("device"))
        .def_buffer([](Tensor &t) -> py::buffer_info {
            if (t.device != Device::cpu) {
                throw std::runtime_error("Only CPU Tensors can be exposed to Numpy. Please call .cpu() first.");
            }
            return py::buffer_info(
                t.get_ptr(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                t.shape.size(),
                std::vector<size_t>(t.shape.begin(), t.shape.end()),
                [&]() {
                    std::vector<size_t> strides(t.shape.size());
                    size_t s = sizeof(float);
                    for (int i = t.shape.size() - 1; i >= 0; --i) {
                        strides[i] = s;
                        s *= t.shape[i];
                    }
                    return strides;
                }()
            );
        })
        .def_property_readonly("shape", [](const Tensor &t) { return t.shape; })
        .def_property_readonly("device", [](const Tensor &t) { return t.device; })
        .def_property_readonly("N", [](const Tensor &t) { return t.N; })
        .def_property_readonly("numpy", [](Tensor &t) {
            if (t.device != Device::cpu) {
                throw std::runtime_error("Tensor is not on CPU. Please call .cpu() first.");
            }
            return py::array_t<float>(
                {t.shape.begin(), t.shape.end()},
                {},
                t.get_ptr(),
                py::cast(t)
            );
        })
        .def("gpu", &Tensor::gpu, "Move tensor to GPU")
        .def("cpu", &Tensor::cpu, "Move tensor to CPU")
        .def("random", &Tensor::random, "Fill tensor with random numbers [0,1)")
        .def("reshape", &Tensor::reshape, py::arg("new_shape"), "Reshape tensor without copying data")
        .def("__repr__", [](const Tensor &t) {
            std::stringstream ss;
            auto old_buf = std::cout.rdbuf(ss.rdbuf());
            Tensor temp = t;
            temp.print();
            std::cout.rdbuf(old_buf);
            std::string device_str = (t.device == Device::gpu) ? "gpu" : "cpu";
            return "Tensor(" + ss.str() + ", device='" + device_str + "')";
        })
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("device"), "Create a tensor of all ones")
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("device"), "Create a tensor of all zeros")
        .def("__eq__", &Tensor::operator==, "Element-wise comparison of two tensors");

    // 3. Bind object-oriented layers
    py::class_<Function>(m, "Function")
        .def("forward", &Function::forward)
        .def("backward", &Function::backward);

    py::class_<Relu, Function>(m, "Relu")
        .def(py::init<>());

    py::class_<Sigmoid, Function>(m, "Sigmoid")
        .def(py::init<>());

    // 4. Bind functions in HW3 namespace
    m.def("fc_forward", &HW3::FC_forward, "Fully-connected layer forward pass",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"));

    m.def("fc_backward", &HW3::FC_backward, "Fully-connected layer backward pass",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"),
        py::arg("grad_output"), py::arg("grad_input"), py::arg("grad_weights"), py::arg("grad_bias"));

    m.def("convolve", &HW3::convolve, "Convolution layer forward pass",
        py::arg("img"), py::arg("kernel"), py::arg("output"),
        py::arg("padding") = 1, py::arg("stride") = 1);

    m.def("convolve_backward", &HW3::convolve_backward, "Convolution layer backward pass",
        py::arg("input"), py::arg("kernel"), py::arg("grad_y"),
        py::arg("grad_input"), py::arg("grad_kernel"));

    m.def("maxpooling", &HW3::maxpooling, "2x2 Max Pooling (stride=2)",
        py::arg("input"), py::arg("output"), py::arg("mask"));

    m.def("maxpooling_backward", &HW3::maxpooling_backward, "Max Pooling backward pass",
        py::arg("grad_y"), py::arg("mask"), py::arg("grad_x"));

    m.def("softmax", &HW3::softmax, "Softmax function",
        py::arg("input"), py::arg("output"));

    m.def("cross_entropy_loss", &HW3::crossentropyloss, "Cross Entropy Loss",
        py::arg("input"), py::arg("label"), py::arg("output"));

    m.def("softmax_cross_entropy_backward", &HW3::softmaxsel_backward, "Softmax with Cross Entropy backward pass",
        py::arg("soutput"), py::arg("label"), py::arg("grad_sinput"));
}