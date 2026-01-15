#include "./tensor.cuh"
#include "./layers/activation.cuh"
#include "./layers/nn.cuh"
#include "./ops/ops.cuh"
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
        .def("reshape", [](const Tensor& t, const std::vector<int>& new_shape) {
            return Ops::reshape(t, new_shape);
        }, py::arg("new_shape"), "Reshape tensor")
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

    // 4. Bind functions in nn namespace
    m.def("fc_forward", &nn::FC_forward, "Fully-connected layer forward pass",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"));

    m.def("fc_backward", &nn::FC_backward, "Fully-connected layer backward pass",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"),
        py::arg("grad_output"), py::arg("grad_input"), py::arg("grad_weights"), py::arg("grad_bias"));

    m.def("convolve", &nn::convolve, "Convolution layer forward pass",
        py::arg("img"), py::arg("kernel"), py::arg("output"),
        py::arg("padding") = 1, py::arg("stride") = 1);

    m.def("convolve_backward", &nn::convolve_backward, "Convolution layer backward pass",
        py::arg("input"), py::arg("kernel"), py::arg("grad_y"),
        py::arg("grad_input"), py::arg("grad_kernel"));

    m.def("maxpooling", &nn::maxpooling, "2x2 Max Pooling (stride=2)",
        py::arg("input"), py::arg("output"), py::arg("mask"));

    m.def("maxpooling_backward", &nn::maxpooling_backward, "Max Pooling backward pass",
        py::arg("grad_y"), py::arg("mask"), py::arg("grad_x"));

    m.def("softmax", &nn::softmax, "Softmax function",
        py::arg("input"), py::arg("output"));

    m.def("cross_entropy_loss", &nn::crossentropyloss, "Cross Entropy Loss",
        py::arg("input"), py::arg("label"), py::arg("output"));

    m.def("softmax_cross_entropy_backward", &nn::softmaxsel_backward, "Softmax with Cross Entropy backward pass",
        py::arg("soutput"), py::arg("label"), py::arg("grad_sinput"));

    m.def("batchnorm_forward", &nn::batchnorm_forward, "BatchNorm forward",
        py::arg("input"), py::arg("gamma"), py::arg("beta"),
        py::arg("output"), py::arg("mean"), py::arg("var"),
        py::arg("running_mean"), py::arg("running_var"),
        py::arg("training") = true, py::arg("momentum") = 0.1f,
        py::arg("eps") = 1e-5f);

    m.def("batchnorm_backward", &nn::batchnorm_backward, "BatchNorm backward",
        py::arg("grad_out"), py::arg("input"), py::arg("gamma"),
        py::arg("mean"), py::arg("var"),
        py::arg("grad_input"), py::arg("grad_gamma"), py::arg("grad_beta"),
        py::arg("eps") = 1e-5f);

    // 5. Bind basic tensor ops (forward only)
    m.def("add", &Ops::add, "Elementwise add", py::arg("a"), py::arg("b"));
    m.def("add_scalar", &Ops::add_scalar, "Add scalar", py::arg("a"), py::arg("scalar"));
    m.def("multiply", &Ops::multiply, "Elementwise multiply", py::arg("a"), py::arg("b"));
    m.def("mul_scalar", &Ops::mul_scalar, "Multiply scalar", py::arg("a"), py::arg("scalar"));
    m.def("divide", &Ops::divide, "Elementwise divide", py::arg("a"), py::arg("b"));
    m.def("div_scalar", &Ops::div_scalar, "Divide scalar", py::arg("a"), py::arg("scalar"));
    m.def("power", &Ops::power, "Elementwise power", py::arg("a"), py::arg("b"));
    m.def("power_scalar", &Ops::power_scalar, "Power scalar", py::arg("a"), py::arg("scalar"));
    m.def("negate", &Ops::negate, "Negate", py::arg("a"));
    m.def("exp", &Ops::exp, "Exp", py::arg("a"));
    m.def("log", &Ops::log, "Log", py::arg("a"));
    m.def("reshape", &Ops::reshape, "Reshape", py::arg("a"), py::arg("shape"));

    m.def("transpose", [](const Tensor& a, py::object axes_obj) {
        if (axes_obj.is_none()) {
            return Ops::transpose(a, {});
        }
        if (py::isinstance<py::tuple>(axes_obj) || py::isinstance<py::list>(axes_obj)) {
            return Ops::transpose(a, axes_obj.cast<std::vector<int>>());
        }
        throw std::runtime_error("transpose: axes must be list/tuple or None");
    }, "Transpose", py::arg("a"), py::arg("axes") = py::none());

    m.def("broadcast_to", &Ops::broadcast_to, "Broadcast to shape", py::arg("a"), py::arg("shape"));

    m.def("summation", [](const Tensor& a, py::object axes_obj) {
        if (axes_obj.is_none()) {
            return Ops::summation(a, {});
        }
        if (py::isinstance<py::int_>(axes_obj)) {
            return Ops::summation(a, std::vector<int>{axes_obj.cast<int>()});
        }
        if (py::isinstance<py::tuple>(axes_obj) || py::isinstance<py::list>(axes_obj)) {
            return Ops::summation(a, axes_obj.cast<std::vector<int>>());
        }
        throw std::runtime_error("summation: axes must be int, list/tuple, or None");
    }, "Summation", py::arg("a"), py::arg("axes") = py::none());

    m.def("matmul", &Ops::matmul, "MatMul", py::arg("a"), py::arg("b"));
}
