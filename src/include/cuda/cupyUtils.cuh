#pragma once

#include <pybind11/numpy.h>          // For NumPy/CuPy array handling
#include <pybind11/pybind11.h>       // For Pybind11 bindings
#include <cuda_runtime.h>
namespace py = pybind11;

template <typename T>
T *getCupyPtr(py::object cp_array) {
    return reinterpret_cast<T *>(py::cast<uintptr_t>(cp_array.attr("data").attr("ptr")));
}

template <typename T>
T *getCupyPtrx(py::object cupy_array) {
    auto cuda_interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    auto data_tuple = cuda_interface["data"].cast<py::tuple>();
    uintptr_t data_ptr = py::cast<uintptr_t>(data_tuple[0]);
    return reinterpret_cast<T *>(data_ptr);
}

template <typename T>
py::object createCupyArray(T *ptr, size_t m, size_t n = 0) {
    // Create memory management capsule if requested
    /*
    float32	<f4
    float64	<f8
    int32	<i4
    uint32	<u4
    complex64	<c8
    complex128	<c16
    */

    std::string typeStr;
    if (std::is_same<T, float>::value) {
        typeStr = "<f4"; // float32
    } else if (std::is_same<T, double>::value) {
        typeStr = "<f8"; // float64
    } else if (std::is_same<T, int32_t>::value) {
        typeStr = "<i4"; // int32
    } else if (std::is_same<T, uint32_t>::value) {
        typeStr = "<u4"; // uint32
    } else if (std::is_same<T, std::complex<float>>::value) {
        typeStr = "<c8"; // complex64
    } else if (std::is_same<T, std::complex<double>>::value) {
        typeStr = "<c16"; // complex128
    } else {
        throw std::invalid_argument("Unsupported type for CuPy array.");
    }

    py::object cupy = py::module_::import("cupy");
    py::module pyTypes = py::module::import("types");

    // Create CUDA array interface dictionary
    py::dict interface;
    interface["data"] = py::make_tuple(reinterpret_cast<uintptr_t>(ptr), false);
    if (n == 0) {
        interface["shape"] = py::make_tuple(m);
    } else {
        interface["shape"] = py::make_tuple(m, n);
    }

    interface["typestr"] = typeStr;
    // interface["typestr"] = py::str(py::dtype::of<T>());
    interface["version"] = 2;

    // Create memory management capsule
    py::capsule cleanupCapsule(ptr, [](void *p) {
        T *typeptr = static_cast<T *>(p);
        cudaFree(typeptr);
    });

    // Create a dummy object with the interface and capsule
    py::object temp = pyTypes.attr("SimpleNamespace")();
    temp.attr("__cuda_array_interface__") = interface;
    temp.attr("_memory_capsule") = cleanupCapsule; // Keep capsule alive via ownership

    // Convert to CuPy array
    return cupy.attr("asarray")(temp);
}