#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <vector>

// Toggle: 0 = float, 1 = half
#define USE_HALF 0

#include <cuda_fp16.h>

#if USE_HALF
    using dtype = half;
#else
    using dtype = float;
#endif

// Host-side conversion utilities
inline dtype float_to_dtype(float f) {
#if USE_HALF
    return __float2half(f);
#else
    return f;
#endif
}

inline float dtype_to_float(dtype d) {
#if USE_HALF
    return __half2float(d);
#else
    return d;
#endif
}

// Vector conversion utilities
inline std::vector<dtype> float_to_dtype_vec(const std::vector<float>& v) {
    std::vector<dtype> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = float_to_dtype(v[i]);
    }
    return result;
}

inline std::vector<float> dtype_to_float_vec(const std::vector<dtype>& v) {
    std::vector<float> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = dtype_to_float(v[i]);
    }
    return result;
}

inline std::vector<half> float_to_half_vec(const std::vector<float>& v) {
    std::vector<half> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = __float2half(v[i]);
    }
    return result;
}

// Device-side conversion (for kernel code)
#if USE_HALF
    #define DTYPE_ZERO __float2half(0.0f)
    #define FLOAT_TO_DTYPE(x) __float2half(x)
    #define DTYPE_TO_FLOAT(x) __half2float(x)
#else
    #define DTYPE_ZERO 0.0f
    #define FLOAT_TO_DTYPE(x) (x)
    #define DTYPE_TO_FLOAT(x) (x)
#endif

#endif // DTYPE_HPP
