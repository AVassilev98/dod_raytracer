#pragma once
#include <immintrin.h>

// mutates o* inputs and returns their sum
static inline __attribute__((always_inline)) __m256 avxDot(const __m256 x, const __m256 y, const __m256 z, __m256 ox, __m256 oy, __m256 oz)
{
    __m256 mmx_px = _mm256_mul_ps(x, ox);
    __m256 mmx_py = _mm256_mul_ps(y, oy);
    __m256 mmx_pz = _mm256_mul_ps(z, oz);

    __m256 mmx_acc = _mm256_add_ps(mmx_px, mmx_py);
    __m256 mmx_res = _mm256_add_ps(mmx_acc, mmx_pz);
    return mmx_res;
}
