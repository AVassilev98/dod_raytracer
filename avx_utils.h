#pragma once
#include <immintrin.h>
#include "glm/glm.hpp"

struct avxVec3 
{
    __m256 x;
    __m256 y;
    __m256 z;
};

// mutates o* inputs and returns their sum
static inline __attribute__((always_inline)) __m256 avxDot(const avxVec3 &v1, const avxVec3 &v2)
{
    __m256 mmx_px = _mm256_mul_ps(v1.x, v2.x);
    __m256 mmx_py = _mm256_mul_ps(v1.y, v2.y);
    __m256 mmx_pz = _mm256_mul_ps(v1.z, v2.z);

    __m256 mmx_acc = _mm256_add_ps(mmx_px, mmx_py);
    __m256 mmx_res = _mm256_add_ps(mmx_acc, mmx_pz);
    return mmx_res;
}

static inline __attribute__((always_inline)) avxVec3 avxCross(const avxVec3 &v1, const avxVec3 &v2)
{
    avxVec3 ret;

    ret.x = _mm256_sub_ps(_mm256_mul_ps(v1.y, v2.z), _mm256_mul_ps(v1.z, v2.y));
    ret.y = _mm256_sub_ps(_mm256_mul_ps(v1.z, v2.x), _mm256_mul_ps(v1.x, v2.z));
    ret.z = _mm256_sub_ps(_mm256_mul_ps(v1.x, v2.y), _mm256_mul_ps(v1.y, v2.x));

    return ret;
}

static inline __attribute__((always_inline)) avxVec3 avxVec3Sub(const avxVec3 &v1, const avxVec3 &v2)
{
    return avxVec3 {
        _mm256_sub_ps(v1.x, v2.x),
        _mm256_sub_ps(v1.y, v2.y),
        _mm256_sub_ps(v1.z, v2.z),
    };
}

static inline __attribute__((always_inline)) avxVec3 avxVec3Add(const avxVec3 &v1, const avxVec3 &v2)
{
    return avxVec3 {
        _mm256_add_ps(v1.x, v2.x),
        _mm256_add_ps(v1.y, v2.y),
        _mm256_add_ps(v1.z, v2.z),
    };
}

static inline __attribute__((always_inline)) avxVec3 avxVec3Load(const glm::vec3 &vec)
{
    return avxVec3 {
        _mm256_set1_ps(vec.x),
        _mm256_set1_ps(vec.y),
        _mm256_set1_ps(vec.z),
    };
}
