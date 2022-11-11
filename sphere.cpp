#include "sphere.h"
#include "common_defs.h"
#include <cstdint>
#include <immintrin.h>
#include <cstring>
#include <vector>
#include "glm/geometric.hpp"
#include "immintrin.h"

constexpr unsigned c_sphereLaneSz = 8;
struct SphereLane
{
    float x[c_sphereLaneSz];
    float y[c_sphereLaneSz];
    float z[c_sphereLaneSz];
    float radiusSq[c_sphereLaneSz];
} __attribute__((aligned (32)));

static unsigned g_numSpheres = 0;
static std::vector<SphereLane> g_sphereLanes;
static std::vector<Sphere::Attributes> g_sphereAttributes;

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

HitRecord *Sphere::intersect(const glm::vec3 &rayDir, const glm::vec3 &rayOrigin, HitRecord &ret)
{
    ret.t = g_frustrumMax * g_frustrumMax;
    unsigned closestSphereIdx = UINT32_MAX;
    static const __m256 zeros = _mm256_setzero_ps();
    float llm[c_sphereLaneSz] __attribute__((aligned(32))) = {};
    float mask_val = 0;
    memset(&mask_val, 0xFFFFFFFF, sizeof(float));
    unsigned sphereRemainder = g_numSpheres % c_sphereLaneSz;
    for (unsigned i = 0; i < sphereRemainder; i++)
    {
        memset(&llm[i], 0xFFFFFFFF, sizeof(float));
    }
    __m256 mmx_lastLaneMask = _mm256_load_ps(llm);

    for (unsigned i = 0; i < g_sphereLanes.size(); i++)
    {
        const auto &sphereLane = g_sphereLanes[i];
        
        // indicates entry is valid
        __m256 mmx_validMask = _mm256_set1_ps(mask_val);

        // fill avx registers with our sphere lane
        __m256 mmx_sx = _mm256_load_ps(sphereLane.x);
        __m256 mmx_sy = _mm256_load_ps(sphereLane.y);
        __m256 mmx_sz = _mm256_load_ps(sphereLane.z);

        // broadcast rayOrigin vec3 in to avx registers
        __m256 mmx_rox = _mm256_set1_ps(rayOrigin.x);
        __m256 mmx_roy = _mm256_set1_ps(rayOrigin.y);
        __m256 mmx_roz = _mm256_set1_ps(rayOrigin.z);

        // L = pos - rayOrigin
        __m256 mmx_lx = _mm256_sub_ps(mmx_sx, mmx_rox);
        __m256 mmx_ly = _mm256_sub_ps(mmx_sy, mmx_roy);
        __m256 mmx_lz = _mm256_sub_ps(mmx_sz, mmx_roz);

        __m256 mmx_distSq = avxDot(mmx_lx, mmx_ly, mmx_lz, mmx_lx, mmx_ly, mmx_lz);
        __m256 mmx_radSq = _mm256_load_ps(sphereLane.radiusSq);

        // Check if all ray is in all spheres in lane
        __m256 mmx_rayInSphere = _mm256_cmp_ps(mmx_distSq, mmx_radSq, _CMP_GT_OS);
        int mask = _mm256_movemask_ps(mmx_rayInSphere);
        if (mask == 0)
        {
            continue;
        }
        mmx_validMask = _mm256_and_ps(mmx_rayInSphere, mmx_validMask);
        
        // broadcast rayDir vec3 in to avx registers
        __m256 mmx_rdx = _mm256_set1_ps(rayDir.x);
        __m256 mmx_rdy = _mm256_set1_ps(rayDir.y);
        __m256 mmx_rdz = _mm256_set1_ps(rayDir.z);

        __m256 mmx_tca = avxDot(mmx_lx, mmx_ly, mmx_lz, mmx_rdx, mmx_rdy, mmx_rdz);
        __m256 mmx_tcaSq = _mm256_mul_ps(mmx_tca, mmx_tca);
        __m256 mmx_d2 = _mm256_sub_ps(mmx_distSq, mmx_tcaSq);

        // mask off results for the last lane if it is not full
        if (sphereRemainder && i == g_sphereLanes.size() - 1)
        {
            mmx_distSq = _mm256_and_ps(mmx_distSq, mmx_lastLaneMask);
        }

        // check if closest point is outside all spheres' radii
        __m256 mmx_rayMissSphere = _mm256_cmp_ps(mmx_d2, mmx_radSq, _CMP_LT_OS);
        mask = _mm256_movemask_ps(mmx_rayMissSphere);
        if (mask == 0)
        {
            continue;
        }
        
        // mask off the rays that did not hit the sphere
        mmx_validMask = _mm256_and_ps(mmx_validMask, mmx_rayMissSphere);

        __m256 mmx_thcSq = _mm256_sub_ps(mmx_radSq, mmx_d2);
        __m256 mmx_thc = _mm256_sqrt_ps(mmx_thcSq);
        __m256 mmx_t0 = _mm256_sub_ps(mmx_tca, mmx_thc);
        __m256 mmx_t1 = _mm256_add_ps(mmx_tca, mmx_thc);


        // Check if the ray is going backwards
        __m256 mmx_t0lz = _mm256_cmp_ps(mmx_t0, zeros, _CMP_GE_OS);
        __m256 mmx_t1lz = _mm256_cmp_ps(mmx_t1, zeros, _CMP_GE_OS);
        __m256 mmx_tCombinedMask = _mm256_and_ps(mmx_t0lz, mmx_t1lz);
        int t0Mask = _mm256_movemask_ps(mmx_t0lz);
        int t1Mask = _mm256_movemask_ps(mmx_t1lz);

        // sphere is behind the ray
        if (t0Mask == 0 && t1Mask == 0)
        {
            continue;
        }
        mmx_validMask = _mm256_and_ps(mmx_validMask, mmx_tCombinedMask);
        int validMask = _mm256_movemask_ps(mmx_validMask);

        __m256 mmx_retDistSq = _mm256_set1_ps(ret.t);
        __m256 hit = _mm256_cmp_ps(mmx_distSq, mmx_retDistSq, _CMP_LT_OS);
        int hitMask = _mm256_movemask_ps(hit);
        if (hitMask == 0)
        {
            continue;
        }

        __m256 mmx_tmin = _mm256_min_ps(mmx_t0, mmx_t1);
        float distSq[c_sphereLaneSz] __attribute__((aligned (32)));
        float tmin[c_sphereLaneSz] __attribute__((aligned (32)));

        _mm256_store_ps(distSq, mmx_distSq);
        _mm256_store_ps(tmin, mmx_tmin);

        unsigned minDistIdx = 0;
        float minDist = ret.t;

        for (unsigned j = 0; j < c_sphereLaneSz; j++)
        {
            if ((validMask >> j) & 1 && tmin[j] < minDist)
            {
                minDist = tmin[j];
                minDistIdx = j;
            }
        }

        if (minDist < ret.t)
        {
            ret.t = tmin[minDistIdx];
            closestSphereIdx = i * c_sphereLaneSz + minDistIdx;
        }
    }

    // no intersection
    if (closestSphereIdx == UINT32_MAX)
    {
        return nullptr;
    }

    unsigned laneIndex = closestSphereIdx / c_sphereLaneSz;
    unsigned sphereIdx = closestSphereIdx % c_sphereLaneSz;

    ret.color = g_sphereAttributes[closestSphereIdx].color;
    ret.spherePos = glm::vec3(g_sphereLanes[laneIndex].x[sphereIdx], g_sphereLanes[laneIndex].y[sphereIdx], g_sphereLanes[laneIndex].z[sphereIdx]);
    ret.hitPoint = rayOrigin + rayDir * ret.t;
    ret.hitNormal = glm::normalize(ret.hitPoint - ret.spherePos);

    return &ret;
}

unsigned Sphere::create(const Sphere::_Create &createStruct)
{
    static const SphereLane emptySphereLane = {};
    unsigned sphereIdx = (g_numSpheres) % c_sphereLaneSz;
    if (sphereIdx == 0)
    {
        g_sphereLanes.push_back(emptySphereLane);
    }
    auto &lane = g_sphereLanes.back();
    lane.x[sphereIdx] = createStruct.position.x;
    lane.y[sphereIdx] = createStruct.position.y;
    lane.z[sphereIdx] = createStruct.position.z;
    lane.radiusSq[sphereIdx] = createStruct.radius * createStruct.radius;

    g_sphereAttributes.emplace_back(createStruct.attributes);
    return ++g_numSpheres;
}