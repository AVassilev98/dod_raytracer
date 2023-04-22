#include "triangle.h"
#include "avx_utils.h"
#include "config.h"
#include "glm/geometric.hpp"
#include "hitrecord.h"
#include "vector"
#include <cstdint>
#include <immintrin.h>
#include <cstring>

constexpr unsigned c_triangleLaneSz = 8;
struct TriangleLane
{
    float Ax[c_triangleLaneSz];
    float Ay[c_triangleLaneSz];
    float Az[c_triangleLaneSz];
    float Bx[c_triangleLaneSz];
    float By[c_triangleLaneSz];
    float Bz[c_triangleLaneSz];
    float Cx[c_triangleLaneSz];
    float Cy[c_triangleLaneSz];
    float Cz[c_triangleLaneSz];
} __attribute__((aligned (32)));

static unsigned g_numTriangles = 0;
static std::vector<TriangleLane> g_triangleLanes;
static std::vector<Triangle::Attributes> g_triangleAttributes;


bool Triangle::intersect_impl(_Intersect &_in)
{
    static const __m256 epsilon = _mm256_set1_ps(Config::Epsilon);
    static const __m256 zero = _mm256_setzero_ps();
    static const __m256 one = _mm256_set1_ps(1.0f);
    static const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    float llm[c_triangleLaneSz] __attribute__((aligned(32))) = {};
    unsigned triangleRemainder = g_numTriangles % c_triangleLaneSz;
    for (unsigned i = 0; i < triangleRemainder; i++)
    {
        memset(&llm[i], 0xFFFFFFFF, sizeof(float));
    }
    __m256 mmx_lastLaneMask = _mm256_load_ps(llm);

    float maximumDistance = _in.clippingDistance;
    unsigned minTriangleIndex = UINT32_MAX;

    avxVec3 rayOrigin = avxVec3Load(_in.rayOrigin);
    avxVec3 rayDir = avxVec3Load(_in.rayDir);

    for (int i = 0; i < g_triangleLanes.size(); i++)
    {
        __m256 mmxMaxDistance = _mm256_set1_ps(maximumDistance);
        __m256 validMask = _mm256_set1_ps(-0.0f);
        // mask off results for the last lane if it is not full
        if (triangleRemainder && i == g_triangleLanes.size() - 1)
        {
            validMask = _mm256_and_ps(validMask, mmx_lastLaneMask);
        }

        const TriangleLane &triangleLane = g_triangleLanes[i];

        avxVec3 A = {
            _mm256_load_ps(triangleLane.Ax),
            _mm256_load_ps(triangleLane.Ay),
            _mm256_load_ps(triangleLane.Az),
        };
        avxVec3 B = {
            _mm256_load_ps(triangleLane.Bx),
            _mm256_load_ps(triangleLane.By),
            _mm256_load_ps(triangleLane.Bz),
        };
        avxVec3 C = {
            _mm256_load_ps(triangleLane.Cx),
            _mm256_load_ps(triangleLane.Cy),
            _mm256_load_ps(triangleLane.Cz),
        };

        avxVec3 AB = avxVec3Sub(B, A);
        avxVec3 AC = avxVec3Sub(C, A);
        avxVec3 pvec = avxCross(rayDir, AC);
        __m256 det = avxDot(pvec, AB);
        __m256 detAbs = _mm256_andnot_ps(sign_mask, det);


        __m256 parallelMask = _mm256_cmp_ps(detAbs, epsilon, _CMP_GT_OS);
        validMask = _mm256_and_ps(parallelMask, validMask);
        int laneValid = _mm256_movemask_ps(validMask);
        if (!laneValid)
        {
            continue;
        }

        __m256 inv_det = _mm256_div_ps(one, det);
        avxVec3 tvec = avxVec3Sub(rayOrigin, A);
        __m256 u = _mm256_mul_ps(avxDot(tvec, pvec), inv_det);

        __m256 uInsideTriangleMask = _mm256_and_ps(
                                        _mm256_cmp_ps(u, zero, _CMP_GT_OS),
                                        _mm256_cmp_ps(u, one, _CMP_LT_OS));
        validMask = _mm256_and_ps(uInsideTriangleMask, validMask);
        laneValid = _mm256_movemask_ps(validMask);
        if (!laneValid)
        {
            continue;
        }


        avxVec3 qvec = avxCross(tvec, AB);
        __m256 v = _mm256_mul_ps(avxDot(rayDir, qvec), inv_det);
        __m256 vInsideTriangleMask = _mm256_and_ps(
                                        _mm256_cmp_ps(v, zero, _CMP_GT_OS),
                                        _mm256_cmp_ps(_mm256_add_ps(u, v), one, _CMP_LT_OS));
        validMask = _mm256_and_ps(vInsideTriangleMask, validMask);
        laneValid = _mm256_movemask_ps(validMask);
        if (!laneValid)
        {
            continue;
        }

        __m256 mmxT = _mm256_mul_ps(avxDot(AC, qvec), inv_det);
        __m256 tInCorrectRange = _mm256_and_ps(
                                    _mm256_cmp_ps(mmxT, zero, _CMP_GT_OS),
                                    _mm256_cmp_ps(mmxT, mmxMaxDistance, _CMP_LT_OS));
        validMask = _mm256_and_ps(tInCorrectRange, validMask);
        laneValid = _mm256_movemask_ps(validMask);
        if (!laneValid)
        {
            continue;
        }

        float laneT[c_triangleLaneSz] __attribute__((aligned (32)));
        _mm256_store_ps(laneT, mmxT);

        for (int j = 0; laneValid; laneValid >>= 1, j++)
        {
            if (!(laneValid & 1u))
            {
                continue;
            }

            if (laneT[j] < maximumDistance)
            {
                maximumDistance = laneT[j];
                minTriangleIndex = i * c_triangleLaneSz + j;
            }
        }
    }

    if (minTriangleIndex == UINT32_MAX)
    {
        return false;
    }

    unsigned laneIdx = minTriangleIndex / c_triangleLaneSz;
    unsigned triangleIdx = minTriangleIndex % c_triangleLaneSz;

    glm::vec3 A = {
        g_triangleLanes[laneIdx].Ax[triangleIdx],
        g_triangleLanes[laneIdx].Ay[triangleIdx],
        g_triangleLanes[laneIdx].Az[triangleIdx],
    };
    glm::vec3 B = {
        g_triangleLanes[laneIdx].Bx[triangleIdx],
        g_triangleLanes[laneIdx].By[triangleIdx],
        g_triangleLanes[laneIdx].Bz[triangleIdx],
    };
    glm::vec3 C = {
        g_triangleLanes[laneIdx].Cx[triangleIdx],
        g_triangleLanes[laneIdx].Cy[triangleIdx],
        g_triangleLanes[laneIdx].Cz[triangleIdx],
    };
    glm::vec3 AB = B - A;
    glm::vec3 AC = C - A;

    _in.record.t = maximumDistance;
    _in.record.color = g_triangleAttributes[minTriangleIndex].color;
    _in.record.hitPoint = _in.rayOrigin + _in.rayDir * maximumDistance;
    _in.record.hitNormal = glm::normalize(glm::cross(AB, AC));
    return true;
}

bool Triangle::intersect_non_vectorized_impl(_Intersect &_in)
{
    float maximumDistance = _in.clippingDistance;
    glm::vec3 minNormal = {0, 0, 0};
    glm::vec3 minHitPoint = {0, 0, 0};
    unsigned minTriangleIndex = UINT32_MAX;

    for (int i = 0; i < g_triangleLanes.size(); i++)
    {
        for (int j = 0; j < c_triangleLaneSz; j++)
        {
            unsigned idx = (i * c_triangleLaneSz) + j;
            if (idx >= g_numTriangles)
            {
                break;
            }

            glm::vec3 A = glm::vec3(g_triangleLanes[i].Ax[j], g_triangleLanes[i].Ay[j], g_triangleLanes[i].Az[j]);
            glm::vec3 B = glm::vec3(g_triangleLanes[i].Bx[j], g_triangleLanes[i].By[j], g_triangleLanes[i].Bz[j]);
            glm::vec3 C = glm::vec3(g_triangleLanes[i].Cx[j], g_triangleLanes[i].Cy[j], g_triangleLanes[i].Cz[j]);

            glm::vec3 AB = B - A;
            glm::vec3 AC = C - A;
            glm::vec3 pvec = glm::cross(_in.rayDir, AC);
            float det = glm::dot(pvec, AB);

            if (fabs(det) < Config::Epsilon)
            {
                continue;
            }
            float inv_det = 1.0f / det;
            glm::vec3 tvec = _in.rayOrigin - A;

            float u = glm::dot(tvec, pvec) * inv_det;
            if (u < 0.0f || u > 1.0f)
            {
                continue;
            }

            glm::vec3 qvec = glm::cross(tvec, AB);
            float v = glm::dot(_in.rayDir, qvec) * inv_det;
            if (v < 0.0f || v + u > 1.0f)
            {
                continue;
            }

            float t = glm::dot(AC, qvec) * inv_det;

            if (t < 0.0f || t > maximumDistance)
            {
                continue;
            }

            minHitPoint = _in.rayOrigin + _in.rayDir * t;
            minNormal = glm::normalize(glm::cross(AB, AC));
            minTriangleIndex = i * c_triangleLaneSz + j;
            maximumDistance = t;
        }
    }

    if (minTriangleIndex == UINT32_MAX)
    {
        return false;
    }

    unsigned triangleLaneIdx = minTriangleIndex / c_triangleLaneSz;
    unsigned triangleIdx = minTriangleIndex % c_triangleLaneSz;

    HitRecord &record = _in.record;
    record.t = maximumDistance;
    record.hitNormal = minNormal;
    record.hitPoint = minHitPoint;
    record.color = g_triangleAttributes[minTriangleIndex].color;

    return true;
}


unsigned Triangle::create(const _Create &createStruct)
{
    constexpr TriangleLane emptyTriangleLane = {};
    unsigned triangleIdx = (g_numTriangles) % c_triangleLaneSz;
    if (triangleIdx == 0)
    {
        g_triangleLanes.push_back(emptyTriangleLane);
    }
    auto &lane = g_triangleLanes.back();
    lane.Ax[triangleIdx] = createStruct.A.x;
    lane.Ay[triangleIdx] = createStruct.A.y;
    lane.Az[triangleIdx] = createStruct.A.z;

    lane.Bx[triangleIdx] = createStruct.B.x;
    lane.By[triangleIdx] = createStruct.B.y;
    lane.Bz[triangleIdx] = createStruct.B.z;

    lane.Cx[triangleIdx] = createStruct.C.x;
    lane.Cy[triangleIdx] = createStruct.C.y;
    lane.Cz[triangleIdx] = createStruct.C.z;

    g_triangleAttributes.emplace_back(createStruct.attributes);
    return ++g_numTriangles;
}
