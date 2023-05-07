#include "triangle.h"
#include "avx_utils.h"
#include "config.h"
#include "glm/geometric.hpp"
#include "hitrecord.h"
#include "vector"
#include <array>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <cstring>
#include <limits>
#include "mesh.h"
#include "utils.h"

bool Triangle::intersect_impl(_Intersect &_in)
{
    std::span<TriangleLane> range(m_triangleLanes.data(), m_triangleLanes.size());
    return intersectInRange(_in, range, 0);
}

bool Triangle::intersectInRange(_Intersect &_in, const std::span<TriangleLane> &range, unsigned startIdx)
{
    static const __m256 zero = _mm256_setzero_ps();
    static const __m256 one = _mm256_set1_ps(1.0f);
    static const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    float llm[c_triangleLaneSz] __attribute__((aligned(32))) = {};
    unsigned triangleRemainder = m_numTriangles % c_triangleLaneSz;
    for (unsigned i = 0; i < triangleRemainder; i++)
    {
        memset(&llm[i], 0xFFFFFFFF, sizeof(float));
    }
    __m256 mmx_lastLaneMask = _mm256_load_ps(llm);

    float maximumDistance = _in.clippingDistance;
    unsigned minTriangleIndex = UINT32_MAX;

    avxVec3 rayOrigin = avxVec3Load(_in.rayOrigin);
    avxVec3 rayDir = avxVec3Load(_in.rayDir);
    glm::vec3 baryCoords = {0, 0, 0};

    for (unsigned i = 0; i < range.size(); i++)
    {
        __m256 mmxMaxDistance = _mm256_set1_ps(maximumDistance);
        __m256 validMask = _mm256_set1_ps(-0.0f);
        // mask off results for the last lane if it is not full
        const TriangleLane &triangleLane  = range[i];

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


        __m256 parallelMask = _mm256_cmp_ps(detAbs, zero, _CMP_GT_OS);
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
        float laneU[c_triangleLaneSz] __attribute__((aligned (32)));
        float laneV[c_triangleLaneSz] __attribute__((aligned (32)));
        _mm256_store_ps(laneT, mmxT);
        _mm256_store_ps(laneU, u);
        _mm256_store_ps(laneV, v);

        for (int j = 0; laneValid; laneValid >>= 1, j++)
        {
            if (!(laneValid & 1u))
            {
                continue;
            }

            if (laneT[j] < maximumDistance)
            {
                maximumDistance = laneT[j];
                minTriangleIndex = (startIdx + i) * c_triangleLaneSz + j;
                baryCoords = {1.0f - (laneU[j] + laneV[j]), laneU[j], laneV[j]};
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
        m_triangleLanes[laneIdx].Ax[triangleIdx],
        m_triangleLanes[laneIdx].Ay[triangleIdx],
        m_triangleLanes[laneIdx].Az[triangleIdx],
    };
    glm::vec3 B = {
        m_triangleLanes[laneIdx].Bx[triangleIdx],
        m_triangleLanes[laneIdx].By[triangleIdx],
        m_triangleLanes[laneIdx].Bz[triangleIdx],
    };
    glm::vec3 C = {
        m_triangleLanes[laneIdx].Cx[triangleIdx],
        m_triangleLanes[laneIdx].Cy[triangleIdx],
        m_triangleLanes[laneIdx].Cz[triangleIdx],
    };
    
    const Attributes &triangleAttributes = m_triangleAttributes[laneIdx];
    const Mesh::Attributes &mesh_attrs = Mesh::m_meshAttributes[triangleAttributes.meshAttrIdx[triangleIdx]];

    _in.record.t = maximumDistance;
    _in.record.color = mesh_attrs.color;
    _in.record.hitPoint = _in.rayOrigin + _in.rayDir * maximumDistance;
    _in.record.hitNormal = glm::mat3(triangleAttributes.AN[triangleIdx],
                                     triangleAttributes.BN[triangleIdx],
                                     triangleAttributes.CN[triangleIdx]) * baryCoords;

    return true;
}

bool Triangle::intersect_non_vectorized_impl(_Intersect &_in)
{
    float maximumDistance = _in.clippingDistance;
    glm::vec3 minNormal = {0, 0, 0};
    glm::vec3 minHitPoint = {0, 0, 0};
    glm::vec3 baryCoords = {0, 0, 0};
    unsigned minTriangleIndex = UINT32_MAX;

    for (int i = 0; i < m_triangleLanes.size(); i++)
    {
        for (int j = 0; j < c_triangleLaneSz; j++)
        {
            unsigned idx = (i * c_triangleLaneSz) + j;
            if (idx >= m_numTriangles)
            {
                break;
            }

            glm::vec3 A = glm::vec3(m_triangleLanes[i].Ax[j], m_triangleLanes[i].Ay[j], m_triangleLanes[i].Az[j]);
            glm::vec3 B = glm::vec3(m_triangleLanes[i].Bx[j], m_triangleLanes[i].By[j], m_triangleLanes[i].Bz[j]);
            glm::vec3 C = glm::vec3(m_triangleLanes[i].Cx[j], m_triangleLanes[i].Cy[j], m_triangleLanes[i].Cz[j]);

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
            minTriangleIndex = i * c_triangleLaneSz + j;
            maximumDistance = t;
            baryCoords = {1.0f - (u + v), u, v};
        }
    }

    if (minTriangleIndex == UINT32_MAX)
    {
        return false;
    }

    unsigned triangleLaneIdx = minTriangleIndex / c_triangleLaneSz;
    unsigned triangleIdx = minTriangleIndex % c_triangleLaneSz;

    const Attributes &triangleAttributes = m_triangleAttributes[triangleLaneIdx];
    const Mesh::Attributes &mesh_attrs = Mesh::m_meshAttributes[triangleAttributes.meshAttrIdx[triangleIdx]];
    HitRecord &record = _in.record;
    record.t = maximumDistance;
    record.hitNormal = glm::mat3(triangleAttributes.AN[triangleIdx],
                                 triangleAttributes.BN[triangleIdx],
                                 triangleAttributes.CN[triangleIdx]) * baryCoords;
    record.hitPoint = minHitPoint;
    record.color = mesh_attrs.color;

    return true;
}


unsigned Triangle::create(const _Create &createStruct)
{
    constexpr TriangleLane emptyTriangleLane = { 0 };
    constexpr Attributes emptyTriangleAttributes = { 0 };
    unsigned triangleIdx = (m_numTriangles) % c_triangleLaneSz;
    if (triangleIdx == 0)
    {
        m_triangleLanes.push_back(emptyTriangleLane);
        m_triangleAttributes.push_back(emptyTriangleAttributes);
    }
    auto &lane = m_triangleLanes.back();
    auto &attributes = m_triangleAttributes.back();
    lane.Ax[triangleIdx] = createStruct.A.x;
    lane.Ay[triangleIdx] = createStruct.A.y;
    lane.Az[triangleIdx] = createStruct.A.z;

    lane.Bx[triangleIdx] = createStruct.B.x;
    lane.By[triangleIdx] = createStruct.B.y;
    lane.Bz[triangleIdx] = createStruct.B.z;

    lane.Cx[triangleIdx] = createStruct.C.x;
    lane.Cy[triangleIdx] = createStruct.C.y;
    lane.Cz[triangleIdx] = createStruct.C.z;

    attributes.meshAttrIdx[triangleIdx] = Mesh::m_meshAttributes.size() - 1;
    attributes.AN[triangleIdx] = createStruct.AN;
    attributes.BN[triangleIdx] = createStruct.BN;
    attributes.CN[triangleIdx] = createStruct.CN;

    return ++m_numTriangles;
}

AxisAlignedBoundingBox Triangle::getBoundingBox(unsigned startIdx, unsigned numElements)
{
    constexpr float inf = std::numeric_limits<float>::infinity();

    AxisAlignedBoundingBox boundingBox
    {
        .minCorner = glm::vec3(inf),
        .maxCorner = glm::vec3(-inf),
    };

    for (int i = 0; i < numElements; i++)
    {
        boundingBox.Union(getTriangleBoundingBox(startIdx + i));
    }

    return boundingBox;
}

AxisAlignedBoundingBox Triangle::getTriangleBoundingBox(unsigned idx)
{
    unsigned laneIdx = idx / c_triangleLaneSz;
    unsigned triangleIdx = idx % c_triangleLaneSz;

    std::array<glm::vec3, 3> triangleVertices = {{
        {
            m_triangleLanes[laneIdx].Ax[triangleIdx],
            m_triangleLanes[laneIdx].Ay[triangleIdx],
            m_triangleLanes[laneIdx].Az[triangleIdx]
        },
        {
            m_triangleLanes[laneIdx].Bx[triangleIdx],
            m_triangleLanes[laneIdx].By[triangleIdx],
            m_triangleLanes[laneIdx].Bz[triangleIdx],
        },
        {
            m_triangleLanes[laneIdx].Cx[triangleIdx],
            m_triangleLanes[laneIdx].Cy[triangleIdx],
            m_triangleLanes[laneIdx].Cz[triangleIdx],
        }
    }};
    
    return {
        .minCorner = getElementWiseMinVec3(std::span(triangleVertices)),
        .maxCorner = getElementWiseMaxVec3(std::span(triangleVertices)),
    };
}

/*
This function reorders the triangleLanes so that the acceleration
structure does not need to jump around in the vector to grab the
lanes that belong in its node. Note that there can be duplication
(The same Lane in multiple nodes), so this will have a (slightly)
larger memory footprint on average than the original vector. This
tradeoff is worth it due to the cache coherency improvement.
*/
void Triangle::reorderLanesByIndices(const std::vector<unsigned> &indices)
{
    std::vector<Attributes> reorderedAttributes;
    std::vector<TriangleLane> reorderedLanes;
    reorderedAttributes.reserve(indices.size());
    reorderedLanes.reserve(indices.size());

    printf("%lu lanes duplicated\n", indices.size() - m_triangleLanes.size());

    for (unsigned idx : indices)
    {
        reorderedAttributes.push_back(m_triangleAttributes[idx]);
        reorderedLanes.push_back(m_triangleLanes[idx]);
    }

    m_triangleLanes = std::move(reorderedLanes);
    m_triangleAttributes = std::move(reorderedAttributes);
    printf("Done reordering triangles\n");
}