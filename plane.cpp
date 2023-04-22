#include "plane.h"
#include "avx_utils.h"
#include "config.h"
#include <immintrin.h>
#include <vector>
#include "immintrin.h"
#include <cstring>

// plane.cpp implementation details
namespace {
    constexpr unsigned c_planeLaneSz = 8;
    struct PlaneLane
    {
        float px[c_planeLaneSz];
        float py[c_planeLaneSz];
        float pz[c_planeLaneSz];
        float nx[c_planeLaneSz];
        float ny[c_planeLaneSz];
        float nz[c_planeLaneSz];
    } __attribute__((aligned (32)));

    unsigned g_numPlanes;
    std::vector<PlaneLane> g_planeLanes;
    std::vector<Plane::Attributes> g_planeAttributes;
};

    bool Plane::intersect_impl(_Intersect &_in)
    {
        _in.record.t = _in.clippingDistance;
        unsigned closestPlaneIdx = UINT32_MAX;
        static const __m256 zeros = _mm256_set1_ps(Config::Epsilon);
        static const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        float llm[c_planeLaneSz] __attribute__((aligned(32))) = {};
        unsigned planeRemainder = g_numPlanes % c_planeLaneSz;
        memset(llm, 0xFFFFFFFF, sizeof(float) * c_planeLaneSz);
        __m256 mmx_lastLaneMask = _mm256_load_ps(llm);

        float minT = _in.clippingDistance;

        for (unsigned i = 0; i < g_planeLanes.size(); i++)
        {
            const auto &planeLane = g_planeLanes[i];

            // indicates entry is valid
            __m256 mmx_validMask = _mm256_castsi256_ps( _mm256_set1_epi32(-1) );
            // mask off results for the last lane if it is not full
            if (planeRemainder && i == g_planeLanes.size() - 1)
            {
                mmx_validMask = _mm256_and_ps(mmx_validMask, mmx_lastLaneMask);
            }

            // fill avx registers with our plane lane
            __m256 mmx_px = _mm256_load_ps(planeLane.px);
            __m256 mmx_py = _mm256_load_ps(planeLane.py);
            __m256 mmx_pz = _mm256_load_ps(planeLane.pz);
            __m256 mmx_nx = _mm256_load_ps(planeLane.nx);
            __m256 mmx_ny = _mm256_load_ps(planeLane.ny);
            __m256 mmx_nz = _mm256_load_ps(planeLane.nz);

            __m256 mmx_rox = _mm256_set1_ps(_in.rayOrigin.x);
            __m256 mmx_roy = _mm256_set1_ps(_in.rayOrigin.y);
            __m256 mmx_roz = _mm256_set1_ps(_in.rayOrigin.z);
            __m256 mmx_rdx = _mm256_set1_ps(_in.rayDir.x);
            __m256 mmx_rdy = _mm256_set1_ps(_in.rayDir.y);
            __m256 mmx_rdz = _mm256_set1_ps(_in.rayDir.z);

            __m256 mmx_denom = avxDot(mmx_rdx, mmx_rdy, mmx_rdz, mmx_nx, mmx_ny, mmx_nz);
            __m256 mmx_abs_denom = _mm256_andnot_ps(sign_mask, mmx_denom);

            __m256 mmx_rayParallel = _mm256_cmp_ps(mmx_abs_denom, zeros, _CMP_GT_OS);
            mmx_validMask = _mm256_and_ps(mmx_rayParallel, mmx_validMask);
            int mask = _mm256_movemask_ps(mmx_validMask);
            if (mask == 0)
            {
                continue;
            }

            __m256 mmx_vpx = _mm256_sub_ps(mmx_px, mmx_rox);
            __m256 mmx_vpy = _mm256_sub_ps(mmx_py, mmx_roy);
            __m256 mmx_vpz = _mm256_sub_ps(mmx_pz, mmx_roz);

            __m256 mmx_num = avxDot(mmx_vpx, mmx_vpy, mmx_vpz, mmx_nx, mmx_ny, mmx_nz);
            __m256 mmx_t = _mm256_div_ps(mmx_num, mmx_denom);

            __m256 mmx_hitBehind = _mm256_cmp_ps(mmx_t, zeros, _CMP_GT_OS);
            mmx_validMask = _mm256_and_ps(mmx_hitBehind, mmx_validMask);
            mask = _mm256_movemask_ps(mmx_validMask);
            if (mask == 0)
            {
                continue;
            }
            mask = _mm256_movemask_ps(mmx_validMask);

            __m256 mmx_minT = _mm256_set1_ps(_in.record.t);
            __m256 mmx_pastClip = _mm256_cmp_ps(mmx_t, mmx_minT, _CMP_LT_OS);
            mmx_validMask = _mm256_and_ps(mmx_pastClip, mmx_validMask);
            if (mask == 0)
            {
                continue;
            }

            float tMin[c_planeLaneSz] __attribute__((aligned (32)));
            _mm256_store_ps(tMin, mmx_t);

            for (int j = 0; j < c_planeLaneSz; j++)
            {
                if (((mask >> j) & 1) && tMin[j] < minT)
                {
                    minT = tMin[j];
                    closestPlaneIdx = i * c_planeLaneSz + j;
                }
            }


        }

        // no intersection
        if (closestPlaneIdx == UINT32_MAX)
        {
            return false;
        }


        unsigned laneIndex = closestPlaneIdx / c_planeLaneSz;
        unsigned planeIdx = closestPlaneIdx % c_planeLaneSz;


        HitRecord &ret = _in.record;
        PlaneLane &minPlaneLane = g_planeLanes[laneIndex];
        Plane::Attributes &minPlaneAttrs = g_planeAttributes[closestPlaneIdx];
        glm::vec3 planeNormal = glm::vec3(minPlaneLane.nx[planeIdx], minPlaneLane.ny[planeIdx], minPlaneLane.nz[planeIdx]);

        ret.color = g_planeAttributes[closestPlaneIdx].color;
        ret.hitNormal = planeNormal;//glm::dot(_in.rayDir, planeNormal) < 0.0f ? planeNormal : -planeNormal;
        ret.hitPoint = _in.rayOrigin + minT * _in.rayDir;
        ret.t = minT;

        return true;
    }

    bool Plane::intersect_non_vectorized_impl(_Intersect &_in)
    {
        // (p - p0) . n = 0
        // ((Ro + Rd * t) - p0) . n = 0
        // t * (Rd . n) + (Ro - p0) . n = 0
        // t = ((p0 - Ro) . n / Rd . n

        float minT = _in.clippingDistance;
        unsigned minPlaneId = UINT32_MAX;
        for (int i = 0; i < g_planeLanes.size(); i++)
        {
            const PlaneLane &planeLane = g_planeLanes[i];

            for (int j = 0; j < c_planeLaneSz; j++)
            {
                glm::vec3 position = glm::vec3(planeLane.px[j], planeLane.py[j], planeLane.pz[j]);
                glm::vec3 normal = glm::vec3(planeLane.nx[j], planeLane.ny[j], planeLane.nz[j]);

                float denom = glm::dot(_in.rayDir, normal);
                if (fabs(denom) < Config::Epsilon)
                {
                    continue;
                }
                glm::vec3 vecToPlane = position - _in.rayOrigin;
                float tnum = glm::dot(vecToPlane, normal);
                float t = tnum / denom;

                if (t < Config::Epsilon || t > minT)
                {
                    continue;
                }

                minT = t;
                minPlaneId = i * c_planeLaneSz + j;
                if (_in.returnOnAny)
                {
                    break;
                }
            }
        }

        if (minPlaneId == UINT32_MAX)
        {
            return false;
        }

        unsigned laneIndex = minPlaneId / c_planeLaneSz;
        unsigned planeIdx = minPlaneId % c_planeLaneSz;


        HitRecord &ret = _in.record;
        PlaneLane &minPlaneLane = g_planeLanes[laneIndex];
        Plane::Attributes &minPlaneAttrs = g_planeAttributes[minPlaneId];
        glm::vec3 planeNormal = glm::vec3(minPlaneLane.nx[planeIdx], minPlaneLane.ny[planeIdx], minPlaneLane.nz[planeIdx]);

        ret.color = g_planeAttributes[minPlaneId].color;
        ret.hitNormal = planeNormal;//glm::dot(_in.rayDir, planeNormal) < 0.0f ? planeNormal : -planeNormal;
        ret.hitPoint = _in.rayOrigin + minT * _in.rayDir;
        ret.t = minT;

        return true;
    }

    unsigned Plane::create(const _Create &createStruct)
    {
        static const PlaneLane emptyPlaneLane = {};
        unsigned planeIdx = (g_numPlanes) % c_planeLaneSz;
        if (planeIdx == 0)
        {
            g_planeLanes.push_back(emptyPlaneLane);
        }
        auto &lane = g_planeLanes.back();
        lane.px[planeIdx] = createStruct.position.x;
        lane.py[planeIdx] = createStruct.position.y;
        lane.pz[planeIdx] = createStruct.position.z;
        lane.nx[planeIdx] = createStruct.normal.x;
        lane.ny[planeIdx] = createStruct.normal.y;
        lane.nz[planeIdx] = createStruct.normal.z;

        g_planeAttributes.emplace_back(createStruct.attributes);
        return ++g_numPlanes;
    }

    Plane::Plane(const _Create &createStruct)
        : m_normal(createStruct.normal)
        , m_position(createStruct.position)
    {}
