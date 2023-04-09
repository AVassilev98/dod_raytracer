#include "plane.h"
#include <vector>

// plane.cpp implementation details
namespace {
    unsigned g_numPlanes;
    std::vector<Plane> g_planes;
    std::vector<Plane::Attributes> g_planeAttributes;
};

    bool Plane::intersect_impl(_Intersect &_in)
    {
        return intersect_non_vectorized_impl(_in);
    }

    bool Plane::intersect_non_vectorized_impl(_Intersect &_in)
    {
        // (p - p0) . n = 0
        // ((Ro + Rd * t) - p0) . n = 0
        // t * (Rd . n) + (Ro - p0) . n = 0
        // t = ((p0 - Ro) . n / Rd . n

        float minT = _in.clippingDistance;
        unsigned minPlaneId = UINT32_MAX;
        for (int i = 0; i < g_numPlanes; i++)
        {
            const Plane &plane = g_planes[i];

            float denom = glm::dot(_in.rayDir, plane.m_normal);
            if (fabs(denom) < 0.0001f)
            {
                continue;
            }
            glm::vec3 vecToPlane = plane.m_position - _in.rayOrigin;
            float t = glm::dot(vecToPlane, plane.m_normal) / denom;

            if (t < 0.0001f || t > minT)
            {
                continue;
            }

            minT = t;
            minPlaneId = i;
            if (_in.returnOnAny)
            {
                break;
            }
        }

        if (minPlaneId == UINT32_MAX)
        {
            return false;
        }

        HitRecord &ret = _in.record;
        Plane &minPlane = g_planes[minPlaneId];
        Plane::Attributes &minPlaneAttrs = g_planeAttributes[minPlaneId];

        ret.color = g_planeAttributes[minPlaneId].color;
        ret.hitNormal = glm::dot(_in.rayDir, g_planes[minPlaneId].m_normal) < 0.0f ? minPlane.m_normal : -minPlane.m_normal;
        ret.hitPoint = _in.rayOrigin + minT * _in.rayDir;
        ret.t = minT;

        return true;
    }

    unsigned Plane::create(const _Create &createStruct)
    {
        g_planes.push_back(Plane(createStruct));
        g_planeAttributes.push_back(createStruct.attributes);
        return ++g_numPlanes;
    }

    Plane::Plane(const _Create &createStruct)
        : m_normal(createStruct.normal)
        , m_position(createStruct.position)
    {}
