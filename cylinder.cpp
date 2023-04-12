#include "cylinder.h"
#include "glm/geometric.hpp"
#include "hitrecord.h"
#include <limits>
#include <vector>

float minNonNegative(float a, float b)
{
    if (a < 0 && b < 0)
    {
        return std::numeric_limits<float>::infinity();
    }
    else if (a < 0)
    {
        return b;
    }
    else if (b < 0) 
    {
        return a;
    } 
    else 
    {
        return fminf(a, b);
    }
}


namespace {
    std::vector<Cylinder> g_cylinders;
    std::vector<Cylinder::Attributes> g_cylinderAttributes;
};


bool Cylinder::intersect_impl(_Intersect &_in)
{
    return Cylinder::intersect_non_vectorized(_in);
}

bool checkDiskIntersect
(
    float &t,
    const glm::vec3 &rayOrigin, 
    const glm::vec3 &rayDir, 
    const glm::vec3 &planeP, 
    const glm::vec3 &planeN, 
    float rSq, 
    float minT
)
{
    float denom = glm::dot(rayDir, planeN);
    if (fabs(denom) < 0.0001f)
    {
        return false;
    }
    glm::vec3 vecToPlane = planeP - rayOrigin;
    float tnum = glm::dot(vecToPlane, planeN);
    float tTotal = tnum / denom;

    if (tTotal < 0.0001f || tTotal > minT)
    {
        return false;
    }
    glm::vec3 hitPoint = rayOrigin + rayDir * tTotal;
    glm::vec3 planeVec = hitPoint - planeP;
    if (glm::dot(planeVec, planeVec) >= rSq)
    {
        return false;
    }


    t = tTotal;
    return true;
}

bool Cylinder::intersect_cylinder_body(_Intersect &_in, HitRecord &hr) const
{
        glm::vec3 deltaP = _in.rayOrigin - m_base;
        glm::vec3 vRem = _in.rayDir - glm::dot(_in.rayDir, m_axis) * m_axis;
        glm::vec3 deltaPRem = deltaP - glm::dot(deltaP, m_axis) * m_axis;

        float a = glm::dot(vRem, vRem);
        float b = 2.0f * glm::dot(vRem, deltaPRem);
        float c = glm::dot(deltaPRem, deltaPRem) - m_radiusSq;

        float discriminant = (b * b) - (4 * a * c);
        if (discriminant < 0.0001f)
        {
            return false;
        }

        float tSub = (-b - sqrt(discriminant)) / (2 * a);
        float tAdd = (-b + sqrt(discriminant)) / (2 * a);
        float minRayDirFactor = minNonNegative(tSub, tAdd);

        if (minRayDirFactor == std::numeric_limits<float>::infinity())
        {
            return false;
        }

        glm::vec3 cmpA = _in.rayOrigin + _in.rayDir * minRayDirFactor - m_base;
        float axisVectorFactor = glm::dot(cmpA, m_axis);

        if (axisVectorFactor < 0.f || axisVectorFactor > m_height)
        {
            return false;
        }

        // glm::vec3 hitPoint = _in.rayDir * minRayDirFactor + _in.rayOrigin;
        // float minX = glm::dot(hitPoint - cylinder.m_base, cylinder.m_axis);
        // glm::vec3 normal = glm::normalize(hitPoint - cylinder.m_base - cylinder.m_axis * minX);

        hr.t = minRayDirFactor;
        hr.hitPoint = _in.rayDir * hr.t + _in.rayOrigin;
        float minX = glm::dot(hr.hitPoint - m_base, m_axis);
        hr.hitNormal = glm::normalize(hr.hitPoint - m_base - m_axis * minX);
        return true;
}

bool Cylinder::intersect_cylinder_disc(_Intersect &_in, float offset, HitRecord &hr) const
{
    float minT = _in.clippingDistance;

    glm::vec3 position = m_base + m_axis * offset;
    glm::vec3 normal = m_axis;

    float denom = glm::dot(_in.rayDir, normal);
    if (fabs(denom) < 0.0001f)
    {
        return false;
    }
    glm::vec3 vecToPlane = position - _in.rayOrigin;
    float tnum = glm::dot(vecToPlane, normal);
    float t = tnum / denom;

    if (t < 0.0001f || t > minT)
    {
        return false;
    }

    glm::vec3 hitPoint = _in.rayOrigin + _in.rayDir * t;
    glm::vec3 vecOnPlane = hitPoint - position;
    if (glm::dot(vecOnPlane, vecOnPlane) > m_radiusSq)
    {
        return false;
    }

    hr.t = t;
    hr.hitPoint = hitPoint;
    hr.hitNormal = glm::dot(_in.rayDir, m_axis) > 0.0f ? -m_axis : m_axis;
    return true;
}


bool Cylinder::intersect_non_vectorized(_Intersect &_in)
{
    // Equation of a cylinder: (q - pa - (va,q - pa)va)^2 - r2 = 0
    // a = (v - (v,va)va)^2
    // b = 2(v - (v,va)va, dp-(dp,va)va)
    // c = (dp -(dp, va)va)^2 - r^2
    // where dp = p - pa

    unsigned minCylinderIdx = UINT32_MAX;
    float tMin = _in.clippingDistance;
    glm::vec3 minNormal;
    glm::vec3 minHitPoint;

    for (int i = 0; i < g_cylinders.size(); i++)
    {
        const Cylinder &cylinder = g_cylinders[i];

        HitRecord hrBody = {};
        HitRecord hrDiscA = {};
        HitRecord hrDiscB = {};
        if (cylinder.intersect_cylinder_body(_in, hrBody) && hrBody.t < tMin)
        {
            tMin = hrBody.t;
            minCylinderIdx = i;
            _in.record = hrBody;
        }
        if (cylinder.intersect_cylinder_disc(_in, 0.0f, hrDiscA) && hrDiscA.t < tMin)
        {
            tMin = hrDiscA.t;
            minCylinderIdx = i;
            _in.record = hrDiscA;
        }
        if (cylinder.intersect_cylinder_disc(_in, cylinder.m_height, hrDiscB) && hrDiscB.t < tMin)
        {
            tMin = hrDiscB.t;
            minCylinderIdx = i;
            _in.record = hrDiscB;
        }
    }

    if (minCylinderIdx == UINT32_MAX)
    {
        return false;
    }

    HitRecord &record = _in.record;
    const Cylinder &cylinder = g_cylinders[minCylinderIdx];
    const Attributes &attr = g_cylinderAttributes[minCylinderIdx];

    // record.color = attr.color;
    // record.hitPoint = minHitPoint;
    // record.hitNormal = minNormal;
    // record.t = tMin;
    return true;

}
unsigned Cylinder::create(const _Create &createStruct)
{
    g_cylinders.push_back(Cylinder(createStruct));
    g_cylinderAttributes.push_back(createStruct.attributes);
    return g_cylinders.size() - 1;
}

Cylinder *Cylinder::getCylinder(unsigned index)
{
    assert(index < g_cylinders.size() && "Cylinder::getCylinder - out of bounds!");
    return &g_cylinders[index];
}

Cylinder::Cylinder(const _Create &createStruct)
    : m_axis(glm::normalize(createStruct.axis))
    , m_base(createStruct.basePosition)
    , m_radiusSq(createStruct.radius * createStruct.radius)
    , m_height(createStruct.height)
{
}

