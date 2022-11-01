#include "sphere.h"
#include "common_defs.h"
#include <cstdint>
#include <vector>

static std::vector<Sphere> g_spheres;
static std::vector<Sphere::Attributes> g_sphereAttributes;

HitRecord *Sphere::intersect(const Sphere *spheres, size_t count, glm::vec3 &rayDir, glm::vec3 &rayOrigin, HitRecord &ret)
{
    ret.distSq = g_frustrumMax * g_frustrumMax;
    float distSq = g_frustrumMax * g_frustrumMax;
    unsigned closestSphereIdx = UINT32_MAX;
    
    // variables used to compute intersection
    glm::vec3 L;
    float tca;
    float d2;
    float thc;
    float t0;
    float t1;

    for (unsigned i = 0; i < count; i++)
    {
        const Sphere& sphere = spheres[i];
        L = sphere.m_position - rayOrigin;
        distSq = glm::dot(L, L);

        // Do not accept hits from within a sphere
        if (distSq <= sphere.m_radiusSq)
        {
            goto Miss;
        }

        tca = glm::dot(L, rayDir);
        d2 = distSq - tca * tca;

        // closest point is outside the radius
        if (d2 > sphere.m_radiusSq)
        {
            goto Miss;
        }

        thc = sqrt(sphere.m_radiusSq - d2); 
        t0 = tca - thc; 
        t1 = tca + thc;

        // sphere is behind the ray
        if (t0 < 0 && t1 < 0)
        {
            goto Miss;
        }

        Hit:
        if (distSq < ret.distSq)
        {
            ret.distSq = distSq;
            ret.t = std::min(t0, t1);
            closestSphereIdx = i;
        }
        Miss:
        continue;
    }

    // no intersection
    if (closestSphereIdx == UINT32_MAX)
    {
        return nullptr;
    }

    ret.color = g_sphereAttributes[closestSphereIdx].color;

    return &ret;
}

Sphere::Sphere(const Sphere::_Create &createStruct)
{
    m_position = createStruct.position;
    m_radiusSq = createStruct.radius * createStruct.radius;
}

unsigned Sphere::create(const Sphere::_Create &createStruct)
{
    g_spheres.push_back(createStruct);
    g_sphereAttributes.emplace_back(createStruct.attributes);
    return g_spheres.size() - 1;
}

Sphere *Sphere::getSphere(unsigned index)
{
    return &g_spheres[index];
}

const Sphere *Sphere::getAllSpheres(unsigned &count)
{
    count = g_spheres.size();
    return g_spheres.data();
}
