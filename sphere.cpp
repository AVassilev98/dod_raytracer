#include "sphere.h"
#include "common_defs.h"
#include <cstdint>
#include <vector>

constexpr unsigned c_sphereLaneSz = 8;
struct SphereLane
{
    float x[c_sphereLaneSz];
    float y[c_sphereLaneSz];
    float z[c_sphereLaneSz];
    float radiusSq[c_sphereLaneSz];
} __attribute__((packed));

static unsigned g_numSpheres = 0;
static std::vector<SphereLane> g_sphereLanes;
static std::vector<Sphere::Attributes> g_sphereAttributes;

HitRecord *Sphere::intersect(glm::vec3 &rayDir, glm::vec3 &rayOrigin, HitRecord &ret)
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


    for (unsigned i = 0; i < g_numSpheres; i++)
    {
        unsigned laneIdx = i / c_sphereLaneSz;
        unsigned sphereIdx = i % c_sphereLaneSz;

        const SphereLane& lane = g_sphereLanes[laneIdx];
        glm::vec3 pos = glm::vec3(lane.x[sphereIdx], lane.y[sphereIdx], lane.z[sphereIdx]);
        L = pos - rayOrigin;
        distSq = glm::dot(L, L);

        // Do not accept hits from within a sphere
        if (distSq <= lane.radiusSq[sphereIdx])
        {
            goto Miss;
        }

        tca = glm::dot(L, rayDir);
        d2 = distSq - tca * tca;

        // closest point is outside the radius
        if (d2 > lane.radiusSq[sphereIdx])
        {
            goto Miss;
        }

        thc = sqrt(lane.radiusSq[sphereIdx] - d2); 
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

unsigned Sphere::create(const Sphere::_Create &createStruct)
{
    unsigned sphereIdx = (g_numSpheres) % c_sphereLaneSz;
    if (sphereIdx == 0)
    {
        g_sphereLanes.emplace_back();
    }
    auto &lane = g_sphereLanes.back();
    lane.x[sphereIdx] = createStruct.position.x;
    lane.y[sphereIdx] = createStruct.position.y;
    lane.z[sphereIdx] = createStruct.position.z;
    lane.radiusSq[sphereIdx] = createStruct.radius * createStruct.radius;

    g_sphereAttributes.emplace_back(createStruct.attributes);
    return ++g_numSpheres;
}