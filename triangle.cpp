#include "triangle.h"
#include "config.h"
#include "glm/geometric.hpp"
#include "hitrecord.h"
#include "vector"
#include <cstdint>

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
    return intersect_non_vectorized_impl(_in);
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

            glm::vec3 CA = A - C;
            glm::vec3 AB = B - A;
            glm::vec3 BC = C - B;
            glm::vec3 norm = -glm::normalize(glm::cross(AB, BC));

            // do ray plane intersection
            // (p0 - p),norm = 0
            // (A - (Ro + Rdt)),norm = 0;
            // t = ((A - Ro),norm)/(Rd,norm)

            float denominator = glm::dot(_in.rayDir, norm);
            if (fabs(denominator) < Config::Epsilon)
            {
                continue;
            }

            float numerator = dot((A - _in.rayOrigin), norm);
            float t = numerator / denominator;
            if (t < 0 || t > maximumDistance)
            {
                continue;
            }

            glm::vec3 hitPoint = _in.rayOrigin + _in.rayDir * t;
            glm::vec3 perp;

            glm::vec3 AH = hitPoint - A;
            perp = glm::cross(AH, AB);
            if (glm::dot(perp, norm) < 0.0f)
            {
                continue;
            }

            glm::vec3 BH = hitPoint - B;
            perp = glm::cross(BH, BC);
            if (glm::dot(perp, norm) < 0.0f)
            {
                continue;
            }

            glm::vec3 CH = hitPoint - C;
            perp = glm::cross(CH, CA);
            if (glm::dot(perp, norm) < 0.0f)
            {
                continue;
            }

            minTriangleIndex = i * c_triangleLaneSz + j;
            maximumDistance = t;
            minNormal = norm;
            minHitPoint = hitPoint;
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
