#include "glm/common.hpp"
#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "glm/glm.hpp"
#include "sphere.h"
#include "plane.h"
#include "light.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include "common_defs.h"

#include <sys/sysinfo.h>
#include "iostream"
#include <thread>

void generateSpheres(std::vector<unsigned> &sphereIds, unsigned numSpheres)
{
    for (unsigned i = 0; i < numSpheres; i++)
    {
        float r = ((float) rand() / RAND_MAX);
        float g = ((float) rand() / RAND_MAX);
        float b = ((float) rand() / RAND_MAX);
        float radius = 1.0f;

        float dist_x = ((float) rand() / RAND_MAX) * 10.0f - 5.0f;
        float dist_y = ((float) rand() / RAND_MAX) * 10.0f - 5.0f;
        float dist_z = ((float) rand() / RAND_MAX) * 10.0f - 5.0f;

        Sphere::_Create createStruct {
            .position = glm::vec3(dist_x, dist_y, dist_z),
            .radius = radius,
            .attributes = 
            {
                glm::vec3(r, g, b)
            },
        };

        sphereIds.emplace_back(Sphere::create(createStruct));
    }
}

void generatePlanes(std::vector<unsigned> &planeIds)
{
    constexpr std::array<Plane::_Create, 6> planes = {{
        {
            .normal = {0.0f, 0.0f, -1.0f},
            .position = {0.0f, 0.0f, 5.0f},
            .attributes = 
            {
                .color = {0.195f, 0.410f, 0.610f},
            },
        },
        {
            .normal = {0.0f, 0.0f, 1.0f},
            .position = {0.0f, 0.0f, -5.0f},
            .attributes = 
            {
                .color = {0.493, 0.265, 0.590},
            },
        },
        {
            .normal = {0.0f, -1.0f, 0.0f},
            .position = {0.0f, 5.0f, 0.0f},
            .attributes = 
            {
                .color = {0.276, 0.600, 0.411},
            },
        },
        {
            .normal = {0.0f, 1.0f, 0.0f},
            .position = {0.0f, -5.0f, 0.0f},
            .attributes = 
            {
                .color = {0.292, 0.680, 0.674},
            },
        },
        {
            .normal = {1.0f, 0.0f, 0.0f},
            .position = {-5.0f, 0.0f, 0.0f},
            .attributes = 
            {
                .color = {0.720, 0.288, 0.389},
            },
        },
        {
            .normal = {-1.0f, 0.0f, 0.0f},
            .position = {5.0f, 0.0f, 0.0f},
            .attributes = 
            {
                .color = {0.680, 0.224, 0.224},
            },
        },
    }};
    for (const Plane::_Create &createStruct : planes)
    {
        planeIds.emplace_back(Plane::create(createStruct));
    }

}


struct RayTraceData
{
    uint8_t *imageData;
    unsigned startRow;
    unsigned endRow;
};

float shadeAmbientFactor()
{
    return 0.2f;
}

static inline float shadeDiffuseFactor(const Light &light, const HitRecord &hr)
{
    glm::vec3 lightDir = glm::normalize(light.position - hr.hitPoint);
    float factor = std::max(0.0f, glm::dot(hr.hitNormal, lightDir));
    return factor;
}

static inline glm::u8vec3 toOutputChannelType(glm::vec3& in)
{
    return glm::clamp(in * 255.0f, glm::vec3(0), glm::vec3(255));
}

static inline float shadeSpecularFactor(const Light &light, const HitRecord &hr, const glm::vec3 rayDir)
{
    glm::vec3 lightDir = glm::normalize(light.position - hr.hitPoint);
    glm::vec3 reflectedLightDir = glm::reflect(lightDir, hr.hitNormal);

    float factor = glm::pow(glm::max(0.0f, glm::dot(reflectedLightDir, rayDir)), 7);
    return factor;
}

static bool canSeeLight(const Light &light, const glm::vec3 &hitPoint)
{
    glm::vec3 lightDir = light.position - hitPoint;
    float lightDistance = glm::length(lightDir);
    lightDir /= lightDistance;

    HitRecord hr;

    _Intersect intersectParams = {
        .rayDir = lightDir,
        .rayOrigin = hitPoint + lightDir * 0.01f,
        .returnOnAny = true,
        .clippingDistance = lightDistance,
        .record = hr,
    };

    bool hit = Sphere::intersect(intersectParams);
    if (hit)
    {
        return false;
    }
    hit |= Plane::intersect(intersectParams);
    if (hit)
    {
        return false;
    }
    return true;
}

float getLightingFactor(const std::vector<Light> &lights, const HitRecord &hr, const glm::vec3 &rayDir)
{
    float lightingFactor = shadeAmbientFactor();
    for (const auto &light : lights)
    {
        if (!canSeeLight(light, hr.hitPoint))
        {
            continue;
        }

        // quadratic intensity fallof with distance
        glm::vec3 distToLight = light.position - hr.hitPoint;
        float distanceFactor = light.intensity / glm::dot(distToLight, distToLight);

        float singleLightFactor = 0.0f;
        singleLightFactor += shadeDiffuseFactor(light, hr);
        singleLightFactor += shadeSpecularFactor(light, hr, rayDir);
        singleLightFactor *= distanceFactor;

        lightingFactor += singleLightFactor;
    }

    return lightingFactor;
}

void compareHitRecords(const HitRecord *hrA, const HitRecord *hrB, unsigned row, unsigned col, unsigned depth)
{
    constexpr float epsilon = 0.01f;
    if (!hrA && !hrB)
    {
        // printf("(%4u,%4u,%4u) - OK\n", row, col, depth);
        return;
    }
    if (!hrA)
    {
        printf("(%4u,%4u,%4u) - RECORD A MISS - RECORD B HIT\n", row, col, depth);
        return;
    }
    if (!hrB)
    {
        printf("(%4u,%4u,%4u) - RECORD A HIT - RECORD B MISS\n", row, col, depth);
        return;
    }
    if (fabs(hrA->t - hrB->t) > epsilon)
    {
        printf("(%4u,%4u,%4u) - T mismatch -- A: %f, B: %f\n", row, col, depth, hrA->t, hrB->t);
        return;
    }

    // printf("(%4u,%4u,%4u) - OK\n", row, col, depth);
}

void rayTrace(RayTraceData data)
{
    glm::vec3 rayOrigin = {0, 0, -4.9};
    glm::vec3 rayDir = {-g_ratio, 1.0f, 1};

    constexpr float widthStep = 2.0f * g_ratio / g_width;
    constexpr float heightStep = 2.0f / g_height;
    
    HitRecord hr;

    std::vector<Light> lights;
    lights.push_back({{0.0f, 0.0f, 0.0f}, 3.0f});
    lights.push_back({{4.0f, 4.3f, 3.3f}, 1.0f});
    lights.push_back({{-4.f, -2.95f, 3.95f}, 1.0f});
    lights.push_back({{3.95f, -4.2f, 3.3f}, 1.0f});
    lights.push_back({{-2.9f, 4.2f, 3.8f}, 1.0f});
    lights.push_back({{3.95f, 2.8f, -4.3f}, 1.0f});
    lights.push_back({{-3.0f, -3.8f, -3.3f}, 1.0f});
    lights.push_back({{4.2f, -4.2f, -3.4f}, 1.0f});
    lights.push_back({{-2.9f, 4.4f, -3.5f}, 1.0f});

    unsigned imageIdx = data.startRow * g_width * STBI_rgb;
    rayDir.y -= heightStep * data.startRow;

    for (unsigned i = data.startRow; i < data.endRow; i++)
    {
        for (unsigned j = 0; j < g_width; j++)
        {
            const static unsigned recursionDepth = 10;
            
            glm::vec3 finalColor = glm::vec3(0);
            glm::vec3 rayNorm = glm::normalize(rayDir);

            _Intersect intersectParams {
                .rayDir = rayNorm,
                .rayOrigin = {0, 0, -4.9},
                .record = hr
            };

            for (unsigned k = 0; k < recursionDepth; k++)
            {
                intersectParams.clippingDistance = std::numeric_limits<float>::infinity();
                bool hit = Sphere::intersect(intersectParams);
                intersectParams.clippingDistance = intersectParams.record.t;
                hit |= Plane::intersect(intersectParams);
                if (!hit)
                {
                    break;
                }
                float weight = 1.0f / pow(2.0f, k);

                float lightingFactor = getLightingFactor(lights, hr, rayDir);
                glm::vec3 color = hr.color * lightingFactor;
                finalColor = ((1.0f - weight) * finalColor) + (weight * color);

                intersectParams.rayDir = glm::reflect(intersectParams.rayDir, hr.hitNormal);
                intersectParams.rayOrigin = hr.hitPoint + intersectParams.rayDir * 0.001f;
            }

            glm::u8vec3 finalColorU8 = toOutputChannelType(finalColor);
            for (unsigned k = 0; k < STBI_rgb; k++)
            {
                data.imageData[imageIdx++] = finalColorU8[k];
            }

            rayDir.x += widthStep;
        }
        rayDir.x = -g_ratio;
        rayDir.y -= heightStep;
    }
}

int main()
{
    srand(time(NULL));
    std::vector<unsigned> sphereIds;
    std::vector<unsigned> planeIds;

    generateSpheres(sphereIds, 16);
    generatePlanes(planeIds);
    uint8_t *imageData = (uint8_t *)calloc(g_width * g_height * STBI_rgb, sizeof(uint8_t));

    unsigned numCores = get_nprocs();
    std::vector<RayTraceData> threadData(numCores);
    unsigned startRow = 0;

    std::vector<std::jthread> threads;
    for (unsigned i = 0; i < numCores; i++)
    {
        if (startRow >= g_height)
        {
            break;
        }

        threadData[i].imageData = imageData;
        threadData[i].startRow = startRow;
        unsigned endRow = startRow + ((g_height + numCores - 1) / numCores);
        threadData[i].endRow = endRow > g_height ? g_height : endRow;

        threads.emplace_back(rayTrace, threadData[i]);
        startRow = threadData[i].endRow;
    }
    threads.clear();
    
    stbi_write_png("output.png", g_width, g_height, STBI_rgb, imageData, g_width * STBI_rgb);
}