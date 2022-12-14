#include "glm/common.hpp"
#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "glm/glm.hpp"
#include "sphere.h"
#include "light.h"
#include <algorithm>
#include <cstdint>
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

        float dist_x = ((float) rand() / RAND_MAX) * 30.0f * g_ratio - 15.0f * g_ratio;
        float dist_y = ((float) rand() / RAND_MAX) * 30.0f - 15.0f;
        float dist_z = ((float) rand() / RAND_MAX) * 20.0f;

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
    glm::vec3 lightDir = glm::normalize(light.position - hr.spherePos);
    float factor = std::max(0.0f, glm::dot(hr.hitNormal, lightDir));
    return factor;
}

static inline glm::u8vec3 toOutputChannelType(glm::vec3& in)
{
    return glm::clamp(in * 255.0f, glm::vec3(0), glm::vec3(255));
}

static inline float shadeSpecularFactor(const Light &light, const HitRecord &hr, const glm::vec3 rayDir)
{
    glm::vec3 lightDir = glm::normalize(light.position - hr.spherePos);
    glm::vec3 reflectedLightDir = glm::reflect(lightDir, hr.hitNormal);

    float factor = glm::pow(glm::max(0.0f, glm::dot(reflectedLightDir, rayDir)), 7);
    return factor;
}

static bool canSeeLight(const Light &light, const glm::vec3 &hitPoint)
{
    glm::vec3 lightDir = glm::normalize(hitPoint - light.position);
    HitRecord hr;
    HitRecord *out = Sphere::intersect(lightDir, light.position, hr);
    if (!out)
    {
        return false;
    }
    glm::vec3 distToPoint = out->hitPoint - hitPoint;
    if (glm::dot(distToPoint, distToPoint) < 0.001f)
    {
        return true;
    }
    return false;
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

void rayTrace(RayTraceData data)
{
    glm::vec3 rayOrigin = {0, 0, 0};
    glm::vec3 rayDir = {-g_ratio, 1.0f, 1};

    constexpr float widthStep = 2.0f * g_ratio / g_width;
    constexpr float heightStep = 2.0f / g_height;
    
    HitRecord hr;

    std::vector<Light> lights;
    lights.push_back({{0, 0, 0}, 10.0f});
    lights.push_back({{0, 10, 10}, 10.0f});
    lights.push_back({{-8, -5, 5}, 10.0f});
    lights.push_back({{0, 0, 10}, 10.0f});

    unsigned imageIdx = data.startRow * g_width * STBI_rgb;
    rayDir.y -= heightStep * data.startRow;

    for (unsigned i = data.startRow; i < data.endRow; i++)
    {
        rayDir.x = -g_ratio;
        for (unsigned j = 0; j < g_width; j++)
        {
            const static unsigned recursionDepth = 5;
            
            glm::vec3 finalColor = glm::vec3(0);
            rayOrigin = {0, 0, 0};
            glm::vec3 rayNorm = glm::normalize(rayDir);

            for (unsigned i = 0; i < recursionDepth; i++)
            {
                HitRecord *hr_p = Sphere::intersect(rayNorm, rayOrigin, hr);
                if (!hr_p)
                {
                    break;
                }
                float weight = 1.0f / (i + 1);

                float lightingFactor = getLightingFactor(lights, hr, rayDir);
                glm::vec3 color = hr.color * lightingFactor;
                finalColor = ((1.0f - weight) * finalColor) + (weight * color);

                rayOrigin = hr.hitPoint;
                rayNorm = glm::reflect(rayNorm, hr.hitNormal);
            }

            glm::u8vec3 finalColorU8 = toOutputChannelType(finalColor);
            for (unsigned k = 0; k < STBI_rgb; k++)
            {
                data.imageData[imageIdx++] = finalColorU8[k];
            }

            rayDir.x += widthStep;
        }
        rayDir.y -= heightStep;
    }
}

int main()
{
    srand(time(NULL));
    std::vector<unsigned> sphereIds;
    generateSpheres(sphereIds, 1024);
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
    
    stbi_write_bmp("output.bmp", g_width, g_height, STBI_rgb, imageData);
}