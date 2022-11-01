#include "glm/glm.hpp"
#include "sphere.h"
#include <cstdint>
#include <pthread.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include "common_defs.h"

#include "pthread.h"
#include <sys/sysinfo.h>
#include "iostream"

void generateSpheres(std::vector<unsigned> &sphereIds, unsigned numSpheres)
{
    for (unsigned i = 0; i < numSpheres; i++)
    {
        uint8_t r = rand() % UINT8_MAX;
        uint8_t g = rand() % UINT8_MAX;
        uint8_t b = rand() % UINT8_MAX;
        float radius = 0.1f;

        float dist_x = ((float) rand() / RAND_MAX) * 2.0f * g_ratio - 1.0f;
        float dist_y = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        float dist_z = ((float) rand() / RAND_MAX) * 500.0f + 500.0f;

        Sphere::_Create createStruct {
            .position = glm::vec3(dist_x, dist_y, dist_z),
            .radius = radius,
            .attributes = 
            {
                glm::u8vec3(r, g, b)
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


void *rayTrace(RayTraceData &data)
{
    glm::vec3 rayOrigin = {-1, 1, 0};
    glm::vec3 rayDir = {0, 0, 1};

    constexpr float widthStep = 2.0f * g_ratio / g_width;
    constexpr float heightStep = 2.0f / g_height;

    unsigned sphereCount;
    const Sphere *spheres = Sphere::getAllSpheres(sphereCount);
    
    HitRecord hr;
    
    unsigned imageIdx = data.startRow * g_width * STBI_rgb;
    rayOrigin.y -= heightStep * data.startRow;

    for (unsigned i = data.startRow; i < data.endRow; i++)
    {
        rayOrigin.x = -1;
        for (unsigned j = 0; j < g_width; j++)
        {
            HitRecord *hr_p = Sphere::intersect(spheres, sphereCount, rayDir, rayOrigin, hr);
            if (hr_p)
            {
                for (unsigned k = 0; k < STBI_rgb; k++)
                {
                    data.imageData[imageIdx++] = hr.color[k];
                }
            }
            else
            {
                for (unsigned k = 0; k < STBI_rgb; k++)
                {
                    data.imageData[imageIdx++] = 0;
                }
            }
            
            rayOrigin.x += widthStep;
        }
        rayOrigin.y -= heightStep;
    }

    return nullptr;
}

void *rayTrace(void *data)
{
    if (data == nullptr)
    {
        return nullptr;
    }
    return rayTrace(*(RayTraceData *)data);
}

int main()
{
    srand(time(NULL));
    std::vector<unsigned> sphereIds;
    generateSpheres(sphereIds, 100);
    uint8_t *imageData = (uint8_t *)malloc(g_width * g_height * STBI_rgb);

    unsigned numCores = get_nprocs();
    std::vector<pthread_t> threads(numCores, -1);
    unsigned startRow = 0;
    
    RayTraceData data;
    data.imageData = imageData;

    for (unsigned i = 0; i < numCores; i++)
    {
        if (startRow >= g_height)
        {
            break;
        }

        data.startRow = startRow;
        unsigned endRow = startRow + ((g_height + numCores - 1) / numCores);
        data.endRow = endRow > g_height ? g_height : endRow;
        std::cout << "startRow " << data.startRow << " endRow " << data.endRow << std::endl;

        pthread_create(&threads[i], nullptr, rayTrace, &data);
        startRow = data.endRow;
    }
    for (unsigned i = 0; i < numCores; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    
    stbi_write_bmp("output.bmp", g_width, g_height, STBI_rgb, imageData);
}