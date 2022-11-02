#include "glm/glm.hpp"
#include "sphere.h"
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


void *rayTrace(RayTraceData data)
{
    glm::vec3 rayOrigin = {-1, 1, 0};
    glm::vec3 rayDir = {0, 0, 1};
    std::cout << "startRow " << data.startRow << " endRow " << data.endRow << std::endl;


    constexpr float widthStep = 2.0f * g_ratio / g_width;
    constexpr float heightStep = 2.0f / g_height;

    unsigned sphereCount;
    
    HitRecord hr;
    
    unsigned imageIdx = data.startRow * g_width * STBI_rgb;
    rayOrigin.y -= heightStep * data.startRow;

    for (unsigned i = data.startRow; i < data.endRow; i++)
    {
        rayOrigin.x = -1;
        for (unsigned j = 0; j < g_width; j++)
        {
            HitRecord *hr_p = Sphere::intersect(rayDir, rayOrigin, hr);
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

int main()
{
    srand(time(NULL));
    std::vector<unsigned> sphereIds;
    generateSpheres(sphereIds, 8192);
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