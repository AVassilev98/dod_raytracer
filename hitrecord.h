#pragma once
#include "glm/glm.hpp"

struct HitRecord
{
    float distSq;
    float t;
    glm::vec3 color;
};