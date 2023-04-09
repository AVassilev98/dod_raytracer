#pragma once
#include "glm/glm.hpp"

struct HitRecord
{
    float t;
    glm::vec3 color;
    glm::vec3 hitNormal;
    glm::vec3 hitPoint;
};