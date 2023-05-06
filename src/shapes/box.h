#pragma once
#include "glm/glm.hpp"
#include "types.h"

struct AxisAlignedBoundingBox
{
    glm::vec3 minCorner;
    glm::vec3 maxCorner;

    AxisAlignedBoundingBox &Union(const AxisAlignedBoundingBox &b2);
    Axis maximumExtent() const;
    float surfaceArea() const;
};