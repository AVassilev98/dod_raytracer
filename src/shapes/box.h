#pragma once
#include "glm/glm.hpp"
#include "types.h"
#include "base_shape.h"

struct AxisAlignedBoundingBox
{
    glm::vec3 minCorner;
    glm::vec3 maxCorner;

    AxisAlignedBoundingBox &Union(const AxisAlignedBoundingBox &b2);
    bool intersect(const _Intersect &_in, const glm::vec3 &invRayDir, float &tminOut, float &tmaxOut) const;
    Axis maximumExtent() const;
    float surfaceArea() const;
};