#include "box.h"
#include "glm/common.hpp"
#include "utils.h"
#include <span>

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Union(const AxisAlignedBoundingBox &b2)
{
    for (int i = 0; i < minCorner.length(); i++)
    {
        minCorner[i] = glm::min(minCorner[i], b2.minCorner[i]);
    }
    for (int i = 0; i < maxCorner.length(); i++)
    {
        maxCorner[i] = glm::max(maxCorner[i], b2.maxCorner[i]);
    }
    return *this;
}

Axis AxisAlignedBoundingBox::maximumExtent() const
{
    glm::vec3 cornerToCorner = maxCorner - minCorner;
    return static_cast<Axis>(getMaxElementIndex(cornerToCorner));
}

float AxisAlignedBoundingBox::surfaceArea() const
{
    glm::vec3 vec = maxCorner - minCorner;
    return ((2 * vec.x * vec.y) + (2 * vec.x * vec.z) + (2 * vec.y * vec.z));
}