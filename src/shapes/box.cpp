#include "box.h"
#include "glm/common.hpp"
#include "utils.h"
#include <span>
#include "base_shape.h"
#include "config.h"

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

bool AxisAlignedBoundingBox::intersect(const _Intersect &_in, const glm::vec3 &invRayDir, float &hitTmin, float &hitTmax) const
{
    float tmin = 0;
    float tmax = _in.clippingDistance;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        float tNear = (minCorner[i] - _in.rayOrigin[i]) * invRayDir[i];
        float tFar = (maxCorner[i] - _in.rayOrigin[i]) * invRayDir[i];

        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar) std::swap(tNear, tFar);

        // Update _tFar_ to ensure robust ray--bounds intersection
        tmin = tNear > tmin ? tNear : tmin;
        tmax = tFar < tmax ? tFar : tmax;
        if (tmin > tmax) return false;
    }
    hitTmin = tmin;
    hitTmax = tmax;
    return true;
}