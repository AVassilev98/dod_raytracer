#pragma once

#include "glm/glm.hpp"
#include "hitrecord.h"
#include <limits>


struct _Intersect
{
    glm::vec3 rayDir;
    glm::vec3 rayOrigin;
    bool returnOnAny = false;
    float clippingDistance = std::numeric_limits<float>::infinity();
    HitRecord &record;
};

template <class DerivedShape>
class BaseShape
{
    public:

        static bool intersect(_Intersect &_in) { return DerivedShape::intersect_impl(_in); }
        static bool intersect_non_vectorized(_Intersect &_in) { return DerivedShape::intersect_non_vectorized_impl(_in); }

    private:
        friend DerivedShape;
        BaseShape() = default;
};