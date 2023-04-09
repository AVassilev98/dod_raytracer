#pragma once
#include "glm/glm.hpp"
#include "hitrecord.h"
#include <limits>
#include <memory>

class Sphere
{
    friend class std::allocator<Sphere>;
    friend class new_allocator;

    public:
        struct Attributes
        {
            glm::vec3 color;
        };

        struct _Create
        {
            glm::vec3 position;
            float radius;
            Attributes attributes;
        };

        struct Intersect 
        {
            const glm::vec3 &rayDir;
            const glm::vec3 &rayOrigin;
            bool returnOnAny = false;
            float clippingDistance = std::numeric_limits<float>::infinity();
            HitRecord &record;
        };

        static bool intersect(Intersect &_in);
        static bool intersect_non_vectorized(Intersect &_in);
        static unsigned create(const _Create &);
        static Sphere *getSphere(unsigned index);

    private:
        Sphere(const _Create &);
};