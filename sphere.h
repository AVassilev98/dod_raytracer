#pragma once
#include "glm/glm.hpp"
#include "hitrecord.h"
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

        static HitRecord *intersect(const glm::vec3 &rayDir, const glm::vec3 &rayOrigin, HitRecord &ret);
        static unsigned create(const _Create &);
        static Sphere *getSphere(unsigned index);

    private:
        Sphere(const _Create &);
};