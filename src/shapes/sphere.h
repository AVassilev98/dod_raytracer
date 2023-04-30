#pragma once
#include "glm/glm.hpp"
#include "base_shape.h"
#include "hitrecord.h"
#include <limits>
#include <memory>

class Sphere : public BaseShape<Sphere>
{
    friend BaseShape<Sphere>;

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

        static bool intersect_impl(_Intersect &_in);
        static bool intersect_non_vectorized(_Intersect &_in);
        static unsigned create(const _Create &);
        static Sphere *getSphere(unsigned index);

    private:
        Sphere(const _Create &);
};