#pragma once
#include "glm/glm.hpp"
#include "hitrecord.h"
#include "base_shape.h"

class Triangle : public BaseShape<Triangle>
{
    public:
        struct Attributes
        {
            glm::vec3 color;
        };

        struct _Create
        {
            glm::vec3 A;
            glm::vec3 B;
            glm::vec3 C;

            Attributes attributes;
        };

        static bool intersect_impl(_Intersect &_in);
        static bool intersect_non_vectorized_impl(_Intersect &_in);
        static unsigned create(const _Create &);

    private:
        Triangle();
};