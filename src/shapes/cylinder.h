#pragma once
#include "glm/glm.hpp"
#include "base_shape.h"
#include "hitrecord.h"
#include <limits>
#include <memory>

class Cylinder : public BaseShape<Cylinder>
{
    friend BaseShape<Cylinder>;

    public:
        struct Attributes
        {
            glm::vec3 color;
        };

        struct _Create
        {
            float radius;
            float height;
            glm::vec3 axis;
            glm::vec3 basePosition;
            Attributes attributes;
        };

        static bool intersect_impl(_Intersect &_in);
        static bool intersect_non_vectorized(_Intersect &_in);
        static unsigned create(const _Create &);
        static Cylinder *getCylinder(unsigned index);

    private:

        glm::vec3 m_base;
        glm::vec3 m_axis;
        float m_radiusSq;
        float m_height;
        Cylinder(const _Create &);
        bool intersect_cylinder_body(_Intersect &_in, HitRecord &hr) const;
        bool intersect_cylinder_disc(_Intersect &_in, float offset, HitRecord &hr) const;
};