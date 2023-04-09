#pragma once
#include "glm/glm.hpp"
#include "hitrecord.h"
#include "base_shape.h"
#include <limits>
#include <memory>

class Plane : public BaseShape<Plane>
{
    friend BaseShape<Plane>;

    public:
        struct Attributes
        {
            glm::vec3 color;
        };

        struct _Create
        {
            glm::vec3 normal;
            glm::vec3 position;
            Attributes attributes;
        };

        static unsigned create(const _Create &);
        static Plane *getPlane(unsigned index);

    protected:
        glm::vec3 m_normal;
        glm::vec3 m_position;

        static bool intersect_impl(_Intersect &_in);
        static bool intersect_non_vectorized_impl(_Intersect &_in);

    private:
        Plane(const _Create &);
};