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
            glm::u8vec3 color;
        };

        struct _Create
        {
            glm::vec3 position;
            float radius;
            Attributes attributes;
        };

        static HitRecord *intersect(const Sphere *spheres, size_t count, glm::vec3 &rayDir, glm::vec3 &rayOrigin, HitRecord &ret);
        static unsigned create(const _Create &);
        static Sphere *getSphere(unsigned index);
        static const Sphere *getAllSpheres(unsigned &count); 

    private:
        Sphere(const _Create &);

        glm::vec3 m_position;
        float m_radiusSq;
};