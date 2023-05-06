#pragma once
#include "glm/glm.hpp"
#include "hitrecord.h"
#include "base_shape.h"
#include <vector>
#include "mesh.h"
#include "box.h"

class Triangle : public BaseShape<Triangle>
{
    private:
        friend class Mesh;
        friend class KDTree;
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
        };

        static AxisAlignedBoundingBox getBoundingBox(unsigned startIdx, unsigned numElements);
        static bool intersect_impl(_Intersect &_in);
        static bool intersect_non_vectorized_impl(_Intersect &_in);
    private:
        static AxisAlignedBoundingBox getTriangleBoundingBox(unsigned idx);
        static unsigned create(const _Create &);
        static const Mesh::Attributes *getMeshAttributes(unsigned triangleIdx);

    private:
        static constexpr unsigned c_triangleLaneSz = 8;
        struct TriangleLane
        {
            float Ax[c_triangleLaneSz];
            float Ay[c_triangleLaneSz];
            float Az[c_triangleLaneSz];
            float Bx[c_triangleLaneSz];
            float By[c_triangleLaneSz];
            float Bz[c_triangleLaneSz];
            float Cx[c_triangleLaneSz];
            float Cy[c_triangleLaneSz];
            float Cz[c_triangleLaneSz];
        } __attribute__((aligned (32)));

        static inline unsigned m_numTriangles = 0;
        static inline std::vector<TriangleLane> m_triangleLanes;
        static inline std::vector<Triangle::Attributes> m_triangleAttributes;
        Triangle();
};