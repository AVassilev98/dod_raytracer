#pragma once
#include "glm/glm.hpp"
#include "hitrecord.h"
#include "base_shape.h"
#include <vector>
#include "mesh.h"
#include "box.h"
#include <span>

class Triangle : public BaseShape<Triangle>
{
    private:
        friend class Mesh;
        friend class KDTree;
    public:

        struct _Create
        {
            glm::vec3 A;
            glm::vec3 B;
            glm::vec3 C;

            glm::vec3 AN;
            glm::vec3 BN;
            glm::vec3 CN;
        };

        static AxisAlignedBoundingBox getBoundingBox(unsigned startIdx, unsigned numElements);
        static bool intersect_impl(_Intersect &_in);
        static bool intersect_non_vectorized_impl(_Intersect &_in);
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
        struct Attributes
        {
            unsigned meshAttrIdx[c_triangleLaneSz];
            glm::vec3 AN[c_triangleLaneSz];
            glm::vec3 BN[c_triangleLaneSz];
            glm::vec3 CN[c_triangleLaneSz];
        };

    private:
        static AxisAlignedBoundingBox getTriangleBoundingBox(unsigned idx);
        static unsigned create(const _Create &);
        static void reorderLanesByIndices(const std::vector<unsigned> &LaneIndices);
        static bool intersectInRange(_Intersect &_in, const std::span<TriangleLane> &range, unsigned startIdx);

        static inline unsigned m_numTriangles = 0;
        static inline std::vector<TriangleLane> m_triangleLanes;
        static inline std::vector<Triangle::Attributes> m_triangleAttributes;
        Triangle();
};