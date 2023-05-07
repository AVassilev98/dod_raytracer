#pragma once
#include <vector>
#include <span>
#include "config.h"
#include "glm/glm.hpp"
#include "box.h"
#include "base_shape.h"

class KDTree
{
    public:
        const static KDTree buildTree();
        bool intersect(_Intersect &_in) const;

    private:
        struct Node
        {
            public:
                void initLeafNode(const std::span<unsigned> &laneNums, std::vector<unsigned> &allLaneIndices);
                void initInteriorNode(Axis splitAxis, float splitOffset, unsigned rightChildIdx);

                bool isLeaf() const { return static_cast<Flags>(m_flags & 0x3) == Flags::Leaf; }
                Axis splitAxis() const { return static_cast<Axis>(m_flags & 0x3); }
                float splitOffset() const { return m_splitOffset; }
                unsigned numLanes() const { return m_numTriangleLanes >> 2; }
                unsigned laneStartIdx() const { return m_triangleLaneIdx; }
                unsigned rightChildIdx() const { return m_rightChildIdx >> 2; }

            private:
                enum class Flags
                {
                    xSplit = 0,
                    ySplit = 1,
                    zSplit = 2,
                    Leaf = 3,
                };

            private:
                union {
                    unsigned m_flags;
                    unsigned m_numTriangleLanes;
                    unsigned m_rightChildIdx;
                };
                union {
                    float m_splitOffset;
                    unsigned m_triangleLaneIdx;
                };   
        };
        struct AxisOffsetInEdge;
        struct LaneBoundingBox;

    private:
        void recursivelyConstructNodes
        (
            unsigned depth,
            unsigned badRefines,
            const AxisAlignedBoundingBox &nodeBounds,
            const std::vector<AxisAlignedBoundingBox> &boundingBoxes,
            std::vector<unsigned> &laneNums
        );
        void init();

    private:
        std::vector<Node> m_nodes;
        std::vector<unsigned> m_primNums;
        unsigned m_maxDepth;
        unsigned m_minLanes = Config::MaxPrims;
        AxisAlignedBoundingBox m_bounds;
};

