#include "kdtree.h"
#include "box.h"
#include "glm/fwd.hpp"
#include "triangle.h"
#include "types.h"
#include <cstdint>
#include <limits>
#include <span>
#include "utils.h"
#include "config.h"

struct KDTree::AxisOffsetInEdge
{
    enum class LocationInEdge
    {
        Start,
        End,
    };

    LocationInEdge m_type;
    unsigned m_laneId;
    float m_offset;

    AxisOffsetInEdge(float offset, unsigned laneId, LocationInEdge type)
        : m_offset(offset)
        , m_laneId(laneId)
        , m_type(type)
    {}
};

struct KDTree::LaneBoundingBox
{
    unsigned m_laneId;
    AxisAlignedBoundingBox m_boundingBox;

    LaneBoundingBox(unsigned laneId, AxisAlignedBoundingBox &boundingBox)
        : m_laneId(laneId)
        , m_boundingBox(boundingBox)
    {}
};

void KDTree::Node::initLeafNode(const std::span<unsigned> &laneNums, std::vector<unsigned> &allLaneIndices)
{
    m_flags = static_cast<unsigned int>(KDTree::Node::Flags::Leaf);
    // lower two bits reserved for flags
    m_numTriangleLanes |= (laneNums.size()) << 2;
    if (laneNums.size() == 0)
    {
        m_triangleLaneIdx = 0;
    }
    m_triangleLaneIdx = allLaneIndices.size();
    for (unsigned idx : laneNums)
    {
        allLaneIndices.push_back(idx);
    }
}

void KDTree::Node::initInteriorNode(Axis splitAxis, float splitOffset, unsigned rightChildIdx)
{
    m_flags = static_cast<unsigned int>(splitAxis);
    m_splitOffset = splitOffset;
    // lower two bits reserved for flags
    m_rightChildIdx |= rightChildIdx << 2;
}

void KDTree::init()
{
    constexpr unsigned numAxes = static_cast<unsigned>(Axis::NumAxes);
    constexpr float infinity = std::numeric_limits<float>::infinity();
    unsigned numLanes = Triangle::m_triangleLanes.size();

    m_maxDepth = std::round(std::log2(8.0f + (1.3f * numLanes)));
    unsigned numTriangles = Triangle::m_numTriangles;
    unsigned triangleLaneSize = Triangle::c_triangleLaneSz;
    std::vector<unsigned> laneNumbers;

    std::vector<AxisAlignedBoundingBox> laneBoundingBoxes;
    AxisAlignedBoundingBox worldBound = 
    {
        .minCorner = glm::vec3(infinity, infinity, infinity),
        .maxCorner = glm::vec3(-infinity, -infinity, -infinity)
    };

    for (unsigned i = 0; i < numTriangles; i+= triangleLaneSize)
    {
        unsigned len = std::min(triangleLaneSize, numTriangles - i);
        laneBoundingBoxes.push_back(Triangle::getBoundingBox(i, len));
        worldBound.Union(laneBoundingBoxes.back());
        laneNumbers.push_back(i / triangleLaneSize);
    }    
    m_bounds = worldBound;
    recursivelyConstructNodes(m_maxDepth, 0, worldBound, laneBoundingBoxes, laneNumbers);
}

void KDTree::recursivelyConstructNodes
(
    unsigned depth,
    unsigned badRefines,
    const AxisAlignedBoundingBox &nodeBounds,
    const std::vector<AxisAlignedBoundingBox> &boundingBoxes,
    std::vector<unsigned> &laneNums
)
{
    constexpr unsigned numAxes = static_cast<unsigned>(Axis::NumAxes);

    if (depth == 0 || laneNums.size() <= m_minLanes)
    {
        m_nodes.emplace_back();
        m_nodes.back().initLeafNode(std::span(laneNums), m_primNums);
        return;
    }

    std::vector<AxisOffsetInEdge> boundingBoxEdges[numAxes];
    for(unsigned i = 0; i < numAxes; i++)
    {
        boundingBoxEdges[i].reserve(laneNums.size() * 2);
    }
    for (unsigned i = 0; i < laneNums.size(); i++)
    {
        const unsigned laneIdx = laneNums[i];
        const AxisAlignedBoundingBox &boundingBox = boundingBoxes[laneIdx];
        for (unsigned j = 0; j < numAxes; j++)
        {
            boundingBoxEdges[j].emplace_back(boundingBox.minCorner[j], laneIdx, AxisOffsetInEdge::LocationInEdge::Start);
            boundingBoxEdges[j].emplace_back(boundingBox.maxCorner[j], laneIdx, AxisOffsetInEdge::LocationInEdge::End);
        }
    }
    for (unsigned i = 0; i < numAxes; i++)
    {
        std::sort(boundingBoxEdges[i].begin(), boundingBoxEdges[i].end(), 
            [](const AxisOffsetInEdge &a, const AxisOffsetInEdge &b) -> bool 
                {
                    return a.m_offset < b.m_offset;
                }
        );

    }

    // do SAH https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#sec:sah
    unsigned bestSplitIdx = UINT32_MAX;
    unsigned bestSplitCost = UINT32_MAX;
    float originalSplitCost = Config::IntersectCost * laneNums.size();
    Axis splitAxis;
    unsigned maxAxis = static_cast<unsigned>(nodeBounds.maximumExtent());
    float invTotalSurfaceArea = 1.0f / nodeBounds.surfaceArea();
    for (unsigned i = 0; i < numAxes; i++)
    {
        unsigned axisIndex = (maxAxis + i) % numAxes;

        unsigned numLanesLeftOfSplit = 0;
        unsigned numLanesRightOfSplit = laneNums.size();

        for(unsigned j = 0; j < boundingBoxEdges[axisIndex].size(); j++)
        {
            const AxisOffsetInEdge &edgePoint = boundingBoxEdges[axisIndex][j];
            float edgeOffs = edgePoint.m_offset;
            if (edgePoint.m_type == AxisOffsetInEdge::LocationInEdge::End)
            {
                numLanesRightOfSplit--;
            }

            if (edgeOffs >= nodeBounds.minCorner[axisIndex] && edgeOffs <= nodeBounds.maxCorner[axisIndex])
            {
                glm::vec3 maxCornerLeft = nodeBounds.maxCorner;
                glm::vec3 minCornerRight = nodeBounds.minCorner;
                maxCornerLeft[axisIndex] = edgePoint.m_offset;
                minCornerRight[axisIndex] = edgePoint.m_offset;

                float leftSurfaceArea = AxisAlignedBoundingBox{nodeBounds.minCorner, maxCornerLeft}.surfaceArea();
                float rightSurfaceArea = AxisAlignedBoundingBox{minCornerRight, nodeBounds.maxCorner}.surfaceArea();

                float intersectLeftProbability = leftSurfaceArea * invTotalSurfaceArea;
                float intersectRightProbability = rightSurfaceArea * invTotalSurfaceArea;

                float emptyBonus = (!numLanesRightOfSplit || !numLanesRightOfSplit) ? Config::EmptyBonus : 0.0f;
                float cost = Config::TraversalCost 
                                + Config::IntersectCost
                                * (1 - emptyBonus) 
                                * (intersectLeftProbability * numLanesLeftOfSplit + intersectRightProbability * numLanesRightOfSplit);
                
                if (cost < bestSplitCost)
                {
                    bestSplitCost = cost;
                    splitAxis = static_cast<Axis>(axisIndex);
                    bestSplitIdx = j;
                }
            }

            if (edgePoint.m_type == AxisOffsetInEdge::LocationInEdge::Start)
            {
                numLanesLeftOfSplit++;
            }
        }

        // Check if a reasonable split was found
        if (bestSplitCost < originalSplitCost)
        {
            break;
        }
    }

    if (bestSplitCost > originalSplitCost)
    {
        badRefines++;
    }
    // Allocate new node and init leaf if no good split found
    m_nodes.emplace_back();
    if (bestSplitIdx == UINT32_MAX 
        || badRefines == 3 
        || (bestSplitCost > 4 * originalSplitCost && laneNums.size() < 16))
    {
        m_nodes.back().initLeafNode(std::span(laneNums), m_primNums);
        return;
    }
    unsigned interiorNodeIdx = m_nodes.size() - 1;

    const unsigned splitAxisNumerical = static_cast<unsigned>(splitAxis);
    float splitOffset = boundingBoxEdges[splitAxisNumerical][bestSplitIdx].m_offset;
    // create Bounds for child nodes
    AxisAlignedBoundingBox leftNodeBounds = nodeBounds;
    AxisAlignedBoundingBox rightNodeBounds = nodeBounds;
    leftNodeBounds.maxCorner[splitAxisNumerical] = splitOffset;
    rightNodeBounds.minCorner[splitAxisNumerical] = splitOffset;

    // calculate number of children in each subnode
    std::vector<unsigned> lanesLeftOfSplit;
    std::vector<unsigned> lanesRightOfSplit;
    assert(boundingBoxEdges[splitAxisNumerical].size() == 2 * laneNums.size() && "There should be two EdgePoints for each Lane!");
    for (unsigned i = 0; i < bestSplitIdx; i++)
    {
        const AxisOffsetInEdge &edgePoint = boundingBoxEdges[splitAxisNumerical][i];
        if (edgePoint.m_type == AxisOffsetInEdge::LocationInEdge::Start)
        {
            lanesLeftOfSplit.push_back(edgePoint.m_laneId);
        }
    }
    for (unsigned i = bestSplitIdx + 1; i < boundingBoxEdges[splitAxisNumerical].size(); i++)
    {
        const AxisOffsetInEdge &edgePoint = boundingBoxEdges[splitAxisNumerical][i];
        if (edgePoint.m_type == AxisOffsetInEdge::LocationInEdge::End)
        {
            lanesRightOfSplit.push_back(edgePoint.m_laneId);
        }
    }
    assert(lanesLeftOfSplit.size() + lanesRightOfSplit.size() >= laneNums.size() && "Split caused missing primitives!");

    recursivelyConstructNodes(depth - 1, badRefines, leftNodeBounds, boundingBoxes, lanesLeftOfSplit);
    m_nodes[interiorNodeIdx].initInteriorNode(splitAxis, splitOffset, m_nodes.size());
    recursivelyConstructNodes(depth - 1, badRefines, rightNodeBounds, boundingBoxes, lanesRightOfSplit);
}

const KDTree KDTree::buildTree()
{
    KDTree tree;
    printf("Beginning tree construction\n");
    tree.init();
    printf("Tree construction complete\n");
    Triangle::reorderLanesByIndices(tree.m_primNums);
    return tree;
}

// based on the pbrt implementation with some performance improvements
bool KDTree::intersect(_Intersect &_in) const
{
    struct workItem
    {
        const Node *node;
        float tmin;
        float tmax;
    };
    glm::vec3 invRayDir = glm::vec3(1.0, 1.0, 1.0) / _in.rayDir;
    float tmin;
    float tmax;
    if (!m_bounds.intersect(_in, invRayDir, tmin, tmax) || tmin > _in.clippingDistance)
    {
        return false;
    }

    workItem worklist[64];
    int worklistPos = 0;
    const Node *node = &m_nodes[0];
    bool hit = false;

    while (node)
    {
        if (_in.clippingDistance < tmin)
        {
            break;
        }
        if (!node->isLeaf())
        {
            unsigned axis = static_cast<unsigned>(node->splitAxis());
            float tPlane = (node->splitOffset() - _in.rayOrigin[axis]) * invRayDir[axis];

            const Node *leftChild;
            const Node *rightChild;
            bool leftFirst =
                (_in.rayOrigin[axis] < node->splitOffset()) ||
                (_in.rayOrigin[axis] == node->splitOffset() && _in.rayDir[axis] <= 0);
            if (leftFirst)
            {
                leftChild = node + 1;
                rightChild = &m_nodes[node->rightChildIdx()];
            }
            else
            {
                leftChild = &m_nodes[node->rightChildIdx()];
                rightChild = node + 1;
            }

            // get next child node
            if (tPlane > tmax || tPlane <= 0)
            {
                node = leftChild;
            }
            else if (tPlane < tmin)
            {
                node = rightChild;
            }
            else 
            {
                // put rightChild in worklist
                worklist[worklistPos].node = rightChild;
                worklist[worklistPos].tmin = tPlane;
                worklist[worklistPos].tmax = tmax;
                ++worklistPos;
                node = leftChild;
                tmax = tPlane;
            }
        }
        else
        {
            // Check for intersections inside leaf node
            int numLanes = node->numLanes();
            std::span<Triangle::TriangleLane> laneRange(&Triangle::m_triangleLanes[node->laneStartIdx()], numLanes);
            if(Triangle::intersectInRange(_in, laneRange, node->laneStartIdx()))
            {
                if(_in.returnOnAny)
                {
                    return true;
                }
                hit = true;
                _in.clippingDistance = _in.record.t;
            }            

            // Grab next node to process from worklist
            if (worklistPos > 0)
            {
                --worklistPos;
                node = worklist[worklistPos].node;
                tmin = worklist[worklistPos].tmin;
                tmax = worklist[worklistPos].tmax;
            } 
            else
            {
                break;
            }
        }
    }
    return hit;
}