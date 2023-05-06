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
    else if (laneNums.size() == 1)
    {
        m_singleLaneLeafNode = laneNums[0];
    }
    else
    {
        m_triangleLaneIdx = allLaneIndices.size();
        for (unsigned idx : laneNums)
        {
            allLaneIndices.push_back(idx);
        }
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
    for (unsigned i = 0; i < bestSplitIdx; i++)
    {
        const AxisOffsetInEdge &edgePoint = boundingBoxEdges[splitAxisNumerical][i];
        if (edgePoint.m_type == AxisOffsetInEdge::LocationInEdge::Start)
        {
            lanesLeftOfSplit.push_back(edgePoint.m_laneId);
        }
    }
    for (unsigned i = bestSplitIdx + 1; i < laneNums.size(); i++)
    {
        const AxisOffsetInEdge &edgePoint = boundingBoxEdges[splitAxisNumerical][i];
        if (edgePoint.m_type == AxisOffsetInEdge::LocationInEdge::End)
        {
            lanesRightOfSplit.push_back(edgePoint.m_laneId);
        }
    }

    recursivelyConstructNodes(depth - 1, badRefines, leftNodeBounds, boundingBoxes, lanesLeftOfSplit);
    m_nodes[interiorNodeIdx].initInteriorNode(splitAxis, splitOffset, m_nodes.size());
    recursivelyConstructNodes(depth - 1, badRefines, rightNodeBounds, boundingBoxes, lanesRightOfSplit);
}

const KDTree KDTree::buildTree()
{
    KDTree tree;
    tree.init();
    return tree;
}