#pragma once
#include <limits>
#include <span>
#include <functional>
#include "assimp/vector3.h"
#include "glm/common.hpp"
#include "glm/fwd.hpp"
#include "glm/glm.hpp"
#include "assimp/mesh.h"

enum class ComparisonResult
{
    LessThan,
    GreaterThan,
    Equal
};

template <typename T, typename E, class Functor>
[[nodiscard]] const T* binarySearch(const std::span<T> &elements, const E& searchElement)
{
    unsigned beginRange = 0;
    unsigned endRange = elements.size() - 1;
    while (beginRange <= endRange)
    {
        unsigned mid = (beginRange + endRange) / 2;
        const T& candidate = elements[mid];
        ComparisonResult compRes = Functor()(candidate, searchElement);
        switch(compRes)
        {
            case ComparisonResult::Equal:
            {
                return &candidate;
            }
            case ComparisonResult::GreaterThan:
            {
                beginRange = mid + 1;
                continue;
            }
            case ComparisonResult::LessThan:
            {
                endRange = mid - 1;
                continue;
            }
        }
    }
    return nullptr;
}

template <typename T, typename E, class Functor>
[[nodiscard]] T* binarySearch(std::span<T> &elements, const E& searchElement)
{
    return const_cast<T*>(binarySearch<T, E, Functor>(const_cast<const std::span<T> &>(elements), searchElement));
}

[[nodiscard]] glm::vec3 inline __attribute__((always_inline)) aiVec3ToGlmVec3(const aiVector3D &ovec)
{
    return glm::vec3(ovec[0], ovec[1], ovec[2]);
}

template<glm::length_t L, typename T, glm::qualifier Q = glm::defaultp, size_t N>
[[nodiscard]] glm::vec<L, T, Q> getElementWiseMinVec3(const std::span<glm::vec<L, T, Q>, N> &vectors)
{
    constexpr T max = std::numeric_limits<T>::max();
    glm::vec<L, T, Q> minVec = glm::vec<L, T, Q>(max);
    for (const glm::vec<L, T, Q> &vec : vectors)
    {
        #pragma unroll
        for (int i = 0; i < L; i++)
        {
            minVec[i] = glm::min(minVec[i], vec[i]);
        }
    }
    return minVec;
}

template<glm::length_t L, typename T, glm::qualifier Q = glm::defaultp, size_t N>
[[nodiscard]] glm::vec<L, T, Q> getElementWiseMaxVec3(const std::span<glm::vec<L, T, Q>, N> &vectors)
{
    constexpr T lowest = std::numeric_limits<T>::lowest();
    glm::vec<L, T, Q> maxVec = glm::vec<L, T, Q>(lowest);
    for (const glm::vec<L, T, Q> &vec : vectors)
    {
        #pragma unroll
        for (int i = 0; i < L; i++)
        {
            maxVec[i] = glm::max(maxVec[i], vec[i]);
        }
    }
    return maxVec;
}

template<glm::length_t L, typename T, glm::qualifier Q = glm::defaultp>
[[nodiscard]] unsigned getMinElementIndex(const glm::vec<L, T, Q> &vec)
{
    T minElem = std::numeric_limits<T>::max();
    unsigned minIndex = std::numeric_limits<unsigned>::max();
    #pragma unroll
    for (int i = 0; i < L; i++)
    {
        if (vec[i] < minElem)
        {
            minElem = vec[i];
            minIndex = i;
        }
    }
    return minIndex;
}

template<glm::length_t L, typename T, glm::qualifier Q = glm::defaultp>
[[nodiscard]] unsigned getMaxElementIndex(const glm::vec<L, T, Q> &vec)
{
    T maxElem = std::numeric_limits<T>::min();
    unsigned maxIndex = std::numeric_limits<unsigned>::max();
    #pragma unroll
    for (int i = 0; i < L; i++)
    {
        if (vec[i] > maxElem)
        {
            maxElem = vec[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
