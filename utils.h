#pragma once
#include <span>
#include <functional>
#include "assimp/vector3.h"
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