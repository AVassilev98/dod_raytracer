#pragma once

#include "base_shape.h"
#include <vector>
#include <string>
#include "assimp/Importer.hpp"

class Mesh : public BaseShape<Mesh>
{
    friend BaseShape<Mesh>;
    friend class Triangle;
    public:
        struct Attributes
        {
            glm::vec3 color;
        };

        struct _Create
        {
            const std::string &loadPath;
        };

    public:
        static void Create(_Create &createStruct);
    
    private:
        Mesh();

        private:
        static inline Assimp::Importer m_importer = Assimp::Importer();
        static inline std::vector<Attributes> m_meshAttributes = {};
};