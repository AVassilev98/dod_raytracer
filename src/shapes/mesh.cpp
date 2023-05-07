#include "mesh.h"
#include "utils.h"
#include "triangle.h"
#include "assimp/Importer.hpp"
#include "assimp/mesh.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

void Mesh::Create(_Create &createStruct)
{
    const aiScene *assimp_mesh = m_importer.ReadFile(createStruct.loadPath.c_str(), 0
        | aiProcess_Triangulate 
        | aiProcess_JoinIdenticalVertices
        | aiProcess_GenSmoothNormals);

    Mesh::Attributes meshAttrs = {};
    if (!assimp_mesh || !assimp_mesh->HasMeshes())
    {
        printf("Missing or empty mesh after assimp loading!\n");
        return;
    }

    meshAttrs.color = {0.1, 0.8, 0.3};
    m_meshAttributes.push_back(meshAttrs);

    unsigned existingGlobalTriangleIdx = Triangle::m_numTriangles;
    Triangle::_Create triangleCreateStruct = {};
    for (int i = 0; i < assimp_mesh->mNumMeshes; i++)
    {
        const aiMesh &subMesh = *assimp_mesh->mMeshes[i];
        if (!subMesh.HasFaces())
        {
            continue;
        }

        for (int j = 0; j < subMesh.mNumFaces; j++)
        {
            const aiFace &face = subMesh.mFaces[j];
            triangleCreateStruct.A = aiVec3ToGlmVec3(subMesh.mVertices[face.mIndices[0]]);
            triangleCreateStruct.B = aiVec3ToGlmVec3(subMesh.mVertices[face.mIndices[1]]);
            triangleCreateStruct.C = aiVec3ToGlmVec3(subMesh.mVertices[face.mIndices[2]]);

            triangleCreateStruct.AN = aiVec3ToGlmVec3(subMesh.mNormals[face.mIndices[0]]);
            triangleCreateStruct.BN = aiVec3ToGlmVec3(subMesh.mNormals[face.mIndices[1]]);
            triangleCreateStruct.CN = aiVec3ToGlmVec3(subMesh.mNormals[face.mIndices[2]]);

            Triangle::create(triangleCreateStruct);
        }
    }
}