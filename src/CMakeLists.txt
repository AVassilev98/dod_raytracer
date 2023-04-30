# add_subdirectory(shapes)
# add_subdirectory(utils)
include_directories(shapes)
include_directories(utils)


add_executable(dod_raytracer
main.cpp
shapes/sphere.cpp
shapes/plane.cpp
shapes/cylinder.cpp
shapes/triangle.cpp
shapes/mesh.cpp
)

target_link_libraries(dod_raytracer
glm
pthread
assimp
)