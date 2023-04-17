  cmake_minimum_required(VERSION 3.5)
  include(FetchContent)
  set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_CXX_COMPILER g++-10)
  add_compile_options(-march=native)

  project(
    DodRaytracer
    VERSION 1.0
    LANGUAGES CXX)

  find_package(glm QUIET)

  if(NOT glm_FOUND)
    FetchContent_Declare(
      glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e
    )
  
    FetchContent_GetProperties(glm)
  
    if(NOT glm_POPULATED)
      FetchContent_Populate(glm)
      add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
    endif()
  endif()

  add_executable(dod_raytracer
      main.cpp
      sphere.cpp
      plane.cpp
      cylinder.cpp
      triangle.cpp
  )

  target_link_libraries(dod_raytracer
    glm
    pthread
  )