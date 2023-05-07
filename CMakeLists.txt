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

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.ini
    ${CMAKE_CURRENT_BINARY_DIR}/config.ini
    COPYONLY
  )

  set(BUILD_SHARED_LIBS FALSE CACHE BOOL "x" FORCE)
  set(ASSIMP_NO_EXPORT TRUE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_TESTS FALSE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE CACHE BOOL "x" FORCE)
  set(ASSIMP_INSTALL_PDB FALSE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_ZLIB TRUE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_ASSIMP_TOOLS FALSE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_OBJ_IMPORTER TRUE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_PLY_IMPORTER TRUE CACHE BOOL "x" FORCE)
  set(ASSIMP_BUILD_BLEND_IMPORTER TRUE CACHE BOOL "x" FORCE)
  find_package(assimp QUIET)

  if(NOT assimp_FOUND)
    FetchContent_Declare(
      assimp
      GIT_REPOSITORY https://github.com/assimp/assimp.git
    )

    FetchContent_GetProperties(assimp)

    if(NOT assimp_POPULATED)
      FetchContent_Populate(assimp)
      add_subdirectory(${assimp_SOURCE_DIR} ${assimp_BINARY_DIR})
    endif()

    include_directories(${assimp_SOURCE_DIR}/include)
    include_directories(${assimp_BINARY_DIR}/include)
  endif()

  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  add_subdirectory(assets)
  add_subdirectory(src)