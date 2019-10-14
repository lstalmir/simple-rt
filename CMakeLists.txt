cmake_minimum_required (VERSION 2.8)

project (simple-rt)

set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory (gtest)

find_package (OpenCL REQUIRED)
find_package (OpenMP REQUIRED)

# Build with C++17
set (CMAKE_CXX_STANDARD 17)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set (architecture x64)
    set (bits 64)
else ()
    set (architecture x86)
    set (bits 32)
endif ()

set (base
    "Application.cpp"
    "Application.h"
    "ApplicationFactory.cpp"
    "ApplicationFactory.h"
    "ApplicationOCL.cpp"
    "ApplicationOCL.h"
    "ApplicationOMP.cpp"
    "ApplicationOMP.h"
    "ApplicationTest.cpp"
    "ApplicationTest.h"
    "Arguments.cpp"
    "Arguments.h"
    "Fbx.h"
    "Main.cpp"
    )
    
set (ocl
    "ocl/Kernel.h"
    "ocl/oclRay.cpp"
    "ocl/oclRay.h"
    )

set (omp
    "omp/Plane.cpp"
    "omp/Plane.h"
    "omp/Ray.h"
    "omp/SSE.h"
    "omp/Vector.h"
    )

set (omptests
    "omp/tests/ompVectorTests.cpp"
    "omp/tests/ompRayTests.cpp"
    )

add_executable (simple-rt ${base} ${ocl} ${omp} ${omptests})

# Compile with SSE
if (MSVC)
    #target_compile_options (simple-rt PUBLIC /arch:SSE3)
else ()
    target_compile_options (simple-rt PUBLIC -msse3)
endif ()

# gtest
target_link_libraries (simple-rt gtest_main)

target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/gtest/googletest/include
    PRIVATE ${CMAKE_SOURCE_DIR}/gtest/googlemock/include)

# OpenCL
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/opencl-headers
    PRIVATE ${CMAKE_SOURCE_DIR}/opencl-clhpp/include)

target_link_libraries (simple-rt ${OpenCL_LIBRARY})

# OpenMP
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    
# FBXSDK
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/fbxsdk/include)

if (UNIX)

    target_link_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/fbxsdk/lib/gcc/${architecture}/release)

    target_link_libraries (simple-rt libfbxsdk.a)

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/fbxsdk/lib/gcc/${architecture}/release/libfbxsdk.so
                ${CMAKE_BINARY_DIR}/libfbxsdk.so)

elseif (WIN32)

    target_link_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/fbxsdk/lib/vs2017/${architecture}/release)

    target_link_libraries (simple-rt libfbxsdk.lib)

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/fbxsdk/lib/vs2017/${architecture}/release/libfbxsdk.dll
                ${CMAKE_BINARY_DIR}/libfbxsdk.dll)

endif ()
