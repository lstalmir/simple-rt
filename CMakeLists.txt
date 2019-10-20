cmake_minimum_required (VERSION 2.8)

project (simple-rt)

set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory (gtest)

add_subdirectory (zlib)

set (PNG_BUILD_ZLIB ON)
set (PNG_LINK_ZLIB_STATIC OFF)
set (SKIP_INSTALL_EXPORT ON)
set (ZLIB_INCLUDE_DIR ${CMAKE_BINARY_DIR}/zlib ${CMAKE_SOURCE_DIR}/zlib)
set (ZLIB_LIBRARY zlib)
add_subdirectory (libpng)

set (TBB_DIR ${CMAKE_SOURCE_DIR}/tbb2019_20191006oss/cmake)

find_package (TBB REQUIRED)
find_package (OpenCL REQUIRED)
find_package (OpenMP REQUIRED)

# Build with C++17
set (CMAKE_CXX_STANDARD 17)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set (architecture x64)
    set (bits 64)
    set (tbb_arch intel64)
else ()
    set (architecture x86)
    set (bits 32)
    set (tbb_arch ia32)
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
    "File.h"
    "Intrin.h"
    "Main.cpp"
    "Vec.h"
    )
    
set (ocl
    "ocl/Kernel.h"
    "ocl/oclRay.cpp"
    "ocl/oclRay.h"
    )

set (omp
    "omp/ompBox.h"
    "omp/ompCamera.h"
    "omp/ompObject.h"
    "omp/ompPlane.h"
    "omp/ompRay.h"
    "omp/ompScene.h"
    "omp/ompTriangle.h"
    )

set (omptests
    "omp/tests/ompCameraTests.cpp"
    "omp/tests/ompDrawTests.cpp"
    "omp/tests/ompIntersectTests.cpp"
    "omp/tests/ompReflectTests.cpp"
    "omp/tests/ompSceneTests.cpp"
    "omp/tests/ompVectorTests.cpp"
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

# TBB
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/tbb2019_20191006oss/include
    PRIVATE ${CMAKE_SOURCE_DIR}/pstl2019_20191006oss/include)

if (UNIX)

    target_link_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/tbb2019_20191006oss/lib/${tbb_arch}/gcc4.8)
        
    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tbb2019_20191006oss/lib/${tbb_arch}/gcc4.8/libtbb.so
                ${CMAKE_BINARY_DIR}/libtbb.so)

elseif (WIN32)

    target_link_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/tbb2019_20191006oss/lib/${tbb_arch}/vc14)
        
    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tbb2019_20191006oss/bin/${tbb_arch}/vc14/tbb.dll
                ${CMAKE_BINARY_DIR}/tbb.dll)

endif ()

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

# zlib
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/zlib
    PRIVATE ${CMAKE_BINARY_DIR}/zlib)

target_link_libraries (simple-rt zlib)

add_custom_command (
    TARGET simple-rt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:zlib>
            ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:zlib>)
        
# libpng
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/libpng
    PRIVATE ${CMAKE_BINARY_DIR}/libpng)

target_link_libraries (simple-rt png)

add_custom_command (
    TARGET simple-rt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:png>
            ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:png>)
        