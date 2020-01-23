cmake_minimum_required (VERSION 3.13)

project (simple-rt CXX CUDA)

set_property (GLOBAL PROPERTY USE_FOLDERS ON)

set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory (third-party/gtest)
add_subdirectory (third-party/zlib)

set (PNG_BUILD_ZLIB ON)
set (PNG_LINK_ZLIB_STATIC OFF)
set (SKIP_INSTALL_EXPORT ON)
set (ZLIB_INCLUDE_DIR ${CMAKE_BINARY_DIR}/third-party/zlib ${CMAKE_SOURCE_DIR}/third-party/zlib)
set (ZLIB_LIBRARY zlib)
add_subdirectory (third-party/libpng)

set (TBB_VER 2019_20191006oss)
set (TBB_DIR ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/cmake)

find_package (TBB REQUIRED)
find_package (CUDA REQUIRED)
find_package (OpenMP REQUIRED)

# Build with C++17
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_SEPARABLE_COMPILATION ON)

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
    "ApplicationTest.cpp"
    "ApplicationTest.h"
    "Arguments.cpp"
    "Arguments.h"
    "Fbx.h"
    "File.h"
    "Intrin.h"
    "Main.cpp"
    "Optimizations.h"
    "Scene.h"
    "Tbb.h"
    "Vec.h"
    )
    
set (cuda
    "cuda/cudaApplication.h"
    "cuda/cudaApplication.cu"
    "cuda/cudaCamera.h"
    "cuda/cudaCamera.cu"
    "cuda/cudaCommon.h"
    "cuda/cudaLight.h"
    "cuda/cudaLight.cu"
    "cuda/cudaLock.h"
    "cuda/cudaMemory.h"
    "cuda/cudaObject.h"
    "cuda/cudaObject.cu"
    "cuda/cudaRay.h"
    "cuda/cudaRay.cu"
    "cuda/cudaScene.h"
    "cuda/cudaTriangle.h"
    )

set (cudatests
    "cuda/tests/cudaArrayTests.cu"
    "cuda/tests/cudaCameraTests.cpp"
    "cuda/tests/cudaDispatchTests.cpp"
    "cuda/tests/cudaLightTests.cpp"
    )
    
set (omp
    "omp/ompApplication.h"
    "omp/ompBox.h"
    "omp/ompCamera.h"
    "omp/ompLight.h"
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

add_executable (simple-rt ${base}
    ${cuda} ${cudatests}
    ${omp} ${omptests})

# Compile with SSE and optimizations
if (MSVC)
else ()
    target_compile_options (simple-rt PUBLIC -msse3)            # Enable SSE3
    target_compile_options (simple-rt PUBLIC -mfma)             # Enable FMA
endif ()


# SSE extension for linux
if (UNIX)
    target_include_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/third-party/sse_mathfun_ext)
endif ()

# gtest
target_link_libraries (simple-rt gtest_main)

target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/gtest/googletest/include
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/gtest/googlemock/include)
    
# TBB
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/pstl${TBB_VER}/include)

if (UNIX)

    target_link_directories (
        simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/lib/${tbb_arch}/gcc4.8)

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/lib/${tbb_arch}/gcc4.8/libtbb.so
                ${CMAKE_BINARY_DIR}/libtbb.so)
                
    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/lib/${tbb_arch}/gcc4.8/libtbb_debug.so
                ${CMAKE_BINARY_DIR}/libtbb_debug.so)

elseif (WIN32)

    target_link_directories (
        simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/lib/${tbb_arch}/vc14)

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/bin/${tbb_arch}/vc14/tbb.dll
                ${CMAKE_BINARY_DIR}/tbb.dll)
                
    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/third-party/tbb${TBB_VER}/bin/${tbb_arch}/vc14/tbb_debug.dll
                ${CMAKE_BINARY_DIR}/tbb_debug.dll)

endif ()

# CUDA
target_include_directories (simple-rt
    PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries (simple-rt ${CUDA_LIBRARIES})

# OpenMP
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# FBXSDK
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/include)

if (UNIX)

    link_directories (
                ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/lib/gcc/${architecture}/release)

    target_link_libraries (simple-rt
                ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/lib/gcc/${architecture}/release/libfbxsdk.a)

    target_link_libraries (simple-rt
                ${CMAKE_DL_LIBS})

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/lib/gcc/${architecture}/release/libfbxsdk.so
                ${CMAKE_BINARY_DIR}/libfbxsdk.so)

elseif (WIN32)

    link_directories (
                ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/lib/vs2017/${architecture}/release)

    target_link_libraries (simple-rt
                ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/lib/vs2017/${architecture}/release/libfbxsdk.lib)

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/third-party/fbxsdk/lib/vs2017/${architecture}/release/libfbxsdk.dll
                ${CMAKE_BINARY_DIR}/libfbxsdk.dll)

endif ()

# LibXML2
if (UNIX)

    add_custom_command (
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third-party/libxml2
        OUTPUT ./Makefile
        COMMAND ./autogen.sh --prefix=${CMAKE_BINARY_DIR}/third-party/libxml2 --without-python)

    add_custom_command (
        TARGET simple-rt PRE_BUILD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third-party/libxml2
        DEPENDS ./Makefile
        COMMAND make -j
        COMMAND make install)

    target_include_directories (simple-rt
        PRIVATE ${CMAKE_BINARY_DIR}/third-party/libxml2/include)

    link_directories (
                ${CMAKE_SOURCE_DIR}/third-party/libxml2/lib)

    target_link_libraries (simple-rt
                ${CMAKE_BINARY_DIR}/third-party/libxml2/lib/libxml2.a)

    add_custom_command (
        TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/third-party/libxml2/lib/libxml2.so
                ${CMAKE_BINARY_DIR}/libxml2.so)

endif ()
    
# zlib
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/zlib
    PRIVATE ${CMAKE_BINARY_DIR}/third-party/zlib)

target_link_libraries (simple-rt zlib)

add_custom_command (
    TARGET simple-rt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:zlib>
            ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:zlib>)
        
# libpng
target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/libpng
    PRIVATE ${CMAKE_BINARY_DIR}/third-party/libpng)

target_link_libraries (simple-rt png)

add_custom_command (
    TARGET simple-rt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:png>
            ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:png>)
