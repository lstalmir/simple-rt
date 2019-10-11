cmake_minimum_required (VERSION 2.8)

project (simple-rt)

set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory (gtest)

# Build with C++17
set (CMAKE_CXX_STANDARD 17)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set (architecture x64)
else ()
    set (architecture x86)
endif ()

set (headers
    "Application.h"
    "ApplicationFactory.h"
    "ApplicationOMP.h"
    "ApplicationTest.h"
    "Arguments.h"
    "Fbx.h"
    "Plane.h"
    "Ray.h"
    "Vector.h"
    "tests/VectorTests.cpp"
    )

set (sources
    "Application.cpp"
    "ApplicationFactory.cpp"
    "ApplicationOMP.cpp"
    "ApplicationTest.cpp"
    "Arguments.cpp"
    "Main.cpp"
    "Ray.cpp"
    )

add_executable (simple-rt ${sources} ${headers})

# Compile with SSE
if (MSVC)
    #target_compile_options (simple-rt PUBLIC /arch:SSE3)
else ()
    target_compile_options (simple-rt PUBLIC -msse3)
endif ()

# Link gtest
target_link_libraries (simple-rt gtest_main)

target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/gtest/googletest/include
    PRIVATE ${CMAKE_SOURCE_DIR}/gtest/googlemock/include)

target_include_directories (simple-rt
    PRIVATE ${CMAKE_SOURCE_DIR}/fbxsdk/include)

if (UNIX)

    target_link_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/fbxsdk/lib/gcc/${architecture}/release)

    target_link_libraries (simple-rt libfbxsdk.a)

    add_custom_command (TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/fbxsdk/lib/gcc/${architecture}/release/libfbxsdk.so
                ${CMAKE_BINARY_DIR}/libfbxsdk.so)

elseif (WIN32)

    target_link_directories (simple-rt
        PRIVATE ${CMAKE_SOURCE_DIR}/fbxsdk/lib/vs2017/${architecture}/release)

    target_link_libraries (simple-rt libfbxsdk.lib)

    add_custom_command (TARGET simple-rt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/fbxsdk/lib/vs2017/${architecture}/release/libfbxsdk.dll
                ${CMAKE_BINARY_DIR}/libfbxsdk.dll)

endif ()
