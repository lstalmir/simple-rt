cmake_minimum_required (VERSION 2.8)

project (simple-rt)

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
    "Arguments.h"
    )

set (sources
    "Application.cpp"
    "ApplicationFactory.cpp"
    "ApplicationOMP.cpp"
    "Arguments.cpp"
    "Main.cpp"
    )

add_executable (simple-rt ${sources} ${headers})

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
