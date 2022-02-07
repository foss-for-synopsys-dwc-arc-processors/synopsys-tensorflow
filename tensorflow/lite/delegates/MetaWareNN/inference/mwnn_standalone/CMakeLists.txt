cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(project_name "mwnn")
project("${project_name}")

file(GLOB_RECURSE metawarenn_lib_srcs
    metawarenn_lib/*.cc
    )
add_executable(${project_name} main.cpp ${metawarenn_lib_srcs})

target_link_libraries(${project_name} $ENV{PROTOBUF_PATH})
