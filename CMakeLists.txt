cmake_minimum_required(VERSION 3.18)

project(EPSE LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_library(embedder
    parser/embedder.cpp
)

target_link_libraries(embedder PRIVATE ${TORCH_LIBRARIES})

add_executable(epse
    parser/parse.c
)

target_link_libraries(epse PRIVATE embedder)
target_include_directories(epse
    PRIVATE
    parser
)

target_link_libraries(epse
    PRIVATE
    ${TORCH_LIBRARIES}
)

target_compile_features(epse PRIVATE cxx_std_17)
