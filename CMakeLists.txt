cmake_minimum_required(VERSION 3.18)

project(EPSE LANGUAGES C CXX)

# Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Testing
enable_testing()

# Torch
find_package(Torch REQUIRED)

# =========================
# Embedder Library
# =========================

add_library(embedder
    parser/legacy/embedder.cpp
)

target_include_directories(embedder
    PUBLIC
    parser/legacy
    src
)

target_link_libraries(embedder
    PRIVATE
    ${TORCH_LIBRARIES}
)

# =========================
# Main Executable
# =========================

add_executable(epse_legacy
    parser/legacy/parse.c
)

target_include_directories(epse_legacy
    PRIVATE
    parser/legacy
    src
)

target_link_libraries(epse_legacy
    PRIVATE
    embedder
    ${TORCH_LIBRARIES}
)

target_compile_features(epse_legacy PRIVATE cxx_std_17)

# =========================
# Test Executable
# =========================

add_executable(test_parse
    tests/test_parse.c
)

target_include_directories(test_parse
    PRIVATE
    parser
    src
)

target_compile_definitions(test_parse
    PRIVATE
    UNIT_TEST
)

target_link_libraries(test_parse
    PRIVATE
    m
)

add_test(
    NAME parse_tests
    COMMAND test_parse
)
