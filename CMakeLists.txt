cmake_minimum_required(VERSION 3.29)
project(parallel_portfolio_optimization LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_executable(parallel_portfolio_optimization
        src/main.cpp
        src/mvo.h
        src/mvo.cpp
        src/constants.h
        src/utils.h
        tests/matrix_solve_test.cpp
        src/matrix_solve.h
        src/mvo_thrust.cu
        src/mvo_thrust.cuh)
target_link_libraries(parallel_portfolio_optimization ${CUDA_LIBRARIES})