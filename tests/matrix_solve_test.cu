//
// Created by anthony on 6/28/24.
//
#include <iostream>
#include "../src/matrix_solve.h"
#include <cassert>
#include "../src/matsol_thrust.cuh"
#include <thrust/for_each.h>

void test_3x3() {
    MatSol<3>::Matrix matrix; MatSol<3>::Vector vector; MatSol<3>::Vector correct_solution;
    MatSol<3>::Vector solution;
    matrix = {2, 1, 1, 3, 2, 3, 2, 1, 2};
    vector = {7, 16, 10};
    correct_solution = {1, 2, 3};
    solution = MatSol<3>::solve(matrix, vector);
    assert(MatSol<3>::VectorEquals(solution, correct_solution));

    matrix = {1, 2, 3, 2, 5, 3, 1, 0, 8};
    vector = {13, 18, 26};
    correct_solution = {2,1,3};
    solution = MatSol<3>::solve(matrix, vector);
    assert(MatSol<3>::VectorEquals(solution, correct_solution));
}

namespace
{
    struct Round : public thrust::unary_function<float, void>
    {
        __device__ void operator()(float& x) const {
            x = (float) round(x);
        }
    };
}
void test_3x3_cuSolver() {
    thrust::device_vector<float> matrix(9);
    thrust::device_vector<float> vector(3);
    thrust::device_vector<float> correct_solution(3);
    thrust::device_vector<float> solution(3);

    matrix = {2, 1, 1, 3, 2, 3, 2, 1, 2};
    vector = {7, 16, 10};
    correct_solution = {1, 2, 3};
    cuMatSol<3>::solve(matrix, vector, solution);
    thrust::for_each(solution.begin(), solution.end(), Round {});
    assert(thrust::equal(solution.begin(), solution.end(), correct_solution.begin()));

    matrix = {1, 2, 3, 2, 5, 3, 1, 0, 8};
    vector = {13, 18, 26};
    correct_solution = {2,1,3};
    cuMatSol<3>::solve(matrix, vector, solution);
    thrust::for_each(solution.begin(), solution.end(), Round {});
    assert(thrust::equal(solution.begin(), solution.end(), correct_solution.begin()));
}
