//
// Created by anthony on 6/28/24.
//
#include <iostream>
#include "../src/matrix_solve.h"
#include <cassert>

#include "../../../../../usr/local/cuda/targets/x86_64-linux/include/thrust/device_vector.h"
#include "../src/matsol_thrust.cuh"

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
    MatSol<3>::PrintVector(solution);
    MatSol<3>::PrintVector(correct_solution);
    assert(MatSol<3>::VectorEquals(solution, correct_solution));
}

void test_3x3_cuSolver() {
    thrust::device_vector<float> matrix(9);
    thrust::device_vector<float> vector(3);
    thrust::device_vector<float> correct_solution(3);
    thrust::device_vector<float> solution(3);

    matrix = {2, 1, 1, 3, 2, 3, 2, 1, 2};
    vector = {7, 16, 10};
    correct_solution = {1, 2, 3};
    MatSol_cuSolver<3>::solve(matrix, vector, solution);
    assert(thrust::equal(solution.begin(), solution.end(), correct_solution.begin()));

    matrix = {1, 2, 3, 2, 5, 3, 1, 0, 8};
    vector = {13, 18, 26};
    correct_solution = {2,1,3};
    MatSol_cuSolver<3>::solve(matrix, vector, solution);
    MatSol_cuSolver<3>::printVector(solution);
    MatSol_cuSolver<3>::printVector(correct_solution);
    assert(thrust::equal(solution.begin(), solution.end(), correct_solution.begin()));
}