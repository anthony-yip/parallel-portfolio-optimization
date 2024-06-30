//
// Created by anthony on 6/30/24.
//

#pragma once
#include <thrust/device_vector.h>
#include <cusolverDn.h>

template<int N>
class MatSol
{
    virtual void solve(thrust::device_vector<float>& A, thrust::device_vector<float>& b, thrust::device_vector<float>& x) = 0;
};


template<int N>
class MatSol_cuSolver
{
public:
    /**
     * Solves a system of linear equations Ax = b using cuSolver
     * @param[in] A
     * @param[in] b
     * @param[out] x
     */
    static void solve(thrust::device_vector<float>& A, thrust::device_vector<float>& b, thrust::device_vector<float>& x)
    {
        // Convert device_vector to raw pointers
        float* raw_A = thrust::raw_pointer_cast(A.data());
        float* raw_b = thrust::raw_pointer_cast(b.data());
        float* raw_x = thrust::raw_pointer_cast(x.data());

        // Initialize cuSolver
        cusolverDnHandle_t handle = NULL;
        cusolverDnCreate(&handle);

        // Create a buffer for cuSolver operations
        int work_size = 0;
        cusolverDnSgetrf_bufferSize(handle, N, N, raw_A, N, &work_size);

        // Allocate the buffer
        float* work;
        cudaMalloc(&work, work_size * sizeof(float));

        // Compute the LU factorization of the matrix A
        int* devIpiv;
        cudaMalloc(&devIpiv, N * sizeof(int));
        int* devInfo;
        cudaMalloc(&devInfo, sizeof(int));
        cusolverDnSgetrf(handle, N, N, raw_A, N, work, devIpiv, &devInfo);

        // Solve the system of linear equations
        cusolverDnSgetrs(handle, CUBLAS_OP_N, N, 1, raw_A, N, devIpiv, raw_b, N, &devInfo);

        // Copy the solution to the output vector
        cudaMemcpy(raw_x, raw_b, N * sizeof(float), cudaMemcpyDeviceToDevice);

        // Free the cuSolver handle and the buffer
        cusolverDnDestroy(handle);
        cudaFree(work);
        cudaFree(devIpiv);
        cudaFree(devInfo);
    }


};