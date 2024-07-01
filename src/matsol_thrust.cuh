//
// Created by anthony on 6/30/24.
//

#pragma once
#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <iostream>
// #include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


/// Given a index into a matrix in row-major order, return the corresponding index in column-major order.
/// This only works on square matrices.
struct RowToColumnMajor : public thrust::unary_function<int, int>
{
    int num_rows;

    explicit RowToColumnMajor(int num_rows) : num_rows(num_rows)
    {
    }

    int operator()(int i)
    {
        int row = i / num_rows;
        int col = i % num_rows;
        return col * num_rows + row;
    }
};

#define CUDA_ERRCHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << #call \
                      << " at line " << __LINE__ \
                      << " in file " << __FILE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define CUBLAS_ERRCHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << #call \
                      << " at line " << __LINE__ \
                      << " in file " << __FILE__ \
                      << ": " << cublasGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function to convert cublasStatus_t to string
inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN";
    }
}

template<int N>
class cuMatSol
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

        // Prepare arguments for both rf and rs
        cublasHandle_t cublas_handle;
        CUBLAS_ERRCHECK(cublasCreate(&cublas_handle));
        float ** a_batch; CUDA_ERRCHECK(cudaMalloc(&a_batch, sizeof(float*)));
        CUDA_ERRCHECK(cudaMemcpy(a_batch, &raw_A, sizeof(float*), cudaMemcpyDefault));
        float ** b_batch; CUDA_ERRCHECK(cudaMalloc(&b_batch, sizeof(float*)));
        CUDA_ERRCHECK(cudaMemcpy(b_batch, &raw_b, sizeof(float*), cudaMemcpyDefault));
        int * permutation_array; CUDA_ERRCHECK(cudaMalloc(&permutation_array, N * sizeof(int)));
        int * info_array_lu; CUDA_ERRCHECK(cudaMalloc(&info_array_lu, sizeof(int)));

#ifdef DEBUG
        float ** h_a_batch_debug = (float**) malloc(sizeof(float*));
        CUDA_ERRCHECK(cudaMemcpy(h_a_batch_debug, a_batch, sizeof(float*), cudaMemcpyDefault));
        float ** h_b_batch_debug = (float**) malloc(sizeof(float*));
        CUDA_ERRCHECK(cudaMemcpy(h_b_batch_debug, b_batch, sizeof(float*), cudaMemcpyDefault));
        free(h_a_batch_debug), free(h_b_batch_debug);
#endif


        // Do LU decomposition
        CUBLAS_ERRCHECK(cublasSgetrfBatched(cublas_handle, N, a_batch, N, permutation_array, info_array_lu, 1));

#ifdef DEBUG
        int * h_info_array_lu = (int*) malloc(sizeof(int));
        CUDA_ERRCHECK(cudaMemcpy(h_info_array_lu, info_array_lu, sizeof(int), cudaMemcpyDefault));
        if (h_info_array_lu[0] != 0) {
            std::cerr << "Error: LU decomposition failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        free(h_into_array_lu);
#endif

        int *info_array_solve = (int*) malloc(sizeof(int));
        // Solve the LU system
        /// CUBLAS_OP_T (transpose) is used because cuBLAS uses column-major order, whereas we use row-major order.
        CUBLAS_ERRCHECK(cublasSgetrsBatched(cublas_handle, CUBLAS_OP_T, N, 1, a_batch, N, permutation_array, b_batch, N, info_array_solve, 1));

#ifdef DEBUG
        if (info_array_solve[0] != 0)
        {
            std::cerr << "Error: Solving the system of linear equations failed" << std::endl;
        }
#endif

        // Copy the solution to the output vector
        CUDA_ERRCHECK(cudaMemcpy(raw_x, raw_b, N * sizeof(float), cudaMemcpyDefault));

        // Free resources
        CUDA_ERRCHECK(cudaFree(a_batch));
        CUDA_ERRCHECK(cudaFree(b_batch));
        CUDA_ERRCHECK(cudaFree(permutation_array));
        CUDA_ERRCHECK(cudaFree(info_array_lu));
        free(info_array_solve);
        CUBLAS_ERRCHECK(cublasDestroy(cublas_handle));
    }
};
