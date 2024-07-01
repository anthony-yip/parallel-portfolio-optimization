//
// Created by anthony on 6/30/24.
//
// #include "matsol_thrust.cuh"
// #include <cusolverDn.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
//
// namespace
// {
//     /// Given a index into a matrix in row-major order, return the corresponding index in column-major order.
//     /// This only works on square matrices.
//     struct RowToColumnMajor : public thrust::unary_function<int, int>
//     {
//         int num_rows;
//
//         explicit RowToColumnMajor(int num_rows) : num_rows(num_rows)
//         {
//         }
//
//         int operator()(int i)
//         {
//             int row = i / num_rows;
//             int col = i % num_rows;
//             return col * num_rows + row;
//         }
//     };
// }
//
// #define CUDA_ERRCHECK(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             std::cerr << "CUDA error in " << #call \
//                       << " at line " << __LINE__ \
//                       << " in file " << __FILE__ \
//                       << ": " << cudaGetErrorString(err) << std::endl; \
//             exit(EXIT_FAILURE); \
//         } \
//     } while (0)
//
//
// #define CUBLAS_ERRCHECK(call) \
//     do { \
//         cublasStatus_t status = call; \
//         if (status != CUBLAS_STATUS_SUCCESS) { \
//             std::cerr << "cuBLAS error in " << #call \
//                       << " at line " << __LINE__ \
//                       << " in file " << __FILE__ \
//                       << ": " << cublasGetErrorString(status) << std::endl; \
//             exit(EXIT_FAILURE); \
//         } \
//     } while (0)
//
// // Function to convert cublasStatus_t to string
// static const char* cublasGetErrorString(cublasStatus_t status) {
//     switch (status) {
//         case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
//         case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
//         case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
//         case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
//         case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
//         case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
//         case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
//         case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
//         case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
//         case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
//         default: return "UNKNOWN";
//     }
// }
//
// template<int N>
// __inline__ void MatSol_cuSolver<N>::solve(thrust::device_vector<float>& A, thrust::device_vector<float>& b, thrust::device_vector<float>& x)
// {
//         thrust::device_vector<float> A_transposed (N * N);
//         auto permutation_it = thrust::make_permutation_iterator(A.begin(), thrust::make_transform_iterator(thrust::make_counting_iterator(0), RowToColumnMajor(N)));
//         thrust::copy(permutation_it, permutation_it + N * N, A_transposed.begin());
//
//         // Convert device_vector to raw pointers
//         float* raw_A = thrust::raw_pointer_cast(A_transposed.data());
//         float* raw_b = thrust::raw_pointer_cast(b.data());
//         float* raw_x = thrust::raw_pointer_cast(x.data());
//
//         cublasHandle_t cublas_handle;
//         CUBLAS_ERRCHECK(cublasCreate(&cublas_handle));
//
//         // Do LU decomposition
//         float ** a_batch;
//         CUDA_ERRCHECK(cudaMalloc(&a_batch, sizeof(float*)));
//         CUDA_ERRCHECK(cudaMemcpy(a_batch, &raw_A, sizeof(float*), cudaMemcpyHostToDevice));
//         int * permutation_array; CUDA_ERRCHECK(cudaMalloc(&permutation_array, N * sizeof(int)));
//         CUBLAS_ERRCHECK(cublasSgetrfBatched(cublas_handle, N, a_batch, N, permutation_array, NULL, 1));
//
//         // Solve the LU system
//         float ** b_batch;
//         CUDA_ERRCHECK(cudaMalloc(&b_batch, sizeof(float*)));
//         CUDA_ERRCHECK(cudaMemcpy(b_batch, &raw_b, sizeof(float*), cudaMemcpyHostToDevice));
//         CUBLAS_ERRCHECK(cublasSgetrsBatched(cublas_handle, CUBLAS_OP_N, N, 1, a_batch, N, permutation_array, b_batch, N, NULL, 1));
//
//         // Copy the solution to the output vector
//         CUDA_ERRCHECK(cudaMemcpy(raw_x, raw_b, N * sizeof(float), cudaMemcpyDeviceToDevice));
//
//         // Free resources
//         CUDA_ERRCHECK(cudaFree(a_batch));
//         CUDA_ERRCHECK(cudaFree(b_batch));
//         CUDA_ERRCHECK(cudaFree(permutation_array));
//         CUBLAS_ERRCHECK(cublasDestroy(cublas_handle));
//
//
//         // // Initialize cuSolver
//         // cusolverDnHandle_t handle = NULL;
//         // cusolverDnCreate(&handle);
//         //
//         // // Create a buffer for cuSolver operations
//         // int work_size = 0;
//         // cusolverDnSgetrf_bufferSize(handle, N, N, raw_A, N, &work_size);
//         //
//         // // Allocate the buffer
//         // float* work;
//         // cudaMalloc(&work, work_size * sizeof(float));
//         //
//         // // Compute the LU factorization of the matrix A
//         // int* devIpiv;
//         // cudaMalloc(&devIpiv, N * sizeof(int));
//         // int* devInfo;
//         // cudaMalloc(&devInfo, sizeof(int));
//         // cusolverDnSgetrf(handle, N, N, raw_A, N, work, devIpiv, &devInfo);
//         //
//         // // Solve the system of linear equations
//         // cusolverDnSgetrs(handle, CUBLAS_OP_N, N, 1, raw_A, N, devIpiv, raw_b, N, &devInfo);
//         //
//         // // Copy the solution to the output vector
//         // cudaMemcpy(raw_x, raw_b, N * sizeof(float), cudaMemcpyDeviceToDevice);
//
//         // Free the cuSolver handle and the buffer
//         // cusolverDnDestroy(handle);
//         // cudaFree(work);
//         // cudaFree(devIpiv);
//         // cudaFree(devInfo);
//     }
//
