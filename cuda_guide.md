CUDA kernel calls are asynchronous (including cuBLAS). So sometimes the error location
that gets printed is not the actual location of the error. To get the actual location of the error,
try placing cudaDeviceSynchronize() after each kernel call. The error will then occur within 
this call, and you can use cudaGetLastError() to get the actual error location.

CUDA calls on the same stream are executed in order, though. So cudaMemcpy will still wait for
the previous kernel to complete before attempting to copy. Additionally,
cudaMemcpy and cudaMalloc are both synchronous. They will block until their operation is complete.

Thus, call cudaDeviceSynchronize() whenever the host depends on the completion of some
device operation.

Inspecting device memory with cuda-gdb using @global doesn't always work. more full-proof is 
to copy to host memory and inspect there.
