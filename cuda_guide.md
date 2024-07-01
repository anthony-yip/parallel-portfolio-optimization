CUDA kernel calls are asynchronous (including cuBLAS). So sometimes the error location
that gets printed is not the actual location of the error. To get the actual location of the error,
try placing cudaDeviceSynchronize() after each kernel call. The error will then occur within 
this call, and you can use cudaGetLastError() to get the actual error location.
