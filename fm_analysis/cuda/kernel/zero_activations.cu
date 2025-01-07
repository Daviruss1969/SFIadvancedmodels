extern "C" {
    __global__ void zero_activations(float* fm, int* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        int value = 0;
        if (gindex < N && fm[gindex] == 0) {
            value = 1;
        }

        // Declare shared memory
        __shared__ int sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = value;
        __syncthreads();

        // Parallel reduction
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (lindex < offset) {
                sharedData[lindex] += sharedData[lindex + offset];
            }
            __syncthreads();
        }

        // Add the results in each block
        if (lindex == 0) {
            atomicAdd(result, sharedData[0]);
        }
    }
}