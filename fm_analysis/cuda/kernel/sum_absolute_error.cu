extern "C" {
    __global__ void sum_absolute_error(float* golden_fm, float* input_fm, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute absolute_difference
        float absolute_difference;
        if (gindex < N) {
            absolute_difference = abs(golden_fm[gindex] - input_fm[gindex]);
        } else {
            absolute_difference = .0f;
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = absolute_difference;
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