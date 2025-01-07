extern "C" {
    __global__ void variance(float* fm, float* result, float mean_value, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the square of the difference between one value and the mean value
        float value = .0f;
        if (gindex < N) {
            float diff = fm[gindex] - mean_value;
            value = diff*diff;
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

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