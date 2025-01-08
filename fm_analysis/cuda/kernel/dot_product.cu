extern "C" {
    __global__ void dot_product(float* golden_fm, float* input_fm, float* result, int N) {
        int batch_id = blockIdx.y;

        int lindex = threadIdx.x;
        int offset_batch = blockDim.x * blockIdx.x + lindex;

        int gindex = batch_id * N + offset_batch;

        // Compute the dot product
        float dot_product = .0f;
        if (offset_batch < N) {
           dot_product = golden_fm[gindex] * input_fm[gindex];
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = dot_product;
        __syncthreads();

        // Parallel sum for dot product
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (lindex < offset) {
                sharedData[lindex] += sharedData[lindex + offset];
            }
            __syncthreads();
        }

        // Add the results in each block
        if (lindex == 0) {
            atomicAdd(&result[batch_id], sharedData[0]);
        }
    }
}