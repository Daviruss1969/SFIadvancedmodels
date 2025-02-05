extern "C" {
    __global__ void entropy(float* input_fm, float* result, float* sum_values, int N) {
        int batch_id = blockIdx.y;

        int lindex = threadIdx.x;
        int offset_batch = blockDim.x * blockIdx.x + lindex;

        int gindex = batch_id * N + offset_batch;

        // Compute probability distribution
        float probability_distribution = .0f;
        if (offset_batch < N) {
            probability_distribution = input_fm[gindex] / sum_values[batch_id];
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = -(probability_distribution * log(probability_distribution + 1e-12)); // 1e-12 to avoid log 0
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
            atomicAdd(&result[batch_id], sharedData[0]);
        }
    }
}