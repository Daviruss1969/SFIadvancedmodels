extern "C" {
    __global__ void covariance(float* golden_fm, float* input_fm, float* result, float golden_mean_value, float input_mean_value, int N) {
        int batch_id = blockIdx.y;

        int lindex = threadIdx.x;
        int offset_batch = blockDim.x * blockIdx.x + lindex;

        int gindex = batch_id * N + offset_batch;

        // Compute the product of the difference between one value and the mean value corresponding to a feature map
        float diff = .0f;
        if (offset_batch < N) {
            diff = (golden_fm[gindex] - golden_mean_value) * (input_fm[gindex] - input_mean_value);
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = diff;
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