extern "C" {
    __global__ void dot_product(float* golden_fm, float* input_fm, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the dot product
        float dot_product;
        if (gindex < N) {
           dot_product = golden_fm[gindex] * input_fm[gindex];
        } else {
            dot_product = .0f;
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
            atomicAdd(result, sharedData[0]);
        }
    }

    __global__ void euclidian_norm(float* fm, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the square
        float square;
        if (gindex < N) {
            square = fm[gindex] * fm[gindex];
        } else {
            square = .0f;
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = square;
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
            atomicAdd(result, sharedData[0]);
        }
    }
}