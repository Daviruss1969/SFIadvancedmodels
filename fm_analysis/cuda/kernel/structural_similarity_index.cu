extern "C" {
    __global__ void mean_value(float* fm, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;


        float fmi;
        if (gindex < N) {
            fmi = fm[gindex];
        } else {
            fmi = .0f;
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = fmi;
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

    __global__ void variance_value(float* fm, float* result, float mean_value, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the square of the difference between one value and the mean value
        float diff;
        if (gindex < N) {
            diff = (fm[gindex] - mean_value) * (fm[gindex] - mean_value);
        } else {
            diff = .0f;
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

    __global__ void covariance_value(float* golden_fm, float* input_fm, float* result, float golden_mean_value, float input_mean_value, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the product of the difference between one value and the mean value corresponding to a feature map
        float diff;
        if (gindex < N) {
            diff = (golden_fm[gindex] - golden_mean_value) * (input_fm[gindex] - input_mean_value);
        } else {
            diff = .0f;
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