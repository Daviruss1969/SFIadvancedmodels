extern "C" {
    __global__ void divide(float* input_1, float* input_2, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Perform the division for each inputs
        if (gindex < N) {
            result[gindex] = input_1[gindex] / input_2[gindex];
        }
    }
}