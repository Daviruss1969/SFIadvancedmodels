extern "C" {
    __global__ void subtract_vector(float* golden_fm, float* input_fm, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the subtraction
        if (gindex < N) {
           result[gindex] = golden_fm[gindex] - input_fm[gindex];
        }
    }
}