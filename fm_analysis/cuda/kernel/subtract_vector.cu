extern "C" {
    __global__ void subtract_vector(float* input1, float* input2, float* result, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Compute the subtraction
        if (gindex < N) {
           result[gindex] = input1[gindex] - input2[gindex];
        }
    }
}