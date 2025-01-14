extern "C" {
    __global__ void logarithm10(float* input, int N, float constant = 1) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Perform the log10 for each inputs
        if (gindex < N) {
            input[gindex] = constant*log10(input[gindex]);
        }
    }
}