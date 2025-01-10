extern "C" {
    __global__ void power(float* input, float p, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Perform the power for each inputs
        if (gindex < N) {
            input[gindex] = pow(input[gindex], p);
        }
    }
}