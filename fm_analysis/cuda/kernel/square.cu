extern "C" {
    __global__ void square(float* input, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Perform the square for each inputs
        if (gindex < N) {
            input[gindex] = sqrt(input[gindex]);
        }
    }
}