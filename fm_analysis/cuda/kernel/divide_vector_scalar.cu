extern "C" {
    __global__ void divide_vector_scalar(float* input, int scalar, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Perform the division with the scalar for each inputs
        if (gindex < N) {
            input[gindex] = input[gindex]/scalar;
        }
    }
}