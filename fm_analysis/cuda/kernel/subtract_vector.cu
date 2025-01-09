extern "C" {
    __global__ void subtract_vector(float* golden_fm, float* input_fm, float* result, int N) {
        int batch_id = blockIdx.y;

        int lindex = threadIdx.x;
        int offset_batch = blockDim.x * blockIdx.x + lindex;

        int gindex = batch_id * N + offset_batch;

        // Compute the subtraction
        if (offset_batch < N) {
           result[gindex] = golden_fm[gindex] - input_fm[gindex];
        }
    }
}