extern "C" {
    __device__ static float atomicMin(float* address, float val) {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    __global__ void minimum(float* fm, float* result, int N) {
        int batch_id = blockIdx.y;

        int lindex = threadIdx.x;
        int offset_batch = blockDim.x * blockIdx.x + lindex;

        int gindex = batch_id * N + offset_batch;

        float value = .0f;
        if (offset_batch < N) {
            value = fm[gindex];
        }

        // Declare shared memory
        __shared__ float sharedData[1024];

        // Put data into shared memory
        sharedData[lindex] = value;
        __syncthreads();

        // Parallel reduction
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (lindex < offset) {
                sharedData[lindex] = sharedData[lindex] <= sharedData[lindex + offset] ? sharedData[lindex] : sharedData[lindex + offset];
            }
            __syncthreads();
        }

        // Add the results in each block
        if (lindex == 0) {
            atomicMin(&result[batch_id], sharedData[0]);
        }
    }
}