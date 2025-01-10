extern "C" {
    __global__ void structural_similarity_index(float* golden_means, float* faulty_means, float* golden_vars, float* faulty_vars, float* covariances, float* result, float C1, float C2, int N) {
        int lindex = threadIdx.x;
        int gindex = blockDim.x * blockIdx.x + lindex;

        // Perform the division for each inputs
        if (gindex < N) {
            result[gindex] = ((2*golden_means[gindex]*faulty_means[gindex] + C1)*(2*covariances[gindex] + C2))/((pow(golden_means[gindex], 2)*pow(faulty_means[gindex], 2) + C1)*(golden_vars[gindex]*faulty_vars[gindex] + C2));
        }
    }
}