// scan_benchmark.cu
// Inclusive prefix-sum (scan) on CPU (seq + OpenMP) and GPU (CUDA two-phase).
// Build: see commands below.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <omp.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do {                                         \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e)          \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
        std::exit(1);                                                  \
    }                                                                  \
} while(0)
#endif

// ============================================================
// 1) Sequential inclusive scan
// ============================================================
void scan_inclusive_seq(const int* in, int* out, size_t n) {
    if (n == 0) return;
    long long acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc += in[i];
        out[i] = (int)acc;
    }
}

// ============================================================
// 2) OpenMP inclusive scan (two-phase)
//    - split into chunks by thread
//    - each thread does local inclusive scan
//    - compute offsets of each chunk
//    - add offsets in parallel
// ============================================================
void scan_inclusive_omp(const int* in, int* out, size_t n) {
    if (n == 0) return;

    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::vector<long long> chunk_sums(num_threads, 0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int T   = omp_get_num_threads();

        size_t chunk = (n + T - 1) / T;
        size_t start = (size_t)tid * chunk;
        size_t end   = std::min(n, start + chunk);

        // local scan
        long long acc = 0;
        for (size_t i = start; i < end; i++) {
            acc += in[i];
            out[i] = (int)acc;
        }
        chunk_sums[tid] = acc;

#pragma omp barrier

        // compute offset for this chunk (prefix of chunk_sums)
        long long offset = 0;
        for (int k = 0; k < tid; k++) offset += chunk_sums[k];

        // add offset
        for (size_t i = start; i < end; i++) {
            out[i] = (int)((long long)out[i] + offset);
        }
    }
}

// ============================================================
// 3) CUDA two-phase scan:
//    Kernel A: per-block scan + write block sum
//    Then scan block sums recursively (on GPU)
//    Kernel B: add scanned block sums to each block (offset)
// ============================================================

static __device__ __forceinline__ int warpInclusiveScan(int val) {
    // warp-level inclusive scan using shfl
    // Assumes full warp participation
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}

// Block inclusive scan using warp scans + shared for warp sums.
// Works for blockDim.x up to 1024. Inclusive result written to out.
__global__ void block_scan_inclusive(const int* in, int* out, int* blockSums, int n) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int x = (gid < n) ? in[gid] : 0;

    // 1) warp inclusive scan
    int val = warpInclusiveScan(x);

    int lane = tid & 31;
    int warp = tid >> 5;

    // write last lane of each warp
    if (lane == 31) smem[warp] = val;
    __syncthreads();

    // 2) scan warp sums in warp 0
    int warpSum = 0;
    if (warp == 0) {
        int wval = (tid < (blockDim.x + 31) / 32) ? smem[lane] : 0;
        wval = warpInclusiveScan(wval);
        smem[lane] = wval; // inclusive scan of warp sums
    }
    __syncthreads();

    // 3) add offset of previous warps
    if (warp > 0) {
        warpSum = smem[warp - 1];
        val += warpSum;
    }

    if (gid < n) out[gid] = val;

    // block sum = last valid element in block (inclusive)
    // We store sum for the whole block (considering tail shorter block too)
    if (blockSums) {
        // last thread writes block sum; but tail blocks need correct last valid
        // easiest: thread0 writes using out[min(end,n)-1]
        __syncthreads();
        if (tid == 0) {
            int last = min((blockIdx.x + 1) * blockDim.x, n) - 1;
            int bsum = out[last];
            blockSums[blockIdx.x] = bsum;
        }
    }
}

// Add offsets to each block (excluding block 0)
__global__ void add_block_offsets(int* data, const int* scannedBlockSums, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    if (gid >= n) return;
    if (bid == 0) return; // first block has no offset
    int offset = scannedBlockSums[bid - 1]; // inclusive scan => offset = sum of all previous blocks
    data[gid] += offset;
}

// Helper: recursive scan of block sums on GPU (in-place/out-of-place)
void gpu_scan_recursive(const int* d_in, int* d_out, int n, int blockSize) {
    int numBlocks = (n + blockSize - 1) / blockSize;

    int* d_blockSums = nullptr;
    if (numBlocks > 1) {
        CHECK_CUDA(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));
    }

    size_t shmem = ((blockSize + 31) / 32) * sizeof(int); // warp sums
    block_scan_inclusive<<<numBlocks, blockSize, shmem>>>(d_in, d_out, d_blockSums, n);
    CHECK_CUDA(cudaGetLastError());

    if (numBlocks > 1) {
        // scan block sums -> scanned block sums
        int* d_scannedBlockSums = nullptr;
        CHECK_CUDA(cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(int)));

        gpu_scan_recursive(d_blockSums, d_scannedBlockSums, numBlocks, blockSize);

        // add offsets to each block of d_out
        add_block_offsets<<<numBlocks, blockSize>>>(d_out, d_scannedBlockSums, n);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaFree(d_scannedBlockSums));
        CHECK_CUDA(cudaFree(d_blockSums));
    }
}

// GPU inclusive scan wrapper: returns kernel+device work time (ms) and optionally total time
struct GpuTiming {
    float ms_kernels = 0.0f;
    float ms_total   = 0.0f;
};

GpuTiming scan_inclusive_cuda(const std::vector<int>& h_in, std::vector<int>& h_out, int blockSize) {
    const int n = (int)h_in.size();
    h_out.assign(n, 0);

    int *d_in=nullptr, *d_out=nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(int)));

    cudaEvent_t e0,e1,e2,e3;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2));
    CHECK_CUDA(cudaEventCreate(&e3));

    // total time includes H2D + kernels + D2H
    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(e1));

    // kernel time: recursive scan includes multiple kernel launches
    CHECK_CUDA(cudaEventRecord(e2));
    gpu_scan_recursive(d_in, d_out, n, blockSize);
    CHECK_CUDA(cudaEventRecord(e3));
    CHECK_CUDA(cudaEventSynchronize(e3));

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventSynchronize(e3)); // ensure kernels done before D2H already, but ok

    float ms_h2d=0, ms_k=0, ms_total=0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, e0, e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms_k,   e2, e3));

    // approximate total: e0 -> after D2H: easiest via extra sync+event (do quick)
    cudaEvent_t e4;
    CHECK_CUDA(cudaEventCreate(&e4));
    CHECK_CUDA(cudaEventRecord(e4));
    CHECK_CUDA(cudaEventSynchronize(e4));
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, e0, e4));
    CHECK_CUDA(cudaEventDestroy(e4));

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaEventDestroy(e2));
    CHECK_CUDA(cudaEventDestroy(e3));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return {ms_k, ms_total};
}

// ============================================================
// Utilities: timing + correctness
// ============================================================
double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

bool check_equal(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    // Sizes to test (можно поменять)
    std::vector<size_t> sizes = {
        10000,
        100000,
        1000000,
        10000000,
        20000000,
        50000000

    };

    // Random input
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 5);

    int blockSize = 256; // typical choice: 256 or 512

    std::cout << "Inclusive prefix-sum (scan) benchmark\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "CUDA blockSize: " << blockSize << "\n\n";

    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(12) << "Seq(ms)"
              << std::setw(12) << "OMP(ms)"
              << std::setw(14) << "GPUkern(ms)"
              << std::setw(14) << "GPUtot(ms)"
              << std::setw(10) << "OMP ok"
              << std::setw(10) << "GPU ok"
              << std::setw(12) << "Spd OMP"
              << std::setw(12) << "Spd GPU(tot)"
              << "\n";

    std::cout << std::string(110, '-') << "\n";

    for (size_t n : sizes) {
        std::vector<int> in(n);
        for (size_t i = 0; i < n; i++) in[i] = dist(rng);

        std::vector<int> out_seq(n), out_omp(n), out_gpu(n);

        // Seq
        double t0 = now_ms();
        scan_inclusive_seq(in.data(), out_seq.data(), n);
        double t1 = now_ms();
        double ms_seq = t1 - t0;

        // OMP
        double t2 = now_ms();
        scan_inclusive_omp(in.data(), out_omp.data(), n);
        double t3 = now_ms();
        double ms_omp = t3 - t2;

        bool ok_omp = check_equal(out_seq, out_omp);

        // GPU
        GpuTiming gt = scan_inclusive_cuda(in, out_gpu, blockSize);
        bool ok_gpu = check_equal(out_seq, out_gpu);

        double spd_omp = ms_seq / ms_omp;
        double spd_gpu = ms_seq / gt.ms_total;

        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(12) << std::fixed << std::setprecision(3) << ms_seq
                  << std::setw(12) << ms_omp
                  << std::setw(14) << gt.ms_kernels
                  << std::setw(14) << gt.ms_total
                  << std::setw(10) << (ok_omp ? "OK" : "FAIL")
                  << std::setw(10) << (ok_gpu ? "OK" : "FAIL")
                  << std::setw(12) << spd_omp
                  << std::setw(12) << spd_gpu
                  << "\n";

        // Если хотите сразу падать при ошибке:
        if (!ok_omp || !ok_gpu) {
            std::cerr << "\nCorrectness check failed at N=" << n << "\n";
            return 2;
        }
    }

    std::cout << "\nNotes:\n"
              << "- GPUkern(ms): только время кернелов (scan + add offsets + рекурсия)\n"
              << "- GPUtot(ms): H2D + kernels + D2H (часто именно это важно в задачах)\n";

    return 0;
}
