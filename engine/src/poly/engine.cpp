// src/poly/engine.cpp - see the header.  What every architecture shares: the files, the weights' way to VRAM, the
// routed experts (the cache, the CPU, a prompt batch), the sequence's state, and the calls that feed tokens.
#include "engine.hpp"
#include "stage_copy.hpp"

#include "deepseek4.hpp"
#include "gemma4.hpp"
#include "glm5next.hpp"
#include "kolibri1.hpp"
#include "lfm2moe.hpp"
#include "flint.hpp"
#include "inkling2.hpp" // its kernels: inkling2.cu, and metal/src/poly/inkling.cpp on Metal
#if !defined(STRATA_GPU_METAL)
#include "laguna.hpp"   // CUDA and HIP: its YaRN turn and softplus gate have no Metal kernels yet
#include "expert_card.hpp"
#endif
#include "mmq.hpp"
#include "qwen35.hpp"
#include "runtime.hpp"
#include "split_plan.hpp"
#include "vocab_text.hpp"
#include "strata/poly/kernels.hpp"
#if defined(STRATA_GPU_METAL)
#include "strata/core/conversation_memory.hpp"
#include "strata/gpu/metal.hpp"
#endif
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/platform/memory.hpp"

#include "gguf_types.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <immintrin.h>
#include <new>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <set>
#include <unordered_map>

#ifdef __linux__
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#include <psapi.h>
#endif
#if defined(__APPLE__)
#include <sys/mman.h>
#endif

namespace strata::poly {
namespace {

// pdl_scope() for a whole pass's recording: CUDA's and HIP's (verify_kernels.cu); the Metal build has no PDL launches
// and does not build that file, so there it is a flag nothing reads
bool& whole_pdl_scope() {
#if defined(STRATA_GPU_METAL)
    static thread_local bool off = false;
    return off;
#else
    return kernels::pdl_scope();
#endif
}

void drop_file_pages(const void* p, size_t n) {
#if defined(__APPLE__)
    constexpr uintptr_t kPage = 16384;
    const uintptr_t a = ((uintptr_t) p + kPage - 1) & ~(kPage - 1);
    const uintptr_t b = ((uintptr_t) p + n) & ~(kPage - 1);
    if (b > a) (void) msync((void*) a, b - a, MS_INVALIDATE);
#else
    (void) p;
    (void) n;
#endif
}

// A launch's tiles are as wide as the most rows one of its matrices has, up to 128, and every matrix in it
// computes whole tiles.  So a prompt batch's cached experts go by how many rows they have (moe_rows()): a launch
// for the experts of up to 32 rows, one for those of up to 128, and one for the experts with more.  Narrower
// widths saved less work than their launches cost.
constexpr int kTile[] = {32, 128};
constexpr int kClasses = (int) (sizeof(kTile) / sizeof(kTile[0])) + 1;
constexpr int kMaxStagedTiles = 8;   // the most size classes STRATA_POLY_STAGED_TILES makes of the staged experts

int tile_class(int rows) {
    int c = 0;
    while (c < kClasses - 1 && rows > kTile[c]) ++c;
    return c;
}

// A container's CPU quota is not in the affinity mask: a 25-CPU pod on a 208-thread host shows 208.
// `quota` is set when there is one.
int usable_cpus(bool* quota_set = nullptr) {
    long long n = (long long) std::thread::hardware_concurrency();
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
    long long quota = -1, period = 0;
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {   // "max 100000" when there is no quota
        if (std::fscanf(f, "%lld %lld", &quota, &period) != 2) quota = -1;
        std::fclose(f);
    } else if (FILE* q = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        if (std::fscanf(q, "%lld", &quota) != 1) quota = -1;
        std::fclose(q);
        if (FILE* p = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(p, "%lld", &period) != 1) period = 0;
            std::fclose(p);
        }
    }
    if (quota > 0 && period > 0) {
        n = std::min(n, std::max(1LL, quota / period));
        if (quota_set) *quota_set = true;
    }
#else
    (void) quota_set;
#endif
    return (int) std::max(n, 1LL);
}

// The CPU threads when --threads names none: 70% of the usable CPUs, so that the PC's other programs keep the
// rest.  With a thread for every CPU the CPU was 90 to 94% busy while an answer was written, on 16 cores of a
// Xeon beside a card held to 8 GB; 22 threads of 32 left it 66% busy, wrote as fast, and 19 to 23% faster
// beside another program's load on 8 CPUs, since a layer waits for its slowest thread and a thread that has to
// share its CPU is the slowest.  A container with a CPU quota keeps them all: that is a server with no desktop
// to leave CPU to, and GLM 5.3 Flash's CPU experts there take 15 ms a token with 18 threads where 12.8 with 24.
int default_threads() {
    bool quota = false;
    const int n = usable_cpus(&quota);
    return quota ? n : std::max(1, (n * 7 + 5) / 10);
}

// On a machine with several NUMA nodes the file cache's pages are on the node whose CPU read them first, and the
// CPU reads the experts at the speed of memory: from the other node that is slower (29.4 against 22.5 ms of CPU
// experts a token on a two-socket Xeon).  `spans` are the experts in the mapped files.  When most of their pages
// are on one node, and the process may use `threads` CPUs of it, this thread is kept on that node's CPUs, and so
// are the threads it starts from then on.  Returns the node, or -1 when nothing was changed.
void drop_pages(const uint8_t* p, size_t n);
int stay_near(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const std::vector<std::string>& files,
              int threads, double& share) {
#ifdef __linux__
    if (access("/sys/devices/system/node/node1", F_OK) != 0) return -1;   // one node
    // A page shows in numa_maps once it is mapped: touch a few thousand, spread over the experts.
    size_t total = 0;
    for (const auto& s : spans) total += s.second;
    const size_t step = std::max<size_t>(total / 4096, 4096);
    volatile uint8_t seen = 0;
    for (const auto& s : spans)
        for (size_t at = 0; at < s.second; at += step) seen = (uint8_t) (seen + s.first[at]);

    std::vector<std::string> real;
    for (const std::string& f : files)
        if (char* r = realpath(f.c_str(), nullptr)) {
            real.emplace_back(r);
            std::free(r);
        }
    std::vector<long long> pages;   // per node, of the model's files
    if (FILE* f = std::fopen("/proc/self/numa_maps", "r")) {
        char* line = nullptr;
        size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            const char* file = std::strstr(line, " file=");
            if (!file) continue;
            file += 6;
            const size_t len = std::strcspn(file, " \n");
            bool ours = false;
            for (const std::string& r : real) ours = ours || (r.size() == len && r.compare(0, len, file, len) == 0);
            for (const char* p = file + len; ours && (p = std::strstr(p, " N")) != nullptr; ++p) {
                int node = 0;
                long long n = 0;
                if (std::sscanf(p, " N%d=%lld", &node, &n) != 2 || node < 0 || node > 1023) continue;
                if ((size_t) node >= pages.size()) pages.resize((size_t) node + 1, 0);
                pages[(size_t) node] += n;
            }
        }
        std::free(line);
        std::fclose(f);
    }
    // The touches mapped 64 KiB around each (the fault-around), 256 MiB of LFM2.5's experts on the RTX 5090 pod,
    // which stayed with every expert in VRAM.  They go again; the CPU's experts come back by map_pages().
    for (const auto& s : spans)
        for (size_t at = 0; at < s.second; at += step) {
            const uintptr_t b = (uintptr_t) s.first, e = b + s.second;
            const uintptr_t lo = std::max(b, (b + at) & ~(uintptr_t) 0xFFFF), hi = std::min(e, lo + 0x10000);
            drop_pages((const uint8_t*) lo, (size_t) (hi - lo));
        }
    long long all = 0;
    for (long long n : pages) all += n;
    if (all == 0) return -1;
    const int best = (int) (std::max_element(pages.begin(), pages.end()) - pages.begin());
    share = (double) pages[(size_t) best] / (double) all;
    if (share < 0.6) return -1;   // split between the nodes: one is as good as another

    cpu_set_t may, node;
    CPU_ZERO(&node);
    const std::string list = "/sys/devices/system/node/node" + std::to_string(best) + "/cpulist";
    if (FILE* f = std::fopen(list.c_str(), "r")) {   // "52-103,156-207"
        int a = 0, b = 0;
        while (std::fscanf(f, "%d", &a) == 1) {
            b = a;
            int c = std::fgetc(f);
            if (c == '-' && std::fscanf(f, "%d", &b) == 1) c = std::fgetc(f);
            for (int i = a; i <= b && i < CPU_SETSIZE; ++i) CPU_SET(i, &node);
            if (c != ',') break;
        }
        std::fclose(f);
    }
    if (sched_getaffinity(0, sizeof(may), &may) != 0) return -1;
    CPU_AND(&node, &node, &may);
    if (CPU_COUNT(&node) < threads || sched_setaffinity(0, sizeof(node), &node) != 0) return -1;
    return best;
#else
    (void) spans, (void) files, (void) threads, (void) share;
    return -1;
#endif
}

#ifdef _WIN32
// The spans' bytes read through the file cache, 16 MiB a request, into a buffer that is thrown away.  A span that
// is not in the model's files (a draft's own file) or a read that fails is left to map_pages()'s faults.
void read_cached(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const Shards& model,
                 const std::atomic<bool>& quit) {
    constexpr size_t kPiece = size_t(1) << 24;
    std::vector<HANDLE> files(model.paths().size(), INVALID_HANDLE_VALUE);
    std::vector<uint8_t> buf(kPiece);
    for (const auto& s : spans) {
        const auto [file, off] = model.place(s.first);
        if (file < 0) continue;
        HANDLE& h = files[(size_t) file];
        if (h == INVALID_HANDLE_VALUE)
            h = CreateFileA(model.paths()[(size_t) file].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) continue;
        for (size_t at = 0; at < s.second && !quit.load(std::memory_order_relaxed); at += kPiece) {
            OVERLAPPED o{};
            o.Offset = (DWORD) (off + at);
            o.OffsetHigh = (DWORD) ((off + at) >> 32);
            DWORD got = 0;
            if (!ReadFile(h, buf.data(), (DWORD) std::min(kPiece, s.second - at), &got, &o)) break;
        }
    }
    for (HANDLE h : files)
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
}
#endif

// A byte of every page of a span of the mapped files, read: its pages come into this process.
void touch_pages(const uint8_t* p, size_t n) {
    volatile uint8_t seen = 0;
    for (size_t at = 0; at < n; at += 4096) seen = (uint8_t) (seen + p[at]);
    if (n > 0) seen = (uint8_t) (seen + p[n - 1]);   // its last page, when it starts in one
}

// The mapped file's pages come into this process's page table as they are first read, a fault every few pages,
// and an expert read through those faults arrives at half the speed: 21.0 against 11.0 ms a token for GLM 5.3
// Flash's CPU experts on the two-socket Xeon, while every expert a token meets is new to the process.  So one
// thread reads a byte of every page of the CPU's experts, once, beside whatever the engine is doing: the 100 GiB of a
// model that the file cache holds take it 4 s.  What the cache does not hold is read from the disk here, not
// while a token waits.
//
// On Windows a fault reads only a few pages from the disk: 1.3 GB/s from a laptop's SSD, so the 10 GiB of experts
// the file cache did not hold took 10 s, and an answer asked meanwhile came at half speed.  The same bytes read
// through the file cache come at 3.2 GB/s (it reads ahead in large requests), and the mapping shares the cache's
// pages, so a second thread reads the spans from the file (read_cached()) while this one touches them: from an
// empty cache it runs ahead and the touches find the pages in memory.  When the cache holds them it only copies
// what it reads, beside the touches.  PrefetchVirtualMemory() returns at once and brings nothing in.
void map_pages(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const Shards& model,
               const std::atomic<bool>& quit) {
#ifdef _WIN32
    std::jthread reader(read_cached, std::cref(spans), std::cref(model), std::cref(quit));
#else
    (void) model;
#endif
    constexpr size_t kLook = size_t(1) << 24;   // `quit` is looked at every 16 MiB
    for (const auto& s : spans)
        for (size_t at = 0; at < s.second; at += kLook) {
            if (quit.load(std::memory_order_relaxed)) return;
            touch_pages(s.first + at, std::min(kLook, s.second - at));
        }
}

// The whole pages of a span of the mapped files leave this process's memory, for what the card has a copy of and
// the CPU does not read.  The file cache keeps them (the standby list on Windows, the page cache on Linux): the
// machine counts them as available and takes them for whatever needs memory, and a read that comes after finds
// them there without the disk.  After a 32K prompt on the laptop the engine held all 15.9 GiB of Qwen3.6's file,
// 4.2 GiB of it dense weights and experts the card had; with those dropped, the 11.7 GiB of experts the CPU reads.
size_t page_bytes() {
    static const size_t page = [] {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
#else
        return (size_t) sysconf(_SC_PAGESIZE);
#endif
    }();
    return page;
}

void drop_pages(const uint8_t* p, size_t n) {
    const size_t page = page_bytes();
    const uintptr_t a = ((uintptr_t) p + page - 1) / page * page, b = ((uintptr_t) p + n) / page * page;
    if (!p || b <= a) return;
#ifdef _WIN32
    VirtualUnlock((void*) a, b - a);   // pages that are not locked leave the working set (and it returns FALSE)
#else
    madvise((void*) a, b - a, MADV_DONTNEED);   // a private mapping that is never written reads the file again
#endif
}

#ifndef _WIN32
// A file's handle whose reads skip the file cache: O_DIRECT (whole 4 KiB blocks into aligned memory), F_NOCACHE on
// macOS; -1 where neither is there.
int open_uncached(const char* path) {
#if defined(O_DIRECT)
    return ::open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
#elif defined(F_NOCACHE)
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0 && fcntl(fd, F_NOCACHE, 1) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
#else
    (void) path;
    return -1;
#endif
}

// n bytes at off of open_uncached()'s fd into dst: the blocks around them into a buffer of the thread's, then dst.
// macOS serves an F_NOCACHE read from the file cache all the same unless it is in whole 16 KiB pages.
bool read_uncached(int fd, uint8_t* dst, uint64_t off, size_t n) {
#if defined(__APPLE__)
    constexpr size_t kBlock = 16384;
#else
    constexpr size_t kBlock = 4096;
#endif
    struct Buf {
        uint8_t* p = nullptr;
        size_t n = 0;
        ~Buf() { std::free(p); }
    };
    thread_local Buf b;
    const uint64_t lo = off / kBlock * kBlock, hi = (off + n + kBlock - 1) / kBlock * kBlock;
    const size_t span = (size_t) (hi - lo), need = (size_t) (off + n - lo);
    if (b.n < span) {
        std::free(b.p);
        b.p = (uint8_t*) std::aligned_alloc(kBlock, span);
        b.n = b.p ? span : 0;
        if (!b.p) return false;
    }
    size_t done = 0;
    while (done < need) {   // the file's last block may end short of hi
        const ssize_t got = ::pread(fd, b.p + done, span - done, (off_t) (lo + done));
        if (got <= 0) return false;
        done += (size_t) got;
    }
    std::memcpy(dst, b.p + (off - lo), n);
    return true;
}
#endif

// Options::lock_experts: a span's pages locked into RAM, where the file cache cannot hand them to another program,
// reading in the ones it had already dropped; false when the system refuses (Linux past `ulimit -l`, Windows past
// the working-set minimum lock_outside() raised).  Unlocking takes the span's whole pages only: one it shares with
// a neighbour stays as the neighbour has it.
bool lock_pages(const uint8_t* p, size_t n, bool lock) {
    if (lock) {
#ifdef _WIN32
        return VirtualLock((void*) p, n) != 0;
#else
        return mlock(p, n) == 0;
#endif
    }
    const size_t page = page_bytes();
    const uintptr_t a = ((uintptr_t) p + page - 1) / page * page, b = ((uintptr_t) p + n) / page * page;
    if (b <= a) return true;
#ifdef _WIN32
    return VirtualUnlock((void*) a, b - a) != 0;
#else
    return munlock((void*) a, b - a) == 0;
#endif
}

// `ulimit -l` as the shell prints it (KiB) for the --lock-experts lines: "" where there is none, or no limit.
std::string memlock_kib() {
#ifdef _WIN32
    return {};
#else
    rlimit r{};
    if (getrlimit(RLIMIT_MEMLOCK, &r) != 0 || r.rlim_cur == RLIM_INFINITY) return {};
    return std::to_string((unsigned long long) (r.rlim_cur >> 10));
#endif
}

#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
// The mirror's lock line: the NVIDIA driver pins past RLIMIT_MEMLOCK (under `ulimit -l 8192` it pinned 3.3 GiB of
// Qwen3.6's experts, driver 580 on Linux).
std::string memlock_note() {
    const std::string kib = memlock_kib();
    return kib.empty() ? std::string() : "; `ulimit -l` (" + kib + " here) does not limit it";
}

// Host memory for the pinned mirror, aligned to 2 MiB.  Not asked for in huge pages (MADV_HUGEPAGE): on
// lab-pro6000, whose memory others had fragmented, the kernel compacted memory for each fault, the mirror took
// minutes and an answer meanwhile came at 4 tokens/s.
uint8_t* host_alloc(size_t bytes) {
    constexpr size_t kAlign = size_t(2) << 20;
    const size_t n = (bytes + kAlign - 1) / kAlign * kAlign;
#ifdef _WIN32
    return (uint8_t*) _aligned_malloc(n, kAlign);
#else
    return (uint8_t*) std::aligned_alloc(kAlign, n);
#endif
}

void host_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
#endif

void free_pieces(std::vector<MirrorPiece>& pieces) {
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    for (const MirrorPiece& p : pieces) {
        if (!p.base) continue;
        if (p.registered) {
            cudaHostUnregister(p.base);
            host_free(p.base);
        } else {
            cudaFreeHost(p.base);
        }
    }
#endif
    pieces.clear();
}

// cuBLAS's libraries: loaded at start, they stayed in the working set after that (135 MiB on the laptop, 68 MiB
// under Linux), though a request runs little of them.  Their pages leave it as drop_pages()'s do, and what a
// request runs comes back from the file cache.  Not the engine's own image: under Linux its code went to the disk
// with MADV_PAGEOUT and came back from it on the next request, 4 ms of a short prompt's 85.
void drop_images() {
#ifdef _WIN32
    const HANDLE self = GetCurrentProcess();
    HMODULE mods[1024];
    DWORD got = 0;
    if (!EnumProcessModules(self, mods, sizeof mods, &got)) return;
    for (DWORD i = 0; i < std::min<DWORD>(got / sizeof(HMODULE), 1024); ++i) {
        char name[MAX_PATH];
        if (!GetModuleBaseNameA(self, mods[i], name, sizeof name) || _strnicmp(name, "cublas", 6) != 0) continue;
        MODULEINFO mi{};
        if (!GetModuleInformation(self, mods[i], &mi, sizeof mi)) continue;
        const uint8_t* p = (const uint8_t*) mi.lpBaseOfDll;
        const uint8_t* end = p + mi.SizeOfImage;
        MEMORY_BASIC_INFORMATION r;
        while (p < end && VirtualQuery(p, &r, sizeof r) == sizeof r) {   // one call a region: they differ in rights
            if (r.State == MEM_COMMIT) VirtualUnlock(r.BaseAddress, r.RegionSize);
            p = (const uint8_t*) r.BaseAddress + r.RegionSize;
        }
    }
#else
    // Their code only: it is never written, so MADV_DONTNEED leaves it in the file cache.  Their other read-only
    // pages may be relocated private copies, which MADV_DONTNEED would throw away.
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[4352];
    while (std::fgets(line, sizeof line, f)) {
        unsigned long a = 0, b = 0;
        char rights[8] = {};
        if (std::sscanf(line, "%lx-%lx %7s", &a, &b, rights) == 3 && std::strcmp(rights, "r-xp") == 0 &&
            std::strstr(line, "/libcublas"))
            madvise((void*) a, b - a, MADV_DONTNEED);
    }
    std::fclose(f);
#endif
}

// Experts [e0, e1) of L in the mapped file, to f(start, bytes): their gates and ups (one span where the file keeps
// an expert's two in one tensor), and their downs.
template <class F> void expert_bytes(const Layer& L, int e0, int e1, F&& f) {
    const size_t n = (size_t) (e1 - e0);
    f(L.exp_gate + (size_t) e0 * L.gu_stride, n * L.gu_stride);
    if (L.gu_stride == L.gu_bytes) f(L.exp_up + (size_t) e0 * L.u_stride, n * L.u_stride);
    f(L.exp_down + (size_t) e0 * L.d_bytes, n * L.d_bytes);
}

// expert_bytes() of each run of L's experts that take() takes, a run of neighbours in one call
template <class P, class F> void expert_runs(const Layer& L, int n_expert, P&& take, F&& f) {
    for (int e = 0; e < n_expert;) {
        if (!take(e)) {
            ++e;
            continue;
        }
        int g = e + 1;
        while (g < n_expert && take(g)) ++g;
        expert_bytes(L, e, g, f);
        e = g;
    }
}

// Lines of "layer expert count".  A missing file is not an error: the first run writes it.
void read_profile(const std::string& path, int n_layer, int n_expert, std::vector<int64_t>& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    out.assign((size_t) n_layer * n_expert, 0);
    int il = 0, e = 0;
    long long n = 0;
    while (std::fscanf(f, "%d %d %lld", &il, &e, &n) == 3)
        if (il >= 0 && il < n_layer && e >= 0 && e < n_expert && n > 0) out[(size_t) il * n_expert + e] = n;
    std::fclose(f);
}

void cu_ck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(what) + ": " + (s ? s : "a CUDA driver error"));
}

size_t up_to(size_t n, size_t grain) { return (n + grain - 1) / grain * grain; }

// The largest of n values, as eight maxima side by side: one best-so-far is a chain every compare waits on, and
// a vocabulary is 150,000 values.
constexpr int kLanes = 8;
float largest(const float* x, int n) {
    float mx[kLanes];
    std::fill(mx, mx + kLanes, -INFINITY);
    int i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (int j = 0; j < kLanes; ++j) mx[j] = std::max(mx[j], x[i + j]);
    float best = -INFINITY;
    for (int j = 0; j < kLanes; ++j) best = std::max(best, mx[j]);
    for (; i < n; ++i) best = std::max(best, x[i]);
    return best;
}

// The CUDA calls of a card's engine go to its card: a split's engines share the caller's thread (but in read()),
// and the card is the thread's.  Back to the caller's card at the end.  One engine alone changes nothing.
class OnCard {
public:
    OnCard(int device, bool split) : split_(split) {
        if (!split_) return;
        cudaGetDevice(&was_);
        cudaSetDevice(device);   // a failure shows in the calls that follow
    }
    ~OnCard() {
        if (split_) cudaSetDevice(was_);
    }
    OnCard(const OnCard&) = delete;
    OnCard& operator=(const OnCard&) = delete;

private:
    bool split_;
    int was_ = 0;
};

// Work on a thread of its own, whose exception wait() throws.  Going out of scope it waits and drops the exception.
class Behind {
public:
    Behind() = default;
    Behind(const Behind&) = delete;
    Behind& operator=(const Behind&) = delete;
    ~Behind() {
        if (th_.joinable()) th_.join();
    }
    void go(std::function<void()> work) {
        th_ = std::thread([this, work = std::move(work)] {
            try {
                work();
            } catch (...) {
                err_ = std::current_exception();
            }
        });
    }
    void wait() {
        if (th_.joinable()) th_.join();
        if (err_) std::rethrow_exception(std::exchange(err_, nullptr));
    }

private:
    std::thread th_;
    std::exception_ptr err_;
};

// A card's memory bandwidth in GB/s for split_plan.hpp, from its peak memory clock and bus width; 0 where it is not
// known (Metal; HIP, whose clocks are not read this way yet).  STRATA_POLY_CARD_GBS=a,b,... gives each card's.
double card_gbs(int d) {
    if (const char* v = std::getenv("STRATA_POLY_CARD_GBS"); v && *v) {
        const char* at = v;
        for (int k = 0; k < d && at; ++k) at = std::strchr(at, ',') ? std::strchr(at, ',') + 1 : nullptr;
        return at ? std::max(0.0, std::atof(at)) : 0.0;
    }
#if defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP)
    return 0;
#else
    int khz = 0, bits = 0;
    if (cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, d) != cudaSuccess ||
        cudaDeviceGetAttribute(&bits, cudaDevAttrGlobalMemoryBusWidth, d) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return 2.0 * khz * 1e3 * (bits / 8.0) / 1e9;   // double data rate
#endif
}

// The first layer of each card's share of n_layer layers, for Options::layer_split on `cards` cards; empty: no
// split.  "auto" shares the layers' bytes out by each card's free VRAM, less what every card's engine takes beside
// its layers and, on the last card, the output and the draft block.  Then every card caches about the same share
// of its layers' experts, which is where a byte of VRAM on one is worth as much as on another.  No card starts
// inside [keep_lo, keep_hi) but at keep_lo: those layers run on one card (Inkling 2's loop); "auto" takes the
// nearer of keep_lo and keep_hi by the layers' bytes.  With such a range run `loop_runs` times a token on two cards
// of different speeds, "auto" may take another start and swap the cards (*swap: the second device takes the first
// layers) where split_plan.hpp estimates that clearly faster: the loop on the faster card.
std::vector<int> layer_starts(const std::string& split, const Shards& file, int n_layer, int cards, int keep_lo = 0,
                              int keep_hi = 0, int loop_runs = 1, bool* swap = nullptr) {
    if (split.empty() || cards < 2) return {};
    const auto inside = [&](int il) { return il > keep_lo && il < keep_hi; };
    std::vector<int> starts{0};
    if (split != "auto") {
        for (size_t at = 0; at < split.size();) {
            const size_t comma = std::min(split.find(',', at), split.size());
            const std::string s = split.substr(at, comma - at);
            char* end = nullptr;
            const long v = std::strtol(s.c_str(), &end, 10);
            // the last card keeps two layers at least: Gemma 4's draft model reads the last two layers' caches
            if (s.empty() || *end || v <= starts.back() || v > n_layer - 2)
                throw std::runtime_error("--layer-split " + split + ": the first layer of each later card, rising, "
                                         "between 1 and " + std::to_string(n_layer - 2) + " (\"auto\" shares them "
                                         "out by each card's free VRAM)");
            if (inside((int) v))
                throw std::runtime_error("--layer-split " + split + ": layers " + std::to_string(keep_lo) + " to " +
                                         std::to_string(keep_hi - 1) + " run on one card, so a card starts at layer " +
                                         std::to_string(keep_lo) + " or " + std::to_string(keep_hi) +
                                         ", or outside them");
            starts.push_back((int) v);
            at = comma + 1;
        }
        if ((int) starts.size() > cards)
            throw std::runtime_error("--layer-split " + split + " is for " + std::to_string(starts.size()) +
                                     " cards; this process sees " + std::to_string(cards));
        return starts;
    }
    if (n_layer < 2 * cards) throw std::runtime_error("the model has too few layers to split them on " + std::to_string(cards) + " cards");
    std::vector<double> before((size_t) n_layer + 1, 0.0);   // the bytes of the layers before each one
    std::vector<SplitLayer> layers((size_t) n_layer);         // split_plan.hpp's
    double head = 0;
    file.each([&](const TensorInfo& t) {
        int il = -1;
        if (std::sscanf(t.name.c_str(), "blk.%d.", &il) == 1 && il >= 0 && il < n_layer) {
            before[(size_t) il + 1] += (double) file.payload(t);
            (t.name.find("_exps.") != std::string::npos ? layers[(size_t) il].experts : layers[(size_t) il].dense) +=
                (double) file.payload(t);
        }
        else if (il >= n_layer || t.name == "output.weight")   // the draft block, the output
            head += (double) file.payload(t);
    });
    if (!file.find("output.weight") && file.find("token_embd.weight"))   // the output is the token table
        head += (double) file.payload(*file.find("token_embd.weight"));
    for (int il = 0; il < n_layer; ++il) before[(size_t) il + 1] += before[(size_t) il];
    constexpr double kBeside = 1.5 * double(1ull << 30);   // scratch, the staging buffer and the reserve, about
    std::vector<double> room((size_t) cards);
    SplitCard two[2];
    double all_room = 0;
    int devices = 1;
    ck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    for (int d = 0; d < cards; ++d) {   // (STRATA_POLY_CARDS: parts on the devices there are, in turn)
        size_t free_b = 0, total_b = 0;
        ck(cudaSetDevice(d % devices), "cudaSetDevice");
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        room[(size_t) d] = std::max((double) free_b - kBeside - (d == cards - 1 ? head : 0.0), double(1ull << 28));
        all_room += room[(size_t) d];
        if (d < 2) two[d] = SplitCard{std::max((double) free_b - kBeside, 0.0), card_gbs(d)};
    }
    double share = 0;
    for (int d = 1; d < cards; ++d) {
        share += room[(size_t) d - 1] / all_room;
        int il = starts.back() + 1;   // card d starts where the layers before it pass the cards before it's share
        const int most = n_layer - 2 * (cards - d);
        while (il < most && before[(size_t) il] < share * before[(size_t) n_layer]) ++il;
        if (inside(il)) {
            const double want = share * before[(size_t) n_layer];
            const bool lo_ok = keep_lo > starts.back(), hi_ok = keep_hi <= most;
            if (!lo_ok && !hi_ok)
                throw std::runtime_error("layers " + std::to_string(keep_lo) + " to " + std::to_string(keep_hi - 1) +
                                         " run on one card: there is no place for card " + std::to_string(d) +
                                         " to start");
            il = !hi_ok || (lo_ok && want - before[(size_t) keep_lo] <= before[(size_t) keep_hi] - want) ? keep_lo
                                                                                                          : keep_hi;
        }
        starts.push_back(il);
    }
    if (swap && cards == 2 && keep_hi > keep_lo && loop_runs > 1) {   // Inkling 2's loop on the faster card
        for (int il = keep_lo; il < keep_hi; ++il) layers[(size_t) il].runs = loop_runs;
        const std::string arch = file.get("general.architecture") ? file.get("general.architecture")->s : "";
        const MetaValue* used = file.get(arch + ".expert_used_count");
        const MetaValue* all = file.get(arch + ".expert_count");
        const double reads = used && all && all->u > 0 ? double(used->u) / double(all->u) : 1.0 / 16;
        const SplitPlan plan = loop_split(layers, head, two, keep_lo, keep_hi, starts[1], reads);
        if (plan.cut != starts[1] || plan.swap) {
            const bool first = plan.cut >= keep_hi;   // the first part has the loop
            std::fprintf(stderr,
                         "strata-poly: layer split auto: the loop (layers %d to %d, %d runs a token) on the faster "
                         "card, device %d (%.0f GB/s, the other %.0f): the second part from layer %d%s, %.1f ms a "
                         "token estimated where %d in the devices' order is %.1f\n",
                         keep_lo, keep_hi - 1, loop_runs, first != plan.swap ? 0 : 1, two[first != plan.swap ? 0 : 1].gbs,
                         two[first != plan.swap ? 1 : 0].gbs, plan.cut, plan.swap ? ", the devices swapped" : "",
                         plan.ms, starts[1], split_ms(layers, head, two, starts[1], false, reads));
        }
        starts[1] = plan.cut;
        *swap = plan.swap;
    }
    return starts;
}

// The reserve when none is asked for: a thirty-second of the card, 256 to 768 MiB.  When the machine's RAM is
// tight for the model, a sixteenth, 512 to 1,024: on an 8 GB laptop with 32 GB, Ornith's APEX-I-Quality (23 GB)
// read a 24,000-token prompt at 1,392 and 1,138 tokens/s with 256 MiB kept free and at 1,532 and 1,451 with 512,
// in turns with the build that kept 512 (1,496 and 1,311); the smaller files read as fast either way.  Under
// Windows what the card is short of moves to the system's RAM, which the file then has less of.
int default_reserve_mib(size_t total_b, bool ram_tight) {
    if (ram_tight) return (int) std::clamp<size_t>(total_b >> 24, 512, 1024);
    return (int) std::clamp<size_t>(total_b >> 25, 256, 768);
}

#ifdef __linux__
// A cgroup's memory limit is not in /proc/meminfo: a container, or a systemd scope, with one shows the host's RAM.
// The room under the tightest limit on the way to the root: each limit less what its group holds that the kernel
// cannot drop (its file pages it can).  SIZE_MAX when no group has a limit.
size_t cgroup_room() {
    std::string path;
    if (FILE* f = std::fopen("/proc/self/cgroup", "r")) {
        char line[512];
        while (std::fgets(line, sizeof(line), f))
            if (std::strncmp(line, "0::", 3) == 0) {
                path = line + 3;
                break;
            }
        std::fclose(f);
    }
    while (!path.empty() && path.back() == '\n') path.pop_back();
    if (path.empty() || path[0] != '/') return SIZE_MAX;   // cgroup v1 only
    size_t room = SIZE_MAX;
    for (;;) {
        const std::string dir = "/sys/fs/cgroup" + (path == "/" ? std::string() : path);
        unsigned long long max = 0, cur = 0, file = 0;
        bool limited = false;   // "max" when there is no limit
        if (FILE* f = std::fopen((dir + "/memory.max").c_str(), "r")) {
            limited = std::fscanf(f, "%llu", &max) == 1;
            std::fclose(f);
        }
        if (limited) {
            if (FILE* f = std::fopen((dir + "/memory.current").c_str(), "r")) {
                if (std::fscanf(f, "%llu", &cur) != 1) cur = 0;
                std::fclose(f);
            }
            if (FILE* f = std::fopen((dir + "/memory.stat").c_str(), "r")) {
                char key[64];
                unsigned long long v = 0;
                while (std::fscanf(f, "%63s %llu", key, &v) == 2)
                    if (std::strcmp(key, "file") == 0) {
                        file = v;
                        break;
                    }
                std::fclose(f);
            }
            const unsigned long long held = cur > file ? cur - file : 0;
            room = std::min(room, max > held ? (size_t) (max - held) : 0);
        }
        if (path == "/") break;
        const size_t cut = path.find_last_of('/');
        path = cut == 0 ? "/" : path.substr(0, cut);
    }
    return room;
}
#endif

// The RAM the machine has free for the model's file, or 0 when it cannot tell: under a cgroup's limit, what that
// leaves if it is less, and then *own is true.
size_t ram_available(bool* own = nullptr) {
    if (own) *own = false;
#if defined(STRATA_GPU_METAL)
    return strata::core::available_host_bytes().value_or(0);
#elif defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#else
    size_t kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[128];
        while (std::fgets(line, sizeof(line), f))
            if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
        std::fclose(f);
    }
#ifdef __linux__
    const size_t group = cgroup_room();
    if (own) *own = group < (kb << 10);
    return std::min(kb << 10, group);
#else
    return kb << 10;
#endif
#endif
}

}  // namespace

// One expert on its way into the cache.  The engine's thread asks, the follower copies and sends, and the engine's
// thread gives it the slot once the upload has arrived.
struct Engine::Swap {
    enum { kFree, kAsked, kSent };
    std::atomic<int> at{kFree};
    int il = 0, e = 0;
    uint8_t* slot = nullptr;       // where it goes in VRAM: the slot of the expert it replaces
    bool adds = false;             // the slot was empty (grow_cache()): the cache holds one more
    cudaEvent_t done = nullptr;    // marks its upload
    int32_t from = -1;             // the pool_ slot it is copied from, held (pool_pin_) until it has arrived
    bool mapped = true;            // else read from the files
    int64_t takes = -1;            // the expert (layer * n_expert + expert) given its place in kept_, or -1
    int64_t gets = -1;             // the expert given `from` once it has arrived, or -1
    const uint8_t* mirror = nullptr;   // its slot in the mirror, uploaded from there; given back once it has arrived
    uint8_t* spare = nullptr;      // a spare mirror slot the expert it replaces is copied back into, or null
    int64_t out = -1;              // that expert (layer * n_expert + expert)
    cudaEvent_t freed = nullptr;   // card_far_: the expert it replaces is in its spare already, copied on st_
    bool wait_freed = false;
};

Engine::Engine() = default;

Engine::~Engine() {
    next_.reset();
    OnCard here(device_, split_);
    map_quit_.store(true);
    if (mapper_.joinable()) mapper_.join();
    {
        std::lock_guard<std::mutex> l(pool_m_);
        pool_quit_ = true;
    }
    pool_cv_.notify_all();
    for (std::thread& t : pool_threads_) t.join();
    if (follower_.joinable()) {
        {
            std::lock_guard<std::mutex> l(swap_m_);
            swap_quit_ = true;
        }
        swap_cv_.notify_all();
        follower_.join();
    }
    if (lock_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> l(lock_m_);
            lock_quit_ = true;
        }
        lock_cv_.notify_all();
        lock_thread_.join();
    }
    if (swap_st_) cudaStreamSynchronize(cs(swap_st_));
    if (up_) cudaStreamSynchronize(cs(up_));
    if (rs_) cudaStreamSynchronize(cs(rs_));
    if (st_) cudaStreamSynchronize(cs(st_));
    aside_.reset();
    workers_.reset();
    mmq_.reset();
#if defined(STRATA_GPU_METAL)
    // Metal recordings retain their resources. Retire them before releasing the allocation owners.
    for (void* g : graphs_) cudaGraphExecDestroy((cudaGraphExec_t) g);
    graphs_.clear();
#endif
    free_pieces(mirror_blocks_);
    free_pieces(lock_blocks_);
    for (void* e : gather_ev_)
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (void* p : dev_allocs_) cudaFree(p);
    for (Kv& k : kv_) vmm_free(k);
    for (Seq& q : seqs_) {
        for (Kv& k : q.kv) vmm_free(k);
        // The fp16 twins of the int8 caches (Gemma 4's --kv int8) are the sequences'.
        if (q.twin_k.base) vmm_free(q.twin_k);
        if (q.twin_v.base) vmm_free(q.twin_v);
    }
    vmm_free(arena_);
    vmm_free(stage_map_);
    vmm_free(carry_map_);
    vmm_free(prompt_map_);
    vmm_free(work_map_);
    for (unsigned long long h : vpool_) cuMemRelease(h);   // the pool goes back to the card with the engine
    vpool_.clear();
    for (void* p : host_allocs_) cudaFreeHost(p);
    for (void* e : {ev_stage_[0], ev_stage_[1], ev_ahead_, ev_free_, ev_in_[0], ev_in_[1], ev_out_[0], ev_out_[1], ev_hr_,
                    ev_saved_, ev_bell_})
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : swap_pin_ev_)
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : ev_row_)
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (const Ahead& g : ahead_) {
        for (void* e : {g.up, g.read})
            if (e) cudaEventDestroy((cudaEvent_t) e);
        for (void* e : g.arrived)
            if (e) cudaEventDestroy((cudaEvent_t) e);
    }
    for (int j = 0; j < n_swap_; ++j)
        for (cudaEvent_t e : {swap_[(size_t) j].done, swap_[(size_t) j].freed})
            if (e) cudaEventDestroy(e);
    for (void* g : graphs_) cudaGraphExecDestroy((cudaGraphExec_t) g);
    blas_destroy(blas_);
    if (swap_st_) cudaStreamDestroy(cs(swap_st_));
    if (up_) cudaStreamDestroy(cs(up_));
    if (rs_) cudaStreamDestroy(cs(rs_));
    if (st_) cudaStreamDestroy(cs(st_));
}

void* Engine::dmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, std::max<size_t>(bytes, 256));
    if (e != cudaSuccess) {
        cudaGetLastError();   // as ck() does
        throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(bytes >> 20) +
                                 " MiB): " + cudaGetErrorString(e));
    }
    dev_allocs_.push_back(p);
    return p;
}

void* Engine::hmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const std::string why = std::string(what) + " (" + std::to_string(bytes) + " bytes)";
    // A container at its memory limit refused pinned memory with "invalid argument" five times in five minutes
    // while a download filled its page cache; a moment later the kernel has reclaimed some.
    cudaError_t e = cudaSuccess;
    for (int tries = 0; tries < 4; ++tries) {
        e = cudaMallocHost(&p, std::max<size_t>(bytes, 256));
        if (e == cudaSuccess) break;
        cudaGetLastError();
        std::fprintf(stderr, "strata-poly: %s: %s, trying again\n", why.c_str(), cudaGetErrorString(e));
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    ck(e, why.c_str());
    host_allocs_.push_back(p);
    return p;
}

// What hmalloc() gave, given back.  Null is nothing.
void Engine::hfree(void* p) {
    if (!p) return;
    host_allocs_.erase(std::find(host_allocs_.begin(), host_allocs_.end(), p));
    ck(cudaFreeHost(p), "freeing pinned memory");
}

// A write into VRAM and a run of zeros in it, both on st_.
//
// They used to be plain cudaMemcpy()/cudaMemset(), which run on the implicit (legacy) stream.  That stream is
// NOT ordered against this engine's - every one of them is created with cudaStreamNonBlocking - so none of the
// cudaStreamSynchronize(cs(st_)) that guard VRAM waits for those copies.  The load fills the expert cache with
// them, and the first request whose prompt is longer than kKvGrow reaches cover() with rows to map: cover()
// asks shrink_cache() for the VRAM, and shrink_cache() gives pieces back (vmm_unmap, "the card must be done
// with them") while the load's last copies are still writing them.  The card then says
// "rms_norm: an illegal memory access" at the first token of the prompt - for any prompt over kKvGrow tokens, on
// any architecture (LFM2.5 and Qwen3.6 alike), and never for a shorter one, whose cover() has nothing to map
// and so never gives VRAM back.  On st_ the same copies are behind the syncs that were already there.
void Engine::h2d(void* dst, const void* src, size_t bytes) {
    ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cs(st_)), "a copy into VRAM");
}

void Engine::zero_vram(void* dst, size_t bytes) {
    ck(cudaMemsetAsync(dst, 0, bytes, cs(st_)), "zeroing VRAM");
}

// A container whose memory is not much more than the model's files.  The file cache cannot hold them beside
// the engine, so an answer reads experts from the disk: on a pod with 80 GiB and two RTX 3080s, GLM REAP50's
// 79 GB file wrote at 6 tokens/s where 20.  And the pod ended the engine with SIGKILL in half of its long reads,
// within 2% of the limit, with no OOM kill counted by the kernel.  Said once at load; the remedy is RAM.
void Engine::mind_memory() {
#ifdef __linux__
    if (split_ && lo_ > 0) return;   // the first card's engine says it for all of them
    static const char* const files[] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    unsigned long long limit = 0;
    for (const char* path : files) {
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        if (std::fscanf(f, "%llu", &limit) != 1) limit = 0;   // "max" when there is none
        std::fclose(f);
        break;
    }
    if (limit == 0 || limit >= (1ULL << 50)) return;   // none, or one beyond any machine's RAM
    const uint64_t file = gg_->bytes() + (dg_ ? dg_->bytes() : 0);
    if (file + (8ULL << 30) <= limit) return;   // the engine and the rest of the container take a few GiB
    std::fprintf(stderr, "strata-poly: the model's files are %.1f GiB and the container has %.1f GiB: the file "
                         "cache cannot hold them beside the engine, so answers will read experts from the disk, "
                         "and the container may end the engine\n",
                 (double) file / (1 << 30), (double) limit / (1 << 30));
#endif
}

#ifdef STRATA_VMM_EQUAL_PIECES
// Whether a reservation's pieces must all be of one size (hip_compat/cuda.h): under WSL2, where ROCm reaches the card
// through librocdxg (/dev/dxg, and no /dev/kfd).  STRATA_VMM_EQUAL_PIECES=1 or 0 says it for any Linux HIP build.
static bool vmm_equal_pieces() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_VMM_EQUAL_PIECES"); v && v[0]) return v[0] != '0';
        std::error_code ec;
        return std::filesystem::exists("/dev/dxg", ec) && !std::filesystem::exists("/dev/kfd", ec);
    }();
    return on;
}

// The unit such pieces are made of, 2 MiB like the pool's handles: vmm_reserve() reserves a whole number of them.
static size_t vmm_unit(size_t grain) { return up_to(size_t(2) << 20, grain); }

// The size of every piece of a reservation of `reserved` bytes: `first`, the size of the piece it has, or for its
// first map the most units that divide the reservation and are no more than the piece asked for and the map's bytes.
// A map then covers a whole number of pieces, and the last one is no shorter than the others.  A map of any piece size
// (SIZE_MAX: the staging buffer, a read's rows) takes the divisor that maps the fewest units past the map's bytes, of
// 16 MiB or more where one maps at most half again as much (a piece is a file descriptor on ROCm), the larger on a tie:
// the most units would map a reservation of 100 units whole for a first map of 51.
static size_t vmm_piece(size_t reserved, size_t first, size_t bytes, size_t piece, size_t unit) {
    if (first) return first;
    const size_t r = reserved / unit;
    if (piece == SIZE_MAX) {
        const size_t b = std::max<size_t>(up_to(std::min(bytes, reserved), unit) / unit, 1);
        const auto least = [&](size_t from) {   // {the divisor of r from `from` on that maps least past b, its excess}
            size_t best = r, past = up_to(b, r) - b;
            for (size_t k = r; k-- > from;)
                if (k > 0 && r % k == 0 && up_to(b, k) - b < past) best = k, past = up_to(b, k) - b;
            return std::pair<size_t, size_t>(best, past);
        };
        const auto wide = least(std::min<size_t>(r, (size_t(16) << 20) / unit));
        return (2 * wide.second <= b ? wide.first : least(1).first) * unit;
    }
    const size_t most = std::min(up_to(std::min(piece, reserved), unit), up_to(bytes, unit)) / unit;
    size_t k = std::max<size_t>(most, 1);
    while (k > 1 && r % k != 0) --k;
    return k * unit;
}
#endif

void Engine::vmm_reserve(Mapped& m, size_t bytes) {
    CUdeviceptr p = 0;
#ifdef STRATA_VMM_EQUAL_PIECES
    // with equal pieces, a whole number of their units
    m.reserved = up_to(std::max<size_t>(bytes, 1), pool_on_ ? pg_ : vmm_equal_pieces() ? vmm_unit(grain_) : grain_);
#else
    m.reserved = up_to(std::max<size_t>(bytes, 1), pool_on_ ? pg_ : grain_);   // a whole number of pool handles
#endif
    cu_ck(cuMemAddressReserve(&p, m.reserved, 0, 0, 0), "reserving addresses for VRAM");
    m.base = p;
}

// The pool: `bytes` of VRAM as handles of pg_ bytes, kept until the engine ends.  Creating a handle takes its VRAM
// (measured on Windows HIP: the card's free figure drops at the create, before anything is mapped), so what is
// claimed here is the engine's from now on.  The card may have less: the pool is what it could give.
void Engine::vpool_claim(size_t bytes) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    vpool_.reserve(vpool_.size() + bytes / pg_ + 1);
    for (size_t got = 0; got + pg_ <= bytes; got += pg_) {
        CUmemGenericAllocationHandle h = 0;
        if (cuMemCreate(&h, pg_, &prop, 0) != CUDA_SUCCESS) {
            (void) cudaGetLastError();
            break;
        }
        vpool_.push_back(h);
    }
}

// The pool sized by what the OS lets this process have, where the OS says (Windows: the video memory budget).  The
// budget is not a constant: it fell from 7,383 to 6,975 MiB on an 8 GB card while this engine claimed 3.6 GiB, and a
// process that is over it has memory moved out to system RAM (150 MiB over: 2.4 GB moved, decode at 6 tokens/s).  So
// handles are created a step at a time and the budget and the usage are read again before each step, until what is
// left under the budget is `margin`: the room for what the engine allocates after the pool (recorded launches, library
// work space) and the desktop's growth.  False when there is no budget to read (elsewhere than Windows): the caller
// sizes the pool from the card's free VRAM.
bool Engine::vpool_claim_to_budget(size_t margin, size_t limit, size_t& claimed) {
    claimed = 0;
    cudaDeviceProp p{};
    uint64_t budget = 0, usage = 0;
    std::string why;
    if (cudaGetDeviceProperties(&p, device_) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (!strata::platform::gpu_local_memory_budget(p.luid, budget, usage, why)) return false;
    const size_t step = size_t(256) << 20;
    for (;;) {
        if (!strata::platform::gpu_local_memory_budget(p.luid, budget, usage, why)) break;
        const uint64_t keep = (uint64_t) margin + pg_;
        if (budget <= usage + keep || vpool_bytes() + pg_ > limit) break;   // at the budget, or all the model can use
        const size_t room = std::min<uint64_t>({budget - usage - margin, step, limit - vpool_bytes()});
        const size_t before = vpool_.size();
        vpool_claim(room);
        claimed += (vpool_.size() - before) * pg_;
        if (vpool_.size() - before < room / pg_) break;   // the card had no more
    }
    return true;
}

// What the attention caches take for one step of their growth (cover() maps kv_grow() rows at a time, in handles of the
// pool's size): the room the pool keeps for the next one.
size_t Engine::kv_step_bytes() const {
    size_t b = 0;
    for (const Kv& k : kv_) b += up_to((size_t) ((kv_grow() + k.per - 1) / k.per) * k.bytes, pg_);
    return b;
}

// ROCm on Linux maps a piece through amdgpu's GEM_VA call (the handle's dmabuf, imported into the render node and
// given its address there), which queues the page-table update and returns.  A graphics job waits for that update
// when it is submitted; a kernel launched from a user-mode queue waits for nothing.  Where amdgpu writes a compute
// VM's page tables with SDMA jobs and not from the CPU - a card whose BAR does not cover its VRAM, so a PC without
// Resizable BAR (amdgpu's vm_update_mode) - the kernel that touches the piece next can run before its pages are
// there: "Memory Fault Error ... kernel: __amd_rocclr_fillBufferAligned", and "an illegal memory access" from every
// call after it.
//
// An ordinary allocation is mapped by KFD's own call, which waits for its page-table job, queued behind the VM's
// earlier ones, and then flushes the TLB.  One made and freed here is the wait, for as long as the kernel takes.  It
// has to reach the kernel: 4 MiB does, 64 KiB came out of a block ROCm already held and waited for nothing.  A card
// with no VRAM left for it gets 20 ms.  STRATA_VMM_SETTLE=0 leaves the wait out, to see the fault.
void Engine::vmm_settle() {
#ifdef STRATA_VMM_MAP_SETTLES
    static const bool off = [] {
        const char* v = std::getenv("STRATA_VMM_SETTLE");
        return v && v[0] == '0';
    }();
    if (off) return;
    OnCard here(device_, split_);
    void* p = nullptr;
    if (cudaMalloc(&p, size_t(4) << 20) == cudaSuccess) {
        ck(cudaFree(p), "waiting for a VRAM mapping");
    } else {
        (void) cudaGetLastError();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
#endif
}

// VRAM for the first `bytes` of m, in pieces of at most `piece` bytes.  Throws std::runtime_error.
void Engine::vmm_map(Mapped& m, size_t bytes, size_t piece, const char* what) {
    // With the pool a piece is one of its handles and comes from it, never from the card; a map it cannot serve
    // throws as a card that has no more VRAM does
    if (pool_on_) piece = pg_;
#ifdef STRATA_VMM_EQUAL_PIECES
    if (!pool_on_ && vmm_equal_pieces()) {   // whole pieces of the reservation's one size
        piece = vmm_piece(m.reserved, m.pieces.empty() ? 0 : m.pieces.front().second, bytes, piece, vmm_unit(grain_));
        bytes = std::min(up_to(bytes, piece), m.reserved);
    } else {
        bytes = std::min(up_to(bytes, pool_on_ ? pg_ : grain_), m.reserved);
    }
#else
    bytes = std::min(up_to(bytes, pool_on_ ? pg_ : grain_), m.reserved);
#endif
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    piece = std::max(grain_, piece / grain_ * grain_);
    while (m.mapped < bytes) {
        const size_t n = std::min(bytes - m.mapped, piece);
        CUmemGenericAllocationHandle h = 0;
        // A handle of the pool when it has one.  Before fill_expert_cache() claims the pool, or in a model that has no
        // expert cache to claim it, a map with the pool empty still creates its handle, of the pool's size, and the
        // handle joins the pool when it is unmapped; once the pool is claimed an empty one means the card has no more.
        if (pool_on_ && !vpool_.empty()) {
            h = vpool_.back();
            vpool_.pop_back();
        } else if (pool_on_ && pool_claimed_) {
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        } else if (cuMemCreate(&h, n, &prop, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        }
        const CUdeviceptr at = m.base + m.mapped;
        const CUresult mapped = cuMemMap(at, n, 0, h, 0);
        if (mapped != CUDA_SUCCESS) {
            if (pool_on_) vpool_.push_back(h);
            else cuMemRelease(h);
        }
        cu_ck(mapped, "mapping VRAM");
        m.pieces.emplace_back(h, n);
        m.mapped += n;
        const CUresult acc = cuMemSetAccess(at, n, &access, 1);
        if (acc != CUDA_SUCCESS) {   // HIP hands out the physical pages here, not at create: the card may not have
            cuMemUnmap(at, n);       // them even though the addresses do.  Leave m as it was and say no.
            if (pool_on_) vpool_.push_back(h);
            else cuMemRelease(h);
            m.pieces.pop_back();
            m.mapped -= n;
            (void) cudaGetLastError();   // HIP's failed SetAccess leaves a sticky error: do not let it escape with the throw
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        }
        vmm_settle();   // on Linux ROCm the piece's pages are there once this returns, not before
        // A handle the pool kept holds what its last map left in it, expert bytes or old rows, where a created one
        // comes zeroed from the system.  Rows nothing has written yet are read by the kernels' tiles (the causal mask
        // does not stop a NaN from them), so a piece is zeroed as it is mapped.  Not the staging buffer: its uploads
        // run on another stream, and everything it holds is written before it is read.
        if (pool_on_ && &m != &stage_map_) zero_vram((void*) at, n);
    }
}

// Gives back the pieces of m that lie past its first `bytes`.  The card must be done with them.
void Engine::vmm_unmap(Mapped& m, size_t bytes) {
    const size_t had = m.mapped;
    while (!m.pieces.empty() && m.mapped - m.pieces.back().second >= bytes) {
        const auto [h, n] = m.pieces.back();
        m.mapped -= n;
        m.pieces.pop_back();
        cu_ck(cuMemUnmap(m.base + m.mapped, n), "unmapping VRAM");
        if (pool_on_) vpool_.push_back(h);   // kept: the next map of any region takes it
        else cu_ck(cuMemRelease(h), "unmapping VRAM");
    }
    if (pool_on_) return;   // nothing went back to the card, so there are no addresses to free for it either
#ifdef STRATA_VMM_RELEASE_WITH_ADDRESSES
    // On ROCm the released pieces stay the process's while their addresses are reserved (an RX 9070 XT: 320 MiB
    // unmapped and released, 32 MiB more free; the addresses freed, all of it).  The kept pieces leave their
    // addresses, which are freed and reserved again where they were, and come back with their bytes.  Once the first
    // has left, the slots, the recorded launches and m point at VRAM that is not there: a step that fails from
    // there on ends the engine (the server starts it again for the next request) rather than reading that VRAM.
    if (m.mapped == had || !m.base) return;
    const auto must = [](CUresult r, const char* what) {
        if (r == CUDA_SUCCESS) return;
        const char* s = nullptr;
        (void) cuGetErrorString(r, &s);
        std::fprintf(stderr, "strata-poly: %s while giving VRAM back: %s; the caches are no longer mapped, so the "
                             "engine stops\n", what, s ? s : "unknown error");
        std::abort();
    };
    for (size_t at = m.mapped, i = m.pieces.size(); i > 0; --i) {
        at -= m.pieces[i - 1].second;
        must(cuMemUnmap(m.base + at, m.pieces[i - 1].second), "unmapping the kept VRAM");
    }
    size_t free_b = 0, total_b = 0;
    (void) cudaMemGetInfo(&free_b, &total_b);
    const CUdeviceptr was = m.base;
    must(cuMemAddressFree(m.base, m.reserved), "freeing the addresses");
    CUdeviceptr p = 0;
    must(cuMemAddressReserve(&p, m.reserved, 0, was, 0), "reserving the addresses again");
    if (p != was) {   // the slots and the recorded launches point into the old addresses
        std::fprintf(stderr, "strata-poly: the VRAM addresses could not be reserved again where they were, so the "
                             "engine stops\n");
        std::abort();
    }
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    size_t at = 0;
    for (const auto& [h, n] : m.pieces) {
        must(cuMemMap(m.base + at, n, 0, h, 0), "mapping VRAM again");
        must(cuMemSetAccess(m.base + at, n, &access, 1), "mapping VRAM again");
        at += n;
    }
    vmm_settle();   // the slots and the recorded launches point into these pieces: nothing may run before they are back
    // amdgpu counts the VRAM as free 2 to 5 ms later: a reading before that (mind_vram()'s log line, cover()) is
    // short by what was given back.  At most 100 ms, as another program may take VRAM meanwhile.
    const size_t back = (had - m.mapped) / 8 * 7, want = free_b + back;
    for (int i = 0; i < 100 && free_b < want; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        (void) cudaMemGetInfo(&free_b, &total_b);
    }
#else
    (void) had;
#endif
}

// For the destructor: what is left of m, and its addresses, without throwing.
void Engine::vmm_free(Mapped& m) {
    for (; !m.pieces.empty(); m.pieces.pop_back()) {
        m.mapped -= m.pieces.back().second;
        cuMemUnmap(m.base + m.mapped, m.pieces.back().second);
        cuMemRelease(m.pieces.back().first);
    }
    if (m.base) cuMemAddressFree(m.base, m.reserved);
    m.base = 0;
}

#if defined(STRATA_GPU_METAL)
namespace {
constexpr size_t kMetalHostReserve = size_t(4) << 30;
constexpr size_t kMetalGrowthMargin = size_t(512) << 20;
constexpr size_t kMetalRefillCushion = size_t(256) << 20;
}

std::optional<uint64_t> Engine::metal_host_available() const {
    return strata::core::available_host_bytes();
}

size_t Engine::metal_host_headroom() {
    const auto memory = metal_host_available();
    if (memory) return *memory > kMetalHostReserve ? size_t(*memory - kMetalHostReserve) : 0;
    if (!metal_host_warning_) {
        std::fprintf(stderr, "strata-poly: host memory is unavailable; balanced Metal uses the Metal budget only\n");
        metal_host_warning_ = true;
    }
    return SIZE_MAX;
}

void Engine::metal_sync_memory() {
    settle_swaps();
    for (void* s : {st_, up_, rs_, swap_st_})
        if (s) ck(cudaStreamSynchronize(cs(s)), "moving Metal allocations");
}

void Engine::metal_invalidate_graphs() {
    // Graphs own resources as well as raw argument pointers. Clear every sequence's recordings before
    // retiring an allocation; otherwise a parked sequence can replay old pointers or keep that memory alive.
    for (void* g : graphs_) ck(cudaGraphExecDestroy((cudaGraphExec_t) g), "invalidating Metal graphs");
    graphs_.clear();
    for (Layer& l : layers_) {
        l.heads.clear(); l.shared.clear(); l.tail.clear(); l.card_moe.clear(); l.rows_heads.clear();
        l.rows_tails.clear();
        l.card = -1;
    }
    passes_.clear();
    step_passes_.clear();
    step_used_.clear();   // (destroyed with graphs_ above)
    tail_ = tail_picks_ = nullptr;
    for (Seq& q : seqs_) {
        q.heads.clear(); q.passes.clear(); q.tail = q.tail_picks = nullptr;
    }
    recorded_ = false;
    // Force the next GPU-routed call to upload fresh expert addresses, including indirect dependencies.
    h_slots_.clear();
    if (d_slots_) {
        const void* empty = nullptr;
        ck(gpu::metal::register_indirect(d_slots_, &empty, 0), "clearing expert dependencies");
    }
}

void Engine::metal_resize_cache(size_t limit) {
    if (freeze_experts_) throw std::runtime_error("balanced Metal cannot resize a frozen expert cache");
    const size_t previous_limit = metal_cache_limit_;
    const int previous_reserve = opt_.vram_reserve_mib;
    metal_cache_limit_ = limit;
    try {
        resize_card_cache(previous_reserve);
    } catch (...) {
        metal_cache_limit_ = previous_limit;
        opt_.vram_reserve_mib = previous_reserve;
        throw;
    }
    metal_cache_limit_ = previous_limit;
    opt_.vram_reserve_mib = previous_reserve;
}

bool Engine::metal_mind_memory() {
    if (!opt_.gpu_experts || n_moe_ == 0 || freeze_experts_) return false;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "balanced Metal budget");
    const size_t reserve = size_t(std::max(reserve_mib_, 256)) << 20;
    const size_t held = metal_cache_bytes_;
    const size_t available = free_b > SIZE_MAX - held ? SIZE_MAX : free_b + held;
    size_t target = available > reserve ? available - reserve : 0;
    const auto host = metal_host_available();
    if (host) {
        const size_t host_total = *host > SIZE_MAX - held ? SIZE_MAX : size_t(*host) + held;
        target = std::min(target, host_total > kMetalHostReserve ? host_total - kMetalHostReserve : 0);
    } else {
        (void) metal_host_headroom(); // issue the one-time fallback diagnostic
    }
    if (target < held) {
        metal_growth_samples_ = 0;
        metal_resize_cache(target);
        return metal_cache_bytes_ != held;
    }
    // A refill leaves its own cushion. Require a useful net increase before rereading the whole cache.
    if (all_cached() || target - held < kMetalGrowthMargin + kMetalRefillCushion ||
        (host && *host < kMetalHostReserve + kMetalGrowthMargin)) {
        metal_growth_samples_ = 0;
        return false;
    }
    if (++metal_growth_samples_ < 2) return false;
    metal_growth_samples_ = 0;
    metal_resize_cache(target);
    return metal_cache_bytes_ != held;
}

void Engine::metal_cover(int64_t end) {
    if (end < 0 || end > opt_.max_context) throw std::runtime_error("the context is full");
    if (end <= kv_rows_) { kv_reach_ = std::max(kv_reach_, end); return; }
    const int64_t rows = kv_rows_for(end);
    struct Replacement { size_t index, capacity; void* p; };
    std::vector<Replacement> next;
    size_t peak = 0;
    for (size_t i = 0; i < bufs_.size(); ++i) {
        const Buf& b = bufs_[i];
        if (!b.per) continue;
        const size_t capacity = size_t((rows + b.per - 1) / b.per) * b.bytes;
        if (capacity <= b.capacity) continue;
        if (capacity > SIZE_MAX - peak) throw std::runtime_error("Metal attention cache size overflow");
        peak += capacity; // old buffers stay alive until every replacement has been copied and rebound
        next.push_back({i, capacity, nullptr});
    }
    if (next.empty()) { kv_rows_ = rows; kv_reach_ = std::max(kv_reach_, end); return; }
    metal_sync_memory();
    const auto headroom = [&] {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "growing Metal attention cache");
        const size_t reserve = size_t(std::max(reserve_mib_, 256)) << 20;
        return std::min(free_b > reserve ? free_b - reserve : 0, metal_host_headroom());
    };
    size_t room = headroom();
    bool reduced = false;
    if (room < peak && exp_) {
        if (freeze_experts_)
            throw KvFull("balanced Metal KV growth needs memory held by frozen experts; reduce context or unfreeze the cache");
        const size_t cut = peak - room;
        metal_resize_cache(metal_cache_bytes_ > cut ? metal_cache_bytes_ - cut : 0);
        reduced = true;
        room = headroom();
    }
    if (room < peak)
        throw KvFull("not enough unified memory to grow the Metal attention cache; existing rows are preserved");
    const auto release = [&](void* p) {
        if (!p) return;
        const auto it = std::find(dev_allocs_.begin(), dev_allocs_.end(), p);
        if (it != dev_allocs_.end()) dev_allocs_.erase(it);
        ck(cudaFree(p), "retiring a Metal attention cache");
    };
    const std::vector<Buf> old = bufs_;
    try {
        for (Replacement& r : next) {
            r.p = dmalloc(r.capacity, "growing the attention cache");
            // Preserve the entire old capacity: speculative draft rows can extend beyond pos_.
            ck(cudaMemcpyAsync(r.p, old[r.index].p, old[r.index].capacity, cudaMemcpyDeviceToDevice, cs(st_)),
               "copying the attention cache");
        }
        ck(cudaStreamSynchronize(cs(st_)), "copying the attention cache");
        metal_invalidate_graphs();
        for (const Replacement& r : next) { bufs_[r.index].p = r.p; bufs_[r.index].capacity = r.capacity; }
        again_ = 0;
        alloc_caches();
        again_ = SIZE_MAX;
    } catch (...) {
        // Rebind the old addresses before freeing replacements, including on an architecture rebind failure.
        bufs_ = old;
        again_ = 0;
        try { alloc_caches(); } catch (...) { again_ = SIZE_MAX; std::terminate(); }
        again_ = SIZE_MAX;
        (void) cudaStreamSynchronize(cs(st_));
        for (const Replacement& r : next) release(r.p);
        throw;
    }
    for (const Replacement& r : next) release(old[r.index].p);
    kv_rows_ = rows;
    kv_reach_ = std::max(kv_reach_, end);
    if (reduced && !freeze_experts_) {
        // Refill only after the old KV buffers have gone; failure must not invalidate committed rows.
        try { metal_resize_cache(SIZE_MAX); }
        catch (const std::exception& e) { std::fprintf(stderr, "strata-poly: Metal expert cache remains reduced: %s\n", e.what()); }
    }
}
#endif

void* Engine::kv_alloc(size_t bytes, int per) {
    if (again_ != SIZE_MAX) return bufs_.at(again_++).p;
    const int64_t cap_rows = opt_.kv_resident > 0 ? std::min<int64_t>(opt_.max_context, opt_.kv_resident) : opt_.max_context;
    const size_t all = (size_t) ((cap_rows + per - 1) / per) * bytes;
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced && opt_.kv_grow) {
        const size_t capacity = (size_t) ((kv_rows_for(1) + per - 1) / per) * bytes;
        void* p = dmalloc(capacity, "the initial attention cache");
        bufs_.push_back({p, bytes, per, capacity});
        return p;
    }
#endif
    void* p = nullptr;
    if (!vmm_ || !opt_.kv_grow) {
        p = dmalloc(all, "the attention cache");
    } else {
        Kv k;
        k.bytes = bytes;
        k.per = per;
#ifdef STRATA_VMM_EQUAL_PIECES
        // with equal pieces, a whole number of cover()'s (addresses only; at most a piece more is mapped at the whole
        // context): vmm_piece() takes the most units that divide the reservation, which for one of another size could
        // be one unit, every piece 2 MiB and a file descriptor on ROCm
        if (!pool_on_ && vmm_equal_pieces()) {
            const size_t unit = vmm_unit(grain_), piece = (size_t) ((kv_grow() + per - 1) / per) * bytes;
            vmm_reserve(k, up_to(all, up_to(std::min(piece, up_to(all, unit)), unit)));
        } else {
            vmm_reserve(k, all);
        }
#else
        vmm_reserve(k, all);
#endif
        kv_.push_back(std::move(k));
        p = (void*) kv_.back().base;
    }
    bufs_.push_back({p, bytes, per});
#if defined(STRATA_GPU_METAL)
    bufs_.back().capacity = all;
#endif
    return p;
}

void* Engine::seq_alloc(size_t bytes, const char* what) {
    if (again_ != SIZE_MAX) return bufs_.at(again_++).p;
    void* p = dmalloc(bytes, what);
    bufs_.push_back({p, bytes, 0});
#if defined(STRATA_GPU_METAL)
    bufs_.back().capacity = bytes;
#endif
    return p;
}

// The bound sequence's members to its place in seqs_, and the layers' launches recorded with its buffers.
void Engine::park_bound() {
    Seq& q = seqs_[(size_t) bound_];
    q.pos = pos_;
    q.kv_rows = kv_rows_;
    q.kv_reach = kv_reach_;
    q.ahead_pos = ahead_pos_;
    q.ahead_tok = ahead_tok_;
    q.state = state_;
    q.kv = std::exchange(kv_, {});
    q.bufs = std::exchange(bufs_, {});
    q.images = std::exchange(images_, {});
    q.heads.resize(layers_.size());
    for (size_t il = 0; il < layers_.size(); ++il) q.heads[il] = std::exchange(layers_[il].heads, {});
    q.passes = std::exchange(passes_, {});
    q.tail = std::exchange(tail_, nullptr);
    q.tail_picks = std::exchange(tail_picks_, nullptr);
    // The fp16 twin of the int8 caches is the sequence's too (Gemma 4's --kv int8).
    q.twin_k = std::exchange(twin_k_, {});
    q.twin_v = std::exchange(twin_v_, {});
    q.twin_mapped = std::exchange(twin_mapped_, 0);
}

// Sequence s's members in place of the ones park_bound() put away, and the architecture's pointers into its
// buffers: alloc_caches() again, handed the buffers it was given for s in the order it asked for them.
void Engine::take_bound(int s) {
    Seq& q = seqs_[(size_t) s];
    pos_ = q.pos;
    kv_rows_ = q.kv_rows;
    kv_reach_ = q.kv_reach;
    ahead_pos_ = q.ahead_pos;
    ahead_tok_ = q.ahead_tok;
    state_ = q.state;
    if (draft_) hn_ = state_ + rec_n_;
    kv_ = std::exchange(q.kv, {});
    bufs_ = std::exchange(q.bufs, {});
    images_ = std::exchange(q.images, {});
    ++images_set_;
    for (size_t il = 0; il < layers_.size() && il < q.heads.size(); ++il) layers_[il].heads = std::exchange(q.heads[il], {});
    passes_ = std::exchange(q.passes, {});
    tail_ = std::exchange(q.tail, nullptr);
    tail_picks_ = std::exchange(q.tail_picks, nullptr);
    // The fp16 twin of the int8 caches (Gemma 4's --kv int8) comes back with the sequence it belongs to.
    twin_k_ = std::exchange(q.twin_k, {});
    twin_v_ = std::exchange(q.twin_v, {});
    twin_mapped_ = std::exchange(q.twin_mapped, 0);
    bound_ = s;
    again_ = 0;
    alloc_caches();
    again_ = SIZE_MAX;
}

void Engine::hold(int s) {
    if (s == bound_) return;
    park_bound();
    take_bound(s);
}

void Engine::bind(int s) {
    if (s < 0 || s >= (int) seqs_.size()) throw std::logic_error("bind: no such sequence");
    if (win_ > 0) throw std::logic_error("bind: a window's tokens are not kept yet");
    hold(s);
    if (next_) next_->bind(s);
}

// The buffers of the two sequences were asked for in the same order with the same sizes.  Of a growing one the rows
// before the position are copied, of another one all of it.
void Engine::copy_seq(int from, int to) {
    if (from < 0 || from >= sequences() || to < 0 || to >= sequences()) throw std::logic_error("copy_seq: no such sequence");
    if (win_ > 0) throw std::logic_error("copy_seq: a window's tokens are not kept yet");
    if (from != to) {
        OnCard here(device_, split_);
        const int home = bound_;
        hold(from);
        const int64_t pos = pos_;
        const std::vector<Buf> src = bufs_;
        const float* state = state_;
        const std::vector<Image> images = images_;
        hold(to);
        if (pos > 0) cover(pos);
        for (size_t i = 0; i < bufs_.size(); ++i) {
            const Buf& b = src[i];
            const size_t bytes = b.per ? (size_t) ((pos + b.per - 1) / b.per) * b.bytes : b.bytes;
            if (bytes) ck(cudaMemcpyAsync(bufs_[i].p, b.p, bytes, cudaMemcpyDeviceToDevice, cs(st_)), "copy_seq");
        }
        ck(cudaMemcpyAsync(state_, state, state_n_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "copy_seq");
        ck(cudaStreamSynchronize(cs(st_)), "copy_seq");
        pos_ = pos;
        images_ = images;
        ++images_set_;
        ahead_pos_ = -1;
        hold(home);
    }
    if (next_) next_->copy_seq(from, to);
}

// The attention caches' VRAM for the rows below `end`: rows kv_grow() at a time, kv_ahead() more for the answer to go
// on.  The expert cache gives it up (shrink_cache()), as if the rows had been there when it was filled: the
// reserve stays for what is recorded later.  A quarter of the reserve stays free whatever else took VRAM since
// (mind_vram()).
void Engine::cover(int64_t end) {
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced && opt_.kv_grow) {
        metal_cover(end);
        return;
    }
#endif
    kv_reach_ = std::max(kv_reach_, end);
    if (kv_.empty() || end <= kv_rows_) return;
    if (opt_.kv_resident > 0 && kv_rows_ >= opt_.kv_resident) return;
    const int64_t rows = kv_rows_for(end);
    const auto size = [rows](const Kv& k) { return (size_t) ((rows + k.per - 1) / k.per) * k.bytes; };
    size_t more = 0, held = arena_.mapped + other_rows();
    for (const Kv& k : kv_) {
#ifdef STRATA_VMM_EQUAL_PIECES
        // what vmm_map() below maps: with equal pieces, whole pieces
        const size_t piece = pool_on_ ? pg_
                             : !vmm_equal_pieces()
                                 ? grain_
                                 : vmm_piece(k.reserved, k.pieces.empty() ? 0 : k.pieces.front().second, size(k),
                                             (size_t) ((kv_grow() + k.per - 1) / k.per) * k.bytes, vmm_unit(grain_));
        const size_t want = std::min(up_to(size(k), piece), k.reserved);
#else
        const size_t want = std::min(up_to(size(k), pool_on_ ? pg_ : grain_), k.reserved);
#endif
        if (want > k.mapped) more += want - k.mapped;   // trim_card_kv() leaves a piece it cuts through
        held += k.mapped;
    }
    if (exp_ && pool_on_) {
        // What the rows need beyond what the pool holds comes out of the expert cache, which gives its pieces to the
        // pool (shrink_cache()); nothing is asked of the card.  With the cache gone and the pool still short, the
        // maps below throw: the context does not fit the VRAM this engine has.
        if (more > vpool_bytes()) shrink_cache(more - vpool_bytes());
    } else if (exp_) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t quarter = (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
        size_t cut = held + more > budget_ ? held + more - budget_ : 0;
        if (free_b < more + quarter) cut = std::max(cut, more + quarter - free_b);
        if (cut) shrink_cache(cut);
    }
    // in pieces of kv_grow() rows, so that trim_card_kv() can give back what a long read mapped in one go
    for (Kv& k : kv_)
        vmm_map(k, size(k), (size_t) ((kv_grow() + k.per - 1) / k.per) * k.bytes, "the attention cache");
    kv_rows_ = rows;
}

int64_t Engine::kv_rows_for(int64_t end) const {
    int64_t cap = opt_.max_context;
    if (opt_.kv_resident > 0) cap = std::min<int64_t>(cap, opt_.kv_resident);
    return std::min<int64_t>(cap, (end + kv_ahead() + kv_grow() - 1) / kv_grow() * kv_grow());
}

// The smaller steps of batch slots are for cover(), which maps more pages of a reservation.  metal_cover() copies the
// sequence's caches into new buffers and drops every recording, and where memory is short it cuts the expert cache
// and reads it in again, or ends the step's answers (KvFull): a slot that starts at 1,024 rows grows in the middle
// of the answer to a 900-token prompt.  On Metal the slots keep kKvGrow's steps, as poly.10's did.
int64_t Engine::kv_grow() const {
#if defined(STRATA_GPU_METAL)
    return kKvGrow;
#else
    return opt_.batch > 0 ? kKvGrowBatch : kKvGrow;
#endif
}

int64_t Engine::kv_ahead() const {
#if defined(STRATA_GPU_METAL)
    return kKvAhead;
#else
    return opt_.batch > 0 ? kKvAheadBatch : kKvAhead;
#endif
}

size_t Engine::trim_kv(int64_t end) {
    size_t freed = 0;
    for (Engine* e = this; e; e = e->next_.get()) freed += e->trim_card_kv(end);
    return freed;
}

// A card that holds every expert keeps the whole context's rows (load()): nothing would take what they give back.
size_t Engine::trim_card_kv(int64_t end) {
    const int64_t before = kv_reach_;
    kv_reach_ = end;
    if (kv_.empty() || !exp_ || all_cached()) return 0;
    const int64_t rows = kv_rows_for(std::max(end, before));
    if (rows >= kv_rows_) return 0;
    OnCard here(device_, split_);
    ck(cudaDeviceSynchronize(), "the attention cache");
    size_t freed = 0;
    for (Kv& k : kv_) {
        const size_t had = k.mapped;
        vmm_unmap(k, (size_t) ((rows + k.per - 1) / k.per) * k.bytes);
        freed += had - k.mapped;
    }
    kv_rows_ = rows;
    return freed;
}

// Every expert on its way into the cache arrives and takes its slot.
void Engine::settle_swaps() {
    if (n_swap_ == 0) return;
    const auto failed = [&] {
        std::lock_guard<std::mutex> l(swap_m_);
        return !swap_error_.empty();
    };
    for (int j = 0; j < n_swap_; ++j)
        while (swap_[(size_t) j].at.load() == Swap::kAsked && !failed()) std::this_thread::yield();
    if (failed()) throw std::runtime_error("expert upload: " + swap_error_);
    ck(cudaStreamSynchronize(cs(swap_st_)), "expert upload");
    follow(0);
}

// The expert cache gives up `bytes` or a little more from its end, and loads nothing again: the experts whose
// slots lie there and that were used more than cached ones before the cut take those ones' slots, as follow()
// ranks them, and the others leave the cache.
void Engine::shrink_cache(size_t bytes) {
    // STRATA_FREEZE_EXPERTS: a frozen answer reads a cache that never changes.  shrink_cache() would move
    // experts between slots mid-answer (cover() calls it when the attention caches grow, and mind_vram() when
    // another program takes VRAM), which is exactly the change the freeze is there to keep out.  A frozen
    // answer's cache is the one its first token saw; the request after it may size it again.
    if (freeze_experts_) return;
    size_t cut = arena_.mapped;
    for (size_t i = arena_.pieces.size(); i > 0 && arena_.mapped - cut < bytes; --i) cut -= arena_.pieces[i - 1].second;
    settle_swaps();
    fill_now();
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    const size_t ne = (size_t) n_expert_, end = cut > kTail ? cut - kTail : 0;   // kTail stays past the last slot
    struct Held {
        float recent;
        int64_t prior;
        int il, e;
    };
    const auto held = [&](int il, int e) {
        const size_t i = (size_t) il * ne + (size_t) e;
        return Held{recent_.empty() ? 0.f : recent_[i], (prior_.empty() ? 0 : prior_[i]) + use_[i], il, e};
    };
    const auto more = [](const Held& a, const Held& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    };
    std::vector<Held> past;
    std::vector<std::vector<Held>> low((size_t) n_trade_);   // the cached ones before the cut, by slot size
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        for (int e = 0; e < n_expert_; ++e) {
            const uint8_t* s = L.slot[(size_t) e];
            if (!s) continue;
            if ((size_t) (s - exp_) + L.slot_bytes > end) past.push_back(held(il, e));
            else low[(size_t) L.trade].push_back(held(il, e));
        }
    }
    std::sort(past.begin(), past.end(), more);
    for (std::vector<Held>& l : low) std::sort(l.begin(), l.end(), [&](const Held& a, const Held& b) { return more(b, a); });
    std::vector<size_t> next((size_t) n_trade_, 0);
    for (const Held& x : past) {
        const Layer& L = layers_[(size_t) x.il];
        uint8_t*& from = layers_[(size_t) x.il].slot[(size_t) x.e];
        const std::vector<Held>& l = low[(size_t) L.trade];
        size_t& k = next[(size_t) L.trade];
        if (k < l.size() && more(x, l[k])) {
            uint8_t*& to = layers_[(size_t) l[k].il].slot[(size_t) l[k].e];
            mirror_keep(l[k].il, l[k].e, to);   // before x's copy takes its slot
            ck(cudaMemcpyAsync(to, from, L.slot_bytes, cudaMemcpyDeviceToDevice, cs(st_)), "the expert cache");
            from = to;
            to = nullptr;
            ++k;
        } else {
            mirror_keep(x.il, x.e, from);
            from = nullptr;
        }
        --cached_;
    }
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    vmm_unmap(arena_, cut);
    if (cut > kTail) zero_vram(exp_ + end, kTail);
    exp_at_ = std::min(exp_at_, end);
    staged_il_ = -1;
}

float* Engine::fbuf(size_t n) {
    float* p = scratch_ ? scratch_ + scratch_at_ : nullptr;
    scratch_at_ += (n + 63) & ~size_t(63);
    return p;
}

std::vector<std::pair<const uint8_t*, size_t>> Engine::model_views() const {
    return gg_ ? gg_->views() : std::vector<std::pair<const uint8_t*, size_t>>{};
}

int64_t Engine::meta_u(const char* key, int64_t fallback) const {
    const MetaValue* v = gg_ ? gg_->get(key) : nullptr;
    return v && v->is_num() ? (int64_t) v->u : fallback;
}

// A layer's tensor, under any of its names.  The draft block's are in the file it comes from.
const TensorInfo* Engine::tensor(int il, const char* name) const {
    const Shards& g = block_ && il >= n_layer_ ? *block_ : *gg_;
    if (const TensorInfo* t = g.find(blk(il, name))) return t;
    if (const Names* n = tensor_alias(name))
        for (const char* other : n->other)
            if (const TensorInfo* t = other ? g.find(blk(il, other)) : nullptr) return t;
    return nullptr;
}

// A value of the metadata of the model that `g` is a file of, under any of the names converters give it.
const MetaValue* Engine::meta_in(const Shards& g, const char* key) const {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return nullptr;
    const std::string pre = arch->s + ".";
    if (const MetaValue* v = g.get(pre + key)) return v;
    if (const Names* n = meta_alias(key))
        for (const char* other : n->other)
            if (const MetaValue* v = other ? g.get(pre + other) : nullptr) return v;
    return nullptr;
}

int Engine::shape_i(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (int) v->u;
}

float Engine::shape_f(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (float) v->f;
}

// What every architecture's file says of the model: its layers, its width, its vocabulary and its experts.
//
// The block that guesses a second token after the model's own is one more block after the model's.  It is the
// model's and not a file's: a file that leaves it out can take it from another file of the model (draft_model).
// One converter gives the block a token table or an output matrix of its own, which this engine does not run.
void Engine::load_shape() {
    const MetaValue* nextn = meta_in(*gg_, "nextn_predict_layers");
    n_layer_ = shape_i("block_count") - (nextn ? (int) nextn->u : 0);
    if (opt_.draft && opt_.window != 1) {
        const std::string probe = blk(n_layer_, "nextn.eh_proj.weight");
        if (nextn && nextn->u == 1 && gg_->find(probe)) block_ = gg_.get();
        else if (dg_ && dg_->find(probe)) block_ = dg_.get();
        else if (dg_)
            throw std::runtime_error(opt_.draft_model + " has no block that guesses ahead for this model: no tensor " +
                                     probe);
    }
    draft_ = block_ && !tensor(n_layer_, "nextn.embed_tokens.weight") &&
             !tensor(n_layer_, "nextn.shared_head_head.weight");
    if (!draft_) block_ = nullptr;
    block_layer_ = draft_;
    n_embd_ = shape_i("embedding_length");
    // a file without the key has as many tokens as its token table
    const MetaValue* vocab = meta_in(*gg_, "vocab_size");
    const TensorInfo* te = gg_->find("token_embd.weight");
    n_vocab_ = vocab ? (int) vocab->u : te && te->shape.size() == 2 ? (int) te->shape[1] : 0;
    {   // the tokens' bytes, for a grammar (src/poly/grammar.hpp); a model whose tokenizer is not byte-level BPE has none
        const MetaValue* model = gg_->get("tokenizer.ggml.model");
        if (model && model->type == MetaType::STRING)
            token_texts_ = gpt2_token_texts(model->s, gg_->first().string_array("tokenizer.ggml.tokens"),
                                            gg_->first().int_array("tokenizer.ggml.token_type"), n_vocab_);
    }
    rms_eps_ = shape_f("attention.layer_norm_rms_epsilon");
    n_expert_ = shape_i("expert_count");
    n_used_ = shape_i("expert_used_count");
    n_ff_exp_ = shape_i("expert_feed_forward_length");
    const MetaValue* shexp = meta_in(*gg_, "expert_shared_feed_forward_length");
    n_ff_shexp_ = shexp ? (int) shexp->u : n_ff_exp_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");

    // a block from another file must be this model's: a pruned model's has fewer experts to choose among
    if (block_ && block_ != gg_.get()) {
        const TensorInfo* gate = tensor(n_layer_, "ffn_gate_inp.weight");
        if (!gate || gate->shape.size() != 2 || (int64_t) gate->shape[0] != n_embd_ ||
            (int64_t) gate->shape[1] != n_expert_)
            throw std::runtime_error(opt_.draft_model + " is not a file of this model: its block's router is " +
                                     (gate ? shape_of(*gate) : std::string("missing")) + ", the model's layers have [" +
                                     std::to_string(n_embd_) + ", " + std::to_string(n_expert_) + "]");
    }
}

// A MoE layer: its router and shared expert to VRAM, and its routed experts' places in the file, where they
// stay; fill_expert_cache() copies the ones VRAM has room for.  A file may keep an expert's gate and up in one
// tensor, the gate's rows first.
void Engine::load_moe(int il, bool shared) {
    Layer& L = layers_[(size_t) il];
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    L.moe = true;
    L.gate_inp = upload(tensor(il, "ffn_gate_inp.weight"), blk(il, "ffn_gate_inp.weight"), n_embd_, n_expert_,
                        router_bf16() ? kExactBF16 : kAnyMat);
    if (shared) {
        L.sh_gate = mat("ffn_gate_shexp.weight", n_embd_, n_ff_shexp_);
        L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
        L.sh_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    }

    auto exps = [&](const char* n, int64_t n_in, int64_t rows) -> const TensorInfo& {
        const TensorInfo* t = tensor(il, n);
        if (!t) throw std::runtime_error("the GGUF has no tensor " + blk(il, n));
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != rows ||
            (int64_t) t->shape[2] != n_expert_)
            throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                     std::to_string(rows) + ", " + std::to_string(n_expert_) + "]");
        return *t;
    };
    const TensorInfo& td = exps("ffn_down_exps.weight", n_ff_exp_, n_embd_);
    L.d_type = td.type;
    L.d_bytes = home(td).payload(td) / (size_t) n_expert_;
    L.exp_down = home(td).data(td);
    if (tensor(il, "ffn_gate_up_exps.weight")) {
        const TensorInfo& gu = exps("ffn_gate_up_exps.weight", n_embd_, 2 * n_ff_exp_);
        L.gu_type = gu.type;
        L.gu_stride = home(gu).payload(gu) / (size_t) n_expert_;
        L.gu_bytes = L.gu_stride / 2;
        L.exp_gate = home(gu).data(gu);
        L.exp_up = L.exp_gate + L.gu_bytes;
        L.u_type = L.gu_type;
        L.u_bytes = L.gu_bytes;
        L.u_stride = L.gu_stride;
    } else {
        const TensorInfo& tg = exps("ffn_gate_exps.weight", n_embd_, n_ff_exp_);
        const TensorInfo& tu = exps("ffn_up_exps.weight", n_embd_, n_ff_exp_);
        L.gu_type = tg.type;
        L.gu_stride = L.gu_bytes = home(tg).payload(tg) / (size_t) n_expert_;
        L.u_type = tu.type;
        L.u_stride = L.u_bytes = home(tu).payload(tu) / (size_t) n_expert_;
        L.exp_gate = home(tg).data(tg);
        L.exp_up = home(tu).data(tu);
        // up starts gu_bytes into a slot, which keeps every slot's blocks at 8 bytes (see the slot sizes)
        if (L.mixed() && L.gu_bytes % 8 != 0)
            throw std::runtime_error(tg.name + " and " + tu.name + " have different types and the gate of an expert is "
                                     + std::to_string(L.gu_bytes) + " bytes, not a multiple of 8");
    }
    L.gu_step = L.gu2();
    if (L.mixed()) {   // Flint NanoPlus: 335,872 + 270,336 bytes, whole blocks of neither IQ2_S (82) nor IQ2_XXS (66)
        const size_t unit = std::lcm(std::lcm(type_block_bytes(L.gu_type), type_block_bytes(L.u_type)),
                                     size_t(8));
        L.gu_step = (L.gu_step + unit - 1) / unit * unit;
    }
    // NVFP4 experts converted from NVIDIA's checkpoints carry each expert's global scale of gate, up and down
    // (llama.cpp's ffn_*_exps_s).  Gate's and up's go on their products ahead of the activation (ExpertScales),
    // down's on the expert's weight in its token's sum.  Only NVFP4's: Gemma 4's ffn_down_exps.scale is its own.
    const bool nv = L.gu_type == kTypeNVFP4 || L.u_type == kTypeNVFP4 || L.d_type == kTypeNVFP4;
    if (nv && !joined_scales_ && tensor(il, "ffn_gate_up_exps.scale"))
        throw std::runtime_error(blk(il, "ffn_gate_up_exps.scale: NVFP4 scales of a joined gate and up are not supported"));
    // A projection of another type in the same layer has none (Qwen3.6 NVFP4-MTP-HQ: some ups are Q4_K).
    const auto one = [&](const char* n, uint32_t type, Dev& d, std::vector<float>& h) {
        const TensorInfo* t = tensor(il, n);
        h.clear();
        if ((type != kTypeNVFP4 && !joined_scales_) || !t) return;
        if (t->type != kTypeF32) throw std::runtime_error(t->name + " is not F32");
        d = upload(t, blk(il, n), n_expert_, 1, kF32);
        h.resize((size_t) n_expert_);
        std::memcpy(h.data(), home(*t).data(*t), h.size() * sizeof(float));
    };
    if (joined_scales_ && tensor(il, "ffn_gate_up_exps.weight")) {
        // one scale an expert on the joined product: its gate's and its up's
        one("ffn_gate_up_exps.scale", L.gu_type, L.s_gate, L.h_s_gate);
        L.s_up = L.s_gate;
        L.h_s_up = L.h_s_gate;
    } else {
        one("ffn_gate_exps.scale", L.gu_type, L.s_gate, L.h_s_gate);
        one("ffn_up_exps.scale", L.u_type, L.s_up, L.h_s_up);
    }
    one("ffn_down_exps.scale", L.d_type, L.s_down, L.h_s_down);
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_, err) ||
        !cpu_expert_types_ok((int) L.u_type, (int) L.d_type, n_embd_, n_ff_exp_, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) n_expert_, nullptr);
    ++n_moe_;
}

// The engine for a model file: its general.architecture says which.
std::unique_ptr<Engine> Engine::open(const Options& o) {
    const Shards file(o.model);
    const MetaValue* arch = file.get("general.architecture");
    const std::string name = arch ? arch->s : std::string();
    // Clover is a qwen35moe model trained to also guess a block of tokens at once (Qwen35)
    const bool qwen35 = name == "qwen35moe" || name == "clover_moe";
    if (o.kv_format == KvFmt::kQ8_0 && !qwen35 && name != "kolibri1" && name != "flint" && name != "lfm2moe" &&
        name != "laguna" && name != "inkling2_loop" &&
        name != "inkling2")
        throw std::runtime_error("--kv q8_0 is supported for qwen35moe, kolibri1, flint, lfm2moe, laguna and "
                                 "Inkling 2 models in v1; this is a '" + name + "' model");
    // q4_0 is Qwen35's (the other architectures read a q8_0 cache, not this one, and would run fp16 without a word)
    if (o.kv_format == KvFmt::kQ4_0 && !qwen35)
        throw std::runtime_error("--kv q4_0 is supported for qwen35moe models in v1; this is a '" + name + "' model");
    // --kv-resident streams the attention caches from host RAM: Qwen35's (kv_stage_, the pinned rows)
    if (o.kv_resident > 0 && !qwen35)
        throw std::runtime_error("--kv-resident is supported for qwen35moe models in v1; this is a '" + name + "' model");
#if defined(STRATA_GPU_METAL)
    // their kernels (store_q4, dequant_q4_to_q8, merge_attn_piece) are CUDA and HIP only (metal/src/poly/unported.cpp)
    if (o.kv_format == KvFmt::kQ4_0 || o.kv_resident > 0)
        throw std::runtime_error("--kv q4_0 and --kv-resident have no Metal kernels yet");
#endif
    // Inkling 2: the inkling2_loop files llama.cpp runs, or Unsloth's own layout of the same weights ("inkling2")
    const bool inkling = name == "inkling2_loop" || name == "inkling2";
    if (o.kv_format == KvFmt::kFp8 && !inkling)
        throw std::runtime_error("--kv fp8 is Inkling 2's; this is a '" + name + "' model");
    // int8 and int8k are Gemma 4's: their readers are not the arithmetic q8_0's are, so a model takes one
    // family or the other rather than a choice between them.
    if (name != "gemma4" && (o.kv_format == KvFmt::kInt8 || o.kv_format == KvFmt::kInt8K))
        throw std::runtime_error("--kv int8 and --kv int8k are Gemma 4's; for a qwen35moe or kolibri1 model use "
                                 "--kv q8_0 (this is a '" + name + "' model)");
    // route_cached(): the engine's is qwen35moe's router (softmax weights); Inkling 2 has its own
    if (o.cache_bias > 0.f && !qwen35 && !inkling)
        throw std::runtime_error("--cache-bias is supported for qwen35moe and Inkling 2 models; this is a '" + name +
                                 "' model");
    if ((!o.attn_requant.empty() || o.loop_bf16 || o.prompt_int8 >= 0) && !inkling)
        throw std::runtime_error("--attn-requant, --loop-bf16 and --prompt-int8 / --no-prompt-int8 are Inkling 2's; this is a '" + name + "' model");
    if (o.expert_gpu >= 0 && (!inkling || !o.layer_split.empty()))
        throw std::runtime_error("--expert-gpu is Inkling 2's, on one card beside it (no --layer-split); this is a '" +
                                 name + "' model");
    if (!o.attn_requant.empty() && o.attn_requant != "q4_0" && o.attn_requant != "q5_0" && o.attn_requant != "iq4_nl")
        throw std::runtime_error("--attn-requant takes q4_0, q5_0 or iq4_nl");
    const auto make = [&]() -> std::unique_ptr<Engine> {
        if (name == "glm5-next" || name == "glm5next") return std::make_unique<Glm5Next>();   // Unsloth: no hyphen
        if (qwen35) return std::make_unique<Qwen35>();
        if (name == "gemma4") return std::make_unique<Gemma4>();
        if (name == "deepseek4") return std::make_unique<DeepSeek4>();
        if (name == "lfm2moe") return std::make_unique<Lfm2Moe>();
        if (name == "flint") return std::make_unique<Flint>();
        if (name == "kolibri1") return std::make_unique<Kolibri1>();
        if (inkling) return std::make_unique<Inkling2>();
#if !defined(STRATA_GPU_METAL)
        if (name == "laguna") return std::make_unique<Laguna>();
#endif
        throw std::runtime_error(o.model + " is a '" + name + "' model, which this engine does not run");
    };
    int cards = 1, devices = 1;
    if (!o.layer_split.empty()) ck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    cards = devices;
    // STRATA_POLY_CARDS=N: a split's N parts on the devices there are, in turn, so that a split is tried on one card
    if (const char* v = std::getenv("STRATA_POLY_CARDS"); v && !o.layer_split.empty()) cards = std::max(1, std::atoi(v));
    std::vector<int> starts;
    bool swap = false;   // "auto" put the first layers on the second device (layer_starts())
    if (cards > 1) {
        const MetaValue* blocks = file.get(name + ".block_count");
        const MetaValue* nextn = file.get(name + ".nextn_predict_layers");
        if (!blocks) throw std::runtime_error("the GGUF has no " + name + ".block_count");
        // Inkling 2's loop runs on one card: a later pass of it runs the core's layers again (Inkling2::next_layer())
        int keep_lo = 0, keep_hi = 0, loop_runs = 1;
        if (name == "inkling2_loop") {
            const MetaValue* lo = file.get(name + ".loop.start");
            const MetaValue* hi = file.get(name + ".loop.end");
            const MetaValue* count = file.get(name + ".loop.count");
            if (!lo || !hi) throw std::runtime_error("unsupported model: no " + name + ".loop.start and .loop.end");
            keep_lo = (int) lo->u;
            keep_hi = (int) hi->u;
            loop_runs = count ? (int) count->u : 1;
        }
        int was = 0;
        ck(cudaGetDevice(&was), "cudaGetDevice");
        starts = layer_starts(o.layer_split, file, (int) blocks->u - (nextn ? (int) nextn->u : 0), cards, keep_lo,
                              keep_hi, loop_runs, &swap);
        ck(cudaSetDevice(was), "cudaSetDevice");
    }
    if (starts.empty()) {
        std::unique_ptr<Engine> engine = make();
        engine->load(o);
        return engine;
    }

    // a card after another, each with the next run of layers; the last one has the output and the draft block
    std::unique_ptr<Engine> first;
    Engine* prev = nullptr;
    for (size_t k = 0; k < starts.size(); ++k) {
        std::unique_ptr<Engine> e = make();
        const bool last = k + 1 == starts.size();
        Options so = o;
        if (!last) {
            so.draft = false;
            so.draft_model.clear();
        }
        if (first) {
            so.window = first->opt_.window;   // every card's windows are the first one's
            e->workers_ = first->workers_;
        }
        e->lo_ = starts[k];
        e->end_ = last ? -1 : starts[k + 1];
        e->split_ = true;
        ck(cudaSetDevice((swap ? 1 - (int) k : (int) k) % devices), "cudaSetDevice");
        try {
            e->load(so);
        } catch (const std::runtime_error& err) {
            throw std::runtime_error("card " + std::to_string(k) + ": " + err.what());
        }
        Engine* at = e.get();
        if (prev) prev->next_ = std::move(e);
        else first = std::move(e);
        prev = at;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    int k = 0;
    for (const Engine* e = first.get(); e; e = e->next_.get(), ++k)
        std::fprintf(stderr, "strata-poly: layers %d to %d on card %d: %d of %d routed experts in VRAM (%.1f GiB)\n",
                     e->lo_, e->hi_ - 1, swap ? e->device_ : k, e->cached_, e->n_moe_ * e->n_expert_,
                     double(e->exp_at_) / double(1ull << 30));
    return first;
}

// The file a tensor is in.
const Shards& Engine::home(const TensorInfo& t) const {
    return block_ && gg_->find(t.name) != &t ? *block_ : *gg_;
}

// A tensor to VRAM.  It goes as the file has it when the kernels compute with that type.  Otherwise it is
// converted on the way: to F32 where a kernel reads floats, to BF16 for a matrix.  `fix` is applied to every
// value, which also takes the float path.  While dense_ is null the call only adds up the arena (load() runs the
// model's tensors through twice).
Dev Engine::upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
                   float (*fix)(float)) {
    Dev d = upload_as(t, name, n_in, n_out, want, fix);
    d.scale = global_scale(*t);
    if (dense_ && d.type != kTypeF32 && d.type != kTypeBF16) mats_.push_back(d);   // (the second pass's)
    return d;
}

// An NVFP4 matrix converted from NVIDIA's checkpoints has its global scale beside it, "<name>.scale" for
// "<name>.weight" (llama.cpp's *_s tensors, which multiply the product): about 1e-4, so it cannot go into the
// blocks' E4M3 scales.  1 for any other matrix.
float Engine::global_scale(const TensorInfo& t) const {
    static const std::string w = ".weight";
    if (t.type != kTypeNVFP4 || t.name.size() <= w.size() ||
        t.name.compare(t.name.size() - w.size(), w.size(), w) != 0)
        return 1.0f;
    const TensorInfo* s = home(t).find(t.name.substr(0, t.name.size() - w.size()) + ".scale");
    if (!s || s->type != kTypeF32 || s->elements() != 1) return 1.0f;
    float v;
    std::memcpy(&v, home(*s).data(*s), sizeof v);
    return v;
}

bool Engine::exact_bf16(const TensorInfo* t) const {
    if (!t || t->type != kTypeF32) return false;
    const uint32_t* v = (const uint32_t*) home(*t).data(*t);
    for (size_t i = 0, n = (size_t) t->elements(); i < n; ++i)
        if (v[i] & 0xffffu) return false;
    return true;
}

Dev Engine::upload_as(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
                      float (*fix)(float)) {
    if (!t) throw std::runtime_error("the GGUF has no tensor " + name);
    if (want == kExactBF16 && (fix || !exact_bf16(t))) want = kAnyMat;
    const uint64_t rest = t->elements() / std::max<uint64_t>(t->shape[0], 1);
    if ((int64_t) t->shape[0] != n_in || (int64_t) rest != n_out)
        throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                 std::to_string(n_out) + "]");
    const size_t bytes = home(*t).payload(*t), count = (size_t) t->elements();
    // Once the matrix is on the card the CPU does not read it again: its pages leave the working set when this
    // returns.  Left until the end of the load, they and the experts the card took were the process's peak, 15 GB
    // of Qwen3.6 UD-Q3_K_M's 15.9 at any VRAM.
    struct Drop {
        const uint8_t* p;
        size_t n;
        ~Drop() {
            if (p) drop_pages(p, n);
        }
    } drop{dense_ ? (const uint8_t*) home(*t).data(*t) : nullptr, bytes};
    if (want == kGuessMat && !fix && n_in % type_block_values(guess_type_) == 0 && type_converts(t->type)) {
        const size_t row = strata::quant::row_bytes((int) guess_type_, n_in), out = row * (size_t) n_out;
        const Dev d{dense_ ? dense_ + dense_at_ : nullptr, guess_type_, (int) n_in, (int) n_out};
        void (*quantize)(const float*, void*, int64_t) = guess_type_ == 6    ? strata::quant::quantize_q5_0_ref
                                                         : guess_type_ == 20 ? strata::quant::quantize_iq4_nl_ref
                                                                             : strata::quant::quantize_q4_0_ref;
        dense_at_ += pad(out) + kTail;
        if (!dense_) return d;
        if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
        // a few thousand rows at a time, to floats and to Q4_0, on threads of their own: the engine's are not
        // started yet
        const size_t in_row = bytes / (size_t) n_out, rows = 4096;
        std::vector<uint8_t> q(out);
        std::vector<float> f(rows * (size_t) n_in);
        const uint8_t* src = (const uint8_t*) home(*t).data(*t);
        Workers pool(std::max(1u, std::thread::hardware_concurrency() / 2));
        for (size_t r0 = 0; r0 < (size_t) n_out; r0 += rows) {
            const size_t m = std::min(rows, (size_t) n_out - r0);
            pool.run((int) ((m + 255) / 256), [&](int k) {
                const size_t a = (size_t) k * 256, b = std::min(m, a + 256);
                type_to_floats(t->type, src + (r0 + a) * in_row, f.data() + a * n_in, (int64_t) ((b - a) * n_in));
                quantize(f.data() + a * n_in, q.data() + (r0 + a) * row, (int64_t) ((b - a) * n_in));
            });
        }
        h2d(d.p, q.data(), out);
#if defined(STRATA_GPU_METAL)
        if (opt_.metal_balanced) drop_file_pages(src, bytes);
#endif
        return d;
    }
    const bool as_is = !fix && ((t->type == kTypeF32 && want != kExactBF16) ||
                                (want != kF32 && t->type == kTypeBF16) ||
                                (want == kAnyMat && kernels::native_mmvq_supported((int) t->type)));
    const uint32_t type = as_is ? t->type : want == kF32 ? kTypeF32 : kTypeBF16;
    const size_t out = as_is ? bytes : count * (type == kTypeF32 ? sizeof(float) : sizeof(uint16_t));
    if (type == kTypeBF16) bf16_max_ = std::max(bf16_max_, count);
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, type, (int) n_in, (int) n_out};
    dense_at_ += pad(out) + kTail;
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
    if (as_is) {
        h2d(d.p, home(*t).data(*t), bytes);
#if defined(STRATA_GPU_METAL)
        if (opt_.metal_balanced) drop_file_pages(home(*t).data(*t), bytes);
#endif
        return d;
    }

    std::vector<float> f(count);
    if (t->type == kTypeF32) std::memcpy(f.data(), home(*t).data(*t), bytes);
    else if (type_converts(t->type)) type_to_floats(t->type, home(*t).data(*t), f.data(), (int64_t) count);
    else throw std::runtime_error(t->name + " is " + shape_of(*t) + ", a type this engine cannot read");
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced) drop_file_pages(home(*t).data(*t), bytes);
#endif
    if (fix)
        for (float& v : f) v = fix(v);
    if (type == kTypeF32) {
        h2d(d.p, f.data(), out);
        return d;
    }
    std::vector<uint16_t> h(count);
    for (size_t i = 0; i < count; ++i) h[i] = bf16_of(f[i]);
    h2d(d.p, h.data(), out);
    return d;
}

int64_t Engine::vram_free_mib() const {
    OnCard here(device_, split_);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const int64_t mib = (int64_t) (free_b >> 20);
    return next_ ? std::min(mib, next_->vram_free_mib()) : mib;
}

// Experts go to VRAM by use per byte when there are counts, the profile's and this run's, and otherwise the same
// number from every layer.
//
// The cache takes what is free but for a reserve.  What the engine allocates after it comes out of the reserve:
// the recorded launches and a library's work space took 180 MiB of an 8 GB card in the first request.  Unless the
// caller names it, the reserve is a thirty-second of the card, between 256 and 768 MiB (mind_vram() keeps what is
// free near half of it later), or a sixteenth when the machine's RAM is within 5 GiB of the model's files
// (default_reserve_mib()).  Under Windows a card that is asked for more than it has does not refuse: memory
// moves to the system's, and an answer came at 13 tokens/s where it comes at 30.
//
// The line was 4 GiB while the two staging chunks held 1 GiB of pinned memory by now; they hold 128 MiB, and
// the line moved with them, so that the same PC in the same state keeps the reserve it had.
void Engine::fill_expert_cache() {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const size_t ram = ram_available(), files = gg_->bytes() + (dg_ ? dg_->bytes() : 0);
    ram_tight_ = ram > 0 && files + (5ULL << 30) > ram;
    // 256 MiB at least, whatever was asked: the launches recorded later take theirs out of the reserve (mind_vram())
    reserve_mib_ = opt_.vram_reserve_mib >= 0 ? std::max(opt_.vram_reserve_mib, 256) :
#if defined(STRATA_GPU_METAL)
                   opt_.metal_balanced ? 256 :
#endif
                   default_reserve_mib(total_b, ram_tight_);
    size_t reserve = size_t(reserve_mib_) << 20;
    // The first time the pool claims what the card has but the margin (the reserve), for the whole run, and no more
    // than the model can use (below, where the experts' bytes are known): from then on the cache and the rows share it.
    // A later call (resize_cache()) divides what the pool holds again.
    const bool claim_later = pool_on_ && !pool_claimed_;
    // the staging buffer is mapped later in load(), from the pool: its narrow size stays out of the cache
    const size_t stage_slack = (size_t(opt_.stage_mib > 0 ? opt_.stage_mib : 128) + 8) << 20;
    if (pool_on_ && !claim_later) {
        free_b = vpool_bytes();
        reserve = stage_map_.base ? 0 : stage_slack;
    }
    // STRATA_GPU_EXPERTS: an N the VRAM cannot hold is refused rather than cached in part
    const auto short_of = [&](size_t room, size_t need) {
        if (expert_cap_ <= 0) return;
        throw std::runtime_error("STRATA_GPU_EXPERTS=" + std::to_string(expert_cap_) + ": the card has " +
                                 std::to_string(room >> 20) + " MiB for the expert cache" +
                                 (need ? ", the first " + std::to_string(expert_cap_) + " experts take " +
                                             std::to_string(need >> 20) + " MiB" : std::string()) +
                                 ": lower N or --vram-reserve-mib");
    };
    // STRATA_GPU_EXPERTS=0 caches nothing: its line (FNV's starting value, no pairs) is printed here, before the
    // returns below for a cache of no bytes, so that every N prints one
    if (expert_cap_ == 0) std::fprintf(stderr, "strata-poly: cache set 0 experts fnv %016llx\n", 14695981039346656037ULL);
    if (!claim_later && free_b <= reserve) return short_of(0, 0);

    struct Cand {
        int il, e;
        size_t bytes;
        double value;
    };
    std::vector<Cand> cands;
    std::vector<size_t> sizes;   // the slot sizes there are; a layer's place among them is its trade number
    size_t all = 0;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe || !kernels::native_mmvq_supported((int) L.gu_type) ||
            !kernels::native_mmvq_supported((int) L.u_type) || !kernels::native_mmvq_supported((int) L.d_type))
            continue;
        // A slot is a whole number of blocks of each of the experts' types, so the slots of a layer that lie one
        // behind the other are the matrices of one launch (moe_rows()), and of 8 bytes, which keeps every block
        // where its type may be read.
        const size_t unit = std::lcm(std::lcm(std::lcm(type_block_bytes(L.gu_type),
                                                       type_block_bytes(L.u_type)),
                                              type_block_bytes(L.d_type)),
                                     size_t(8));
        const size_t bytes = (L.gu2() + L.d_bytes + kTail + unit - 1) / unit * unit;
        L.slot_bytes = bytes;
        if (std::find(sizes.begin(), sizes.end(), bytes) == sizes.end()) sizes.push_back(bytes);
        L.trade = (int) (std::find(sizes.begin(), sizes.end(), bytes) - sizes.begin());
        slot_max_ = std::max(slot_max_, bytes);
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, bytes, used / (double) bytes});
            all += bytes;
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });
    if (expert_cap_ >= 0) {   // the first N and nothing but them: the cache is sized to hold them
        if ((size_t) expert_cap_ > cands.size())
            throw std::runtime_error("STRATA_GPU_EXPERTS=" + std::to_string(expert_cap_) + ": the model has " +
                                     std::to_string(cands.size()) + " experts the card can hold");
        cands.resize((size_t) expert_cap_);
        all = 0;
        for (const Cand& c : cands) all += c.bytes;
    }

    n_trade_ = (int) sizes.size();
    if (claim_later) {
        // What the engine can use is every expert's slot, the attention caches' whole address range and 512 MiB for the
        // staging buffer and the rounding: a card that holds more than that keeps the rest for its other programs.
        size_t kv_total = 0;
        for (const Kv& k : kv_) kv_total += k.reserved;
        for (const Seq& q : seqs_)
            for (const Kv& k : q.kv) kv_total += k.reserved;
        const size_t limit = all + 256 + kv_total + (size_t(512) << 20);
        // The card's free VRAM bounds the budget's claim too.  Under Windows on an RTX 4070 Laptop (8,188 MiB) the
        // budget's usage did not grow while the handles were created, and the pool took 13,204 MiB with 4,760 free:
        // the expert cache filled all of it, and every prompt past a few dozen words then failed with "no VRAM for
        // the prompt products' work memory".
        const size_t cap = std::min(free_b > reserve ? free_b - reserve : (size_t) 0, limit);
        size_t by_budget = 0;
        const bool budgeted = vpool_claim_to_budget(reserve, cap, by_budget);
        if (!budgeted) vpool_claim(cap);
        pool_claimed_ = vpool_bytes() > 0;   // a card with nothing to spare keeps the maps that create their handles
        std::fprintf(stderr, "strata-poly: VRAM pool: %zu MiB claimed for the run (the card had %zu MiB free; margin left "
                             "outside it %d MiB, %s; the most the model can use: %zu MiB)\n",
                     vpool_bytes() >> 20, free_b >> 20, reserve_mib_,
                     budgeted ? "under the OS's budget for this process" : "of the card's free VRAM", limit >> 20);
        free_b = vpool_bytes();
        reserve = stage_map_.base ? 0 : stage_slack;
        if (free_b <= reserve) return short_of(0, 0);
    }
    // sized again (resize_cache()), it is no larger than load() made it, or than its addresses with VMM
    const size_t most = vmm_ ? std::min(all, total_b) :
#if defined(STRATA_GPU_METAL)
                       opt_.metal_balanced ? all :
#endif
                       full_bytes_ > 0 ? full_bytes_ : all;
    size_t arena = std::min({all, free_b - reserve, arena_.base ? arena_.reserved - 256 : most});
#if defined(STRATA_GPU_METAL)
    size_t metal_cap = 0, metal_stage_need = 0;
    if (opt_.metal_balanced) {
        for (const Layer& L : layers_) {
            if (!L.moe) continue;
            drop_file_pages(L.exp_gate, (size_t) n_expert_ * L.gu_stride);
            if (L.gu_stride == L.gu_bytes) drop_file_pages(L.exp_up, (size_t) n_expert_ * L.u_stride);
            drop_file_pages(L.exp_down, (size_t) n_expert_ * L.d_bytes);
        }
        metal_cap = metal_cache_limit_ != SIZE_MAX ? metal_cache_limit_ : metal_host_headroom();
        arena = std::min(arena, metal_cap > 256 ? metal_cap - 256 : 0);
        if (arena < all) {
            if (mmq_ && !stage_) metal_stage_need += stage_bytes_ + 256 + kTail + 2 * chunk_bytes_;
            if (!swap_ && opt_.expert_swaps != 0)   // -1 is decided after the cache is filled: as many as it can be
                metal_stage_need += (size_t) (opt_.expert_swaps < 0 ? 64 : opt_.expert_swaps) * slot_max_;
            if (metal_cache_limit_ == SIZE_MAX && !recorded_)
                metal_stage_need += size_t(scratch_at_) * sizeof(float) + state_n_ * sizeof(float) + (size_t(512) << 20);
            else if (arena > (size_t(512) << 20))
                metal_stage_need += kMetalRefillCushion;
            arena = arena > metal_stage_need ? arena - metal_stage_need : 0;
        }
    }
#endif
    if (expert_cap_ > 0 && arena < all) short_of(arena, all);
    if (arena == 0) return;
    if (vmm_) {   // in pieces, so that it can give some up to the attention caches (shrink_cache()) and grow
#ifdef STRATA_VMM_EQUAL_PIECES
        // with equal pieces, a whole number of kArenaPiece, so that its pieces can all be of that size (vmm_map())
        if (!arena_.base)
            vmm_reserve(arena_, pool_on_ || !vmm_equal_pieces() ? most + 256 : up_to(most + 256, kArenaPiece));
#else
        if (!arena_.base) vmm_reserve(arena_, most + 256);   // addresses only: grow_cache() maps more of them
#endif
        try {
            vmm_map(arena_, arena + 256, kArenaPiece, "the expert cache");
        } catch (const std::runtime_error&) {   // the card may not have it all (grow_cache() is just as tolerant)
            arena = arena_.mapped > 256 ? arena_.mapped - 256 : 0;
            if (expert_cap_ > 0 && arena < all) short_of(arena, all);
            if (arena == 0) return;
        }
        exp_ = (uint8_t*) arena_.base;
        budget_ = free_b - reserve + other_rows();   // when it holds every expert, the rest goes to the rows first
        for (const Kv& k : kv_) budget_ += k.mapped;
    } else {
        exp_ = (uint8_t*) dmalloc(arena + 256, "the expert cache");
    }
#if defined(STRATA_GPU_METAL)
    metal_cache_bytes_ = arena + 256;
    if (!opt_.metal_balanced)
#endif
    zero_vram(exp_, arena + 256);   // the slots' tails stay zero (kTail)
    // Which experts is by value; where they go is by layer, a layer's side by side in the experts' order.
    std::vector<Cand> taken;
    size_t room = arena;
    for (const Cand& c : cands) {
        if (c.bytes > room) continue;   // a smaller expert of another layer may still fit
        room -= c.bytes;
        taken.push_back(c);
    }
    std::sort(taken.begin(), taken.end(),
              [](const Cand& a, const Cand& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
#if !defined(STRATA_GPU_METAL)
    // A layer's experts on the card leave the working set once they are uploaded, as the dense weights do
    // (upload_as()): the CPU does not read them, and all of them held were most of the load's peak.
    int at_il = -1;
    const auto uploaded = [&](int il) {
        if (il < 0) return;
        const Layer& L = layers_[(size_t) il];
        expert_runs(L, n_expert_, [&](int e) { return L.slot[(size_t) e] != nullptr; }, drop_pages);
    };
#endif
    for (const Cand& c : taken) {
        Layer& L = layers_[(size_t) c.il];
#if !defined(STRATA_GPU_METAL)
        if (c.il != at_il) {
            uploaded(at_il);
            at_il = c.il;
        }
#endif
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += c.bytes;
        const size_t e = (size_t) c.e;
        const uint8_t* g_src = L.exp_gate + e * L.gu_stride;
        const uint8_t* u_src = L.exp_up + e * L.u_stride;
        const uint8_t* d_src = L.exp_down + e * L.d_bytes;
        h2d(s, g_src, L.gu_bytes);
        h2d(s + L.gu_bytes, u_src, L.u_bytes);
        h2d(s + L.gu2(), d_src, L.d_bytes);
#if defined(STRATA_GPU_METAL)
        if (opt_.metal_balanced) {
            const size_t used = L.gu2() + L.d_bytes;
            if (c.bytes > used) {
                if (void* hv = gpu::host_view(s + used)) std::memset(hv, 0, c.bytes - used);
                else zero_vram(s + used, c.bytes - used);
            }
            drop_file_pages(g_src, L.gu_bytes);
            drop_file_pages(u_src, L.u_bytes);
            drop_file_pages(d_src, L.d_bytes);
        }
#endif
        L.slot[e] = s;
        ++cached_;
    }
#if !defined(STRATA_GPU_METAL)
    uploaded(at_il);
#endif
    // STRATA_GPU_EXPERTS: which experts the cache holds, an FNV-style hash of its (layer, expert) pairs (FNV-1a's
    // constants, one 32-bit value a step).  Only there is the set fixed for the run; experts of several sizes are
    // taken past one that does not fit, so two runs can cache as many experts but not the same ones.  (N = 0: above.)
    if (expert_cap_ > 0) {
        uint64_t h = 14695981039346656037ULL;
        int n = 0;
        for (size_t il = 0; il < layers_.size(); ++il)
            for (size_t e = 0; e < layers_[il].slot.size(); ++e)
                if (layers_[il].slot[e]) {
                    for (const size_t v : {il, e}) h = (h ^ (uint32_t) v) * 1099511628211ULL;
                    ++n;
                }
        std::fprintf(stderr, "strata-poly: cache set %d experts fnv %016llx\n", n, (unsigned long long) h);
    }
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced) {
        if (arena + 256 > exp_at_) {
            if (void* hv = gpu::host_view(exp_ + exp_at_)) std::memset(hv, 0, arena + 256 - exp_at_);
            else zero_vram(exp_ + exp_at_, arena + 256 - exp_at_);
        }
        const size_t used_host = exp_at_ + 256 + metal_stage_need;
        metal_warm_bytes_ = arena == all && metal_cap > used_host ? metal_cap - used_host : 0;
        for (const Layer& L : layers_) {
            if (!L.moe) continue;
            drop_file_pages(L.exp_gate, (size_t) n_expert_ * L.gu_stride);
            if (L.gu_stride == L.gu_bytes) drop_file_pages(L.exp_up, (size_t) n_expert_ * L.u_stride);
            drop_file_pages(L.exp_down, (size_t) n_expert_ * L.d_bytes);
        }
    }
#endif
    if (full_bytes_ == 0) {
        full_bytes_ = arena;
        full_cached_ = cached_;
    }
}

void Engine::ensure_stage() {
    if (!mmq_ || stage_) return;
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced && all_cached()) return;
#endif
    if (stage_wide_ > stage_narrow_) {
        vmm_reserve(stage_map_, stage_wide_ + 256 + kTail);
        // with the pool the buffer comes out of it, and out of the expert cache when the pool lacks the room
        if (pool_on_ && pool_claimed_ && exp_) {
            const size_t need = up_to(stage_narrow_ + 256 + kTail, pg_);
            if (need > vpool_bytes()) shrink_cache(need - vpool_bytes());
        }
        vmm_map(stage_map_, stage_narrow_ + 256 + kTail, SIZE_MAX, "the expert staging buffer");
        stage_ = (uint8_t*) stage_map_.base;
    } else {
        stage_ = (uint8_t*) dmalloc(stage_bytes_ + 256 + kTail, "the expert staging buffer");
    }
}

// The pinned buffers that stage() fills in turn.  A card that holds every expert stages none: they wait for the
// first one that it lacks, 128 MiB of RAM the load no longer takes.
void Engine::stage_pins() {
    for (int b = 0; b < 2; ++b)
        if (!h_stage_[b]) h_stage_[b] = (uint8_t*) hmalloc(chunk_bytes_, "pinned memory");
}

// The thread and the stream that bring experts into the cache while an answer is written (follow()).
void Engine::start_swaps() {
    if (swap_ || cached_ == 0 || opt_.expert_swaps == 0) return;
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced && all_cached()) return;
#endif
    swap_ = std::make_unique<Swap[]>((size_t) opt_.expert_swaps);
    cudaStream_t up = nullptr;
    ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
    swap_st_ = up;
    for (; n_swap_ < opt_.expert_swaps; ++n_swap_) {
        ck(cudaEventCreateWithFlags(&swap_[(size_t) n_swap_].done, cudaEventDisableTiming), "cudaEventCreate");
        ck(cudaEventCreateWithFlags(&swap_[(size_t) n_swap_].freed, cudaEventDisableTiming), "cudaEventCreate");
    }
    for (int b = 0; b < kSwapPins; ++b) {
        swap_pin_[b] = (uint8_t*) hmalloc(slot_max_, "pinned memory");
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        swap_pin_ev_[b] = ev;
    }
    recent_.assign(use_.size(), 0.f);
    coming_.assign(use_.size(), 0);
    follower_ = std::thread([this] { follow_loop(); });
}

// The cache gives up what it has and is filled again for another reserve.  The experts on their way into it
// arrive first: their slots are about to go.
void Engine::resize_cache(int mib) {
    for (Engine* e = this; e; e = e->next_.get()) e->resize_card_cache(mib);
}

void Engine::resize_card_cache(int mib) {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    // As shrink_cache(): a frozen answer's cache does not change under it (the vram API resizes between
    // requests; unfreeze_experts() ends the freeze before one begins).  STRATA_GPU_EXPERTS holds it for the run.
    if (freeze_experts_ || expert_cap_ >= 0) return;
    OnCard here(device_, split_);
#if defined(STRATA_GPU_METAL)
    // Every Metal graph retains its indirect resources, including in the legacy memory mode.
    metal_sync_memory();
    metal_invalidate_graphs();
#endif
    ck(cudaStreamSynchronize(cs(st_)), "resize");
    const auto failed = [&] {
        std::lock_guard<std::mutex> l(swap_m_);
        return !swap_error_.empty();
    };
    for (int j = 0; j < n_swap_; ++j)   // one that is asked for is the follower's, now or in a moment
        while (swap_[(size_t) j].at.load() == Swap::kAsked && !failed()) std::this_thread::yield();
    if (failed()) throw std::runtime_error("expert upload: " + swap_error_);
    if (swap_st_) ck(cudaStreamSynchronize(cs(swap_st_)), "resize");
    for (int j = 0; j < n_swap_; ++j) swap_[(size_t) j].at.store(Swap::kFree);
    std::fill(coming_.begin(), coming_.end(), (uint8_t) 0);
    for (Layer& L : layers_) std::fill(L.slot.begin(), L.slot.end(), nullptr);
    refill_.clear();
    staged_il_ = -1;
    cached_ = 0;
    exp_at_ = 0;
    if (exp_ && vmm_) {
        vmm_unmap(arena_, 0);
        exp_ = nullptr;
    } else if (exp_) {
        dev_allocs_.erase(std::find(dev_allocs_.begin(), dev_allocs_.end(), (void*) exp_));
        ck(cudaFree(exp_), "resize");
        exp_ = nullptr;
    }
#if defined(STRATA_GPU_METAL)
    metal_cache_bytes_ = 0;
#endif
    opt_.vram_reserve_mib = mib < 0 ? asked_reserve_mib_ : mib;
    fill_expert_cache();
    ensure_stage();
    start_swaps();
}

// The launches the engine records as it runs take VRAM after the cache was sized: 180 to 200 MiB of an 8 GB card,
// most of it in the first request.  That is what the reserve is for.  From the second request on the cache keeps
// `keep` free: the reserve the caller named, else half the one it was filled for and 128 MiB at least, or the
// whole of it when the PC's RAM is tight for the model (default_reserve_mib(): with the reserve named, the laptop
// read as the release did; with half of it kept, 7% to 14% slower).  It gives back what another program took, its
// least used experts, and takes VRAM that came free again for the most used experts it lacks.  Filled again
// whole it went down with every program that held VRAM a moment, and never came back up.
bool Engine::mind_vram() {
    bool changed = false;
    for (Engine* e = this; e; e = e->next_.get()) changed = e->mind_card_vram() || changed;
    return changed;
}

bool Engine::mind_card_vram() {
#if defined(STRATA_GPU_METAL)
    if (opt_.metal_balanced) return metal_mind_memory();
#endif
    if (!exp_ || reserve_mib_ <= 0) return false;
    // As shrink_cache(): a frozen answer's cache is the one its first token saw.
    if (freeze_experts_) return false;
    OnCard here(device_, split_);
    if (pool_on_) {
        // The pool is the cache's and the rows': what the rows gave back (trim_kv()) goes to the experts the cache
        // lacks that are used most.  Nothing outside the pool is looked at: the margin stays where it was left.
        if (minds_++ == 0) return false;   // nothing is recorded before the first request
        // Two steps of the rows' growth stay in it (kv_step_bytes()), so that an answer that goes on does not take
        // experts out again
        const size_t have = vpool_bytes(), slack = 2 * kv_step_bytes();
        return have >= slack + (16u << 20) && grow_cache(have - slack) > 0;
    }
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const int64_t free = (int64_t) (free_b >> 20);   // this card's
    if (!vmm_) {   // a cache in one allocation is filled again
        if (free >= reserve_mib_ / 4) return false;
        resize_card_cache(opt_.vram_reserve_mib);
        return true;
    }
    if (minds_++ == 0) return false;   // nothing is recorded before the first request
    const int64_t keep = opt_.vram_reserve_mib >= 0 || ram_tight_ ? reserve_mib_ : std::max(reserve_mib_ / 2, 128);
    if (free < keep / 2) {
        shrink_cache(size_t(keep - free) << 20);
        held_is_budget();
        return true;
    }
    return free >= keep + (int64_t) (kArenaPiece >> 20) && grow_cache(size_t(free - keep) << 20) > 0;
}

// A read of several calls copies each layer's experts that are not cached through the staging buffer, a round
// at a time, and again for each call: at 1 GiB, a layer of GLM 5.3 Flash Q2_K (about 2.4 GiB) went through in
// three rounds a call.  For the read the buffer grows to stage_wide_, which holds the layer, with VRAM the
// expert cache gives up (its least used experts) when the card has not that much free; narrow_stage() gives it
// back.  VRAM the card does not give leaves the buffer as it was.
// A call of kStageRows rows or more needs the buffers lend_prompt() gave the expert cache: it takes the VRAM back,
// the cache's least used experts first, and maps them again.  The products' work memory (work_map_) is lent from
// the start, so the first prompt takes it here.
void Engine::take_prompt() {
    if (!prompt_lent_ && !work_lent_) {
        if (two_scratch_ && !prompt_scratch_) use_scratch(true);
        return;
    }
    if (up_) ck(cudaStreamSynchronize(cs(up_)), "a prompt's buffers");
    ck(cudaStreamSynchronize(cs(st_)), "a prompt's buffers");
    const size_t grain = pool_on_ ? pg_ : grain_;
    const size_t mine = prompt_lent_ ? up_to(prompt_bytes_, grain) : 0;
    const size_t want = mine + (work_lent_ ? up_to(work_bytes_, grain) : 0);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    if (pool_on_) free_b = vpool_bytes();
    const size_t quarter = pool_on_ ? 0 : (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
    work_over_ = 0;
    if (exp_ && free_b < want + quarter) {
        // shrink_cache() cuts whole pieces (kArenaPiece) from the cache's end, more than the need.  The part of that
        // excess the prompt's buffers alone would not have cut is the work memory's, and lend_prompt() gives it back
        // with the memory: Qwen3.6 UD-Q3_K_M at a prompt batch of 1,024 cut 242 MiB for a need of 184, where the
        // buffers alone cut 114 for 88, and the 32 MiB more stayed free on the card.
        const size_t need = want + quarter - free_b, alone = mine + quarter > free_b ? mine + quarter - free_b : 0;
        size_t alone_cut = 0;
        for (size_t i = arena_.pieces.size(); i > 0 && alone_cut < alone; --i) alone_cut += arena_.pieces[i - 1].second;
        const size_t had = arena_.mapped;
        shrink_cache(need);
        const size_t over = had - arena_.mapped > need ? had - arena_.mapped - need : 0;
        const size_t over_alone = alone_cut > alone ? alone_cut - alone : 0;
        if (over > over_alone) work_over_ = over - over_alone;
    }
    if (!pool_on_) budget_ -= std::min(budget_, want);
    if (prompt_lent_) vmm_map(prompt_map_, prompt_bytes_, SIZE_MAX, "a prompt's buffers");
    if (work_lent_) vmm_map(work_map_, work_bytes_, SIZE_MAX, "the prompt products' work memory");
    prompt_lent_ = work_lent_ = false;
    if (two_scratch_) use_scratch(true);
}

// Between prompts, the expert cache has the VRAM of the buffers only a prompt's calls use (alloc()), the scratch's
// layout for a prompt's batch among them: a call of a few tokens has its own.
void Engine::lend_prompt() {
    if (two_scratch_ && prompt_scratch_) use_scratch(false);
    if (prompt_lent_ || !prompt_lends_ || freeze_experts_ || !exp_ || all_cached()) return;
    ck(cudaStreamSynchronize(cs(st_)), "a prompt's buffers");
    vmm_unmap(prompt_map_, 0);
    prompt_lent_ = true;
    size_t bytes = prompt_bytes_, budget = up_to(prompt_bytes_, grain_);
    if (work_map_.base && !work_lent_) {   // (whole pieces: alloc() rounds it to them)
        vmm_unmap(work_map_, 0);
        work_lent_ = true;
        bytes += work_bytes_ + work_over_;
        budget += work_bytes_;
    }
    if (!pool_on_) budget_ += budget;
    grow_cache(bytes, true);
}

void Engine::widen_stage() {
    if (stage_lent_ || stage_wide_ <= stage_narrow_) return;
    if (up_) ck(cudaStreamSynchronize(cs(up_)), "the expert staging buffer");
    ck(cudaStreamSynchronize(cs(st_)), "the expert staging buffer");
    const size_t want = up_to(stage_wide_ + 256 + kTail, pool_on_ ? pg_ : grain_) - stage_map_.mapped;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    if (pool_on_) free_b = vpool_bytes();   // what the card has is the pool's
    const size_t quarter = pool_on_ ? 0 : (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
    stage_cut_ = 0;
    if (exp_ && free_b < want + quarter) {
        const size_t had = arena_.mapped;
        shrink_cache(want + quarter - free_b);
        held_is_budget();
        stage_cut_ = had - arena_.mapped;
    }
    stage_lent_ = true;   // narrow_stage() gives back what was cut even if the map below fails
    try {
        vmm_map(stage_map_, stage_wide_ + 256 + kTail, SIZE_MAX, "the expert staging buffer");
    } catch (const std::runtime_error&) {
        narrow_stage();
        return;
    }
    stage_bytes_ = stage_wide_;
    staged_il_ = -1;
}

// The read is done: the staging buffer goes back to stage_narrow_, and the expert cache takes back what it gave
// up for it, the most used experts it lacks.
void Engine::narrow_stage() {
    if (!stage_lent_) return;
    if (up_) ck(cudaStreamSynchronize(cs(up_)), "the expert staging buffer");
    ck(cudaStreamSynchronize(cs(st_)), "the expert staging buffer");
    vmm_unmap(stage_map_, stage_narrow_ + 256 + kTail);
    stage_bytes_ = stage_narrow_;
    staged_il_ = -1;
    stage_lent_ = false;
    if (exp_ && stage_cut_ > 0) grow_cache(stage_cut_, true);
    stage_cut_ = 0;
}

// The cache's VRAM and the attention caches' as they are now become the budget cover() keeps them to.
void Engine::held_is_budget() {
    budget_ = arena_.mapped + other_rows();
    for (const Kv& k : kv_) budget_ += k.mapped;
}

float* Engine::seq_state(int s, const float* p) const {
    const float* base = s == bound_ ? state_ : seqs_[(size_t) s].state;
    return const_cast<float*>(base) + (p - state_);
}

// The buffers of every sequence were asked for in the same order with the same sizes (kv_alloc(), seq_alloc()).
void* Engine::seq_buf(int s, const void* p) const {
    const std::vector<Buf>& to = s == bound_ ? bufs_ : seqs_[(size_t) s].bufs;
    const auto* q = (const uint8_t*) p;
    for (size_t i = 0; i < bufs_.size(); ++i) {
        const Buf& b = bufs_[i];
#if defined(STRATA_GPU_METAL)
        const size_t all = b.capacity;
#else
        const size_t all = b.per ? (size_t) ((opt_.max_context + b.per - 1) / b.per) * b.bytes : b.bytes;
#endif
        const auto* at = (const uint8_t*) b.p;
        if (q >= at && q < at + all) {
#if defined(STRATA_GPU_METAL)
            if ((size_t) (q - at) >= to[i].capacity) throw std::logic_error("seq_buf: destination capacity");
#endif
            return (uint8_t*) to[i].p + (q - at);
        }
    }
    throw std::logic_error("seq_buf: not in a buffer of the sequence");
}

// The VRAM the attention caches of the sequences that are not bound hold.
size_t Engine::other_rows() const {
    size_t b = 0;
    for (const Seq& q : seqs_)
        for (const Kv& k : q.kv) b += k.mapped;
    return b;
}

// `bytes` more VRAM at the cache's end, for the experts it lacks that are used most (use per byte, as
// fill_expert_cache() chooses), side by side by layer.  VRAM the card does not give is not an error: the cache
// keeps what was mapped.  Returns the experts it took.  `later`: the follower uploads them while the answer is
// written (follow()), in place of a wait here: 57 ms for the 148 experts Gemma 4 lent a read's staging buffer.
// Options::lock_experts: an expert that comes back this way keeps its slot in the mirror (refills_mirror()).
int Engine::grow_cache(size_t bytes, bool later) {
    if (!vmm_ || !exp_) return 0;
    // As shrink_cache(): a frozen answer's cache does not change under it.
    if (freeze_experts_) return 0;
    later = later && n_swap_ > 0;
    settle_swaps();
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    if (swap_st_) ck(cudaStreamSynchronize(cs(swap_st_)), "the expert cache");
    struct Cand {
        int il, e;
        double value;
    };
    std::vector<Cand> cands;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            if (L.slot[(size_t) e] || (!coming_.empty() && coming_[i])) continue;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, used / (double) L.slot_bytes});
        }
    }
    if (cands.empty()) return 0;
    try {
        vmm_map(arena_, std::min(arena_.mapped + bytes, arena_.reserved), kArenaPiece, "the expert cache");
    } catch (const std::runtime_error&) {
    }
    if (arena_.mapped < exp_at_ + kTail) return 0;
    zero_vram(exp_ + exp_at_, arena_.mapped - exp_at_);   // the new slots' tails
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });
    std::vector<Cand> taken;
    size_t room = arena_.mapped - kTail - exp_at_;
    for (const Cand& c : cands) {
        const size_t n = layers_[(size_t) c.il].slot_bytes;
        if (n > room) continue;
        room -= n;
        taken.push_back(c);
    }
    std::sort(taken.begin(), taken.end(),
              [](const Cand& a, const Cand& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
    for (const Cand& c : taken) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += L.slot_bytes;
        if (later) {
            refill_.push_back({c.il, c.e, s});
            coming_[(size_t) c.il * n_expert_ + c.e] = 1;
            continue;
        }
        const size_t e = (size_t) c.e;
        if (const uint8_t* m = mirror(c.il, c.e)) {
            h2d(s, m, L.gu2() + L.d_bytes);
            if (!refills_mirror()) mirror_give(c.il, c.e);
        } else {
            h2d(s, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            h2d(s + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes);
            h2d(s + L.gu2(), L.exp_down + e * L.d_bytes, L.d_bytes);
        }
        L.slot[e] = s;
        ++cached_;
    }
    staged_il_ = -1;
    held_is_budget();
    return (int) taken.size();
}

// The slots grow_cache() left to the follower that it has not asked for yet, filled here: the cache is about to
// be cut or sized again, and an empty slot of it is lost to both.
void Engine::fill_now() {
    for (const Refill& r : refill_) {
        Layer& L = layers_[(size_t) r.il];
        const size_t e = (size_t) r.e;
        if (const uint8_t* m = mirror(r.il, r.e)) {
            h2d(r.slot, m, L.gu2() + L.d_bytes);
            if (!refills_mirror()) mirror_give(r.il, r.e);
        } else {
            h2d(r.slot, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            h2d(r.slot + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes);
            h2d(r.slot + L.gu2(), L.exp_down + e * L.d_bytes, L.d_bytes);
        }
        L.slot[e] = r.slot;
        coming_[(size_t) r.il * n_expert_ + e] = 0;
        ++cached_;
    }
    refill_.clear();
}

// What a layer copies to the staging buffer ahead of its routing: the experts outside the cache, the most used
// first, as many as the buffer holds.  Worked out when the layer runs: the cache changes (follow()).
void Engine::plan_ahead(Layer& L, int il) {
    L.ahead.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (!L.slot[(size_t) e]) L.ahead.push_back(e);
    if (!prior_.empty()) {
        const int64_t* used = prior_.data() + (size_t) il * n_expert_;
        std::stable_sort(L.ahead.begin(), L.ahead.end(), [used](int32_t a, int32_t b) { return used[a] > used[b]; });
    }
    L.ahead.resize(std::min(L.ahead.size(), stage_bytes_ / (L.gu_step + L.d_bytes)));
}

// The follower: an asked-for expert goes from the mapped file to a pinned buffer, and from there to its slot.
// The file may have to be read for it, which is why this is not the engine's thread.  Two buffers take turns: an
// expert is copied into one while the other uploads, and the upload is the quicker of the two.  (A buffer for
// each of the 64 uploads that may be asked at once held 515 MiB of LFM2.5's RAM.)
void Engine::follow_loop() {
    cudaSetDevice(device_);   // a thread starts on card 0
    std::unique_lock<std::mutex> lk(swap_m_);
    for (int b = 0;; b = (b + 1) % kSwapPins) {
        Swap* s = nullptr;
        swap_cv_.wait(lk, [&] {
            for (int j = 0; j < n_swap_ && !s; ++j)
                if (swap_[(size_t) j].at.load() == Swap::kAsked) s = &swap_[(size_t) j];
            return swap_quit_ || s;
        });
        if (swap_quit_) return;
        // Under the RAM budget an expert comes from its pool_ slot, or else from the files: through the mapping its
        // pages stayed in the working set.  A copy from RAM is many times faster than a read: with every one read,
        // Qwen3.6's cache kept up with an answer after a 32K prompt so much more slowly that the CPU computed 14%
        // more experts.
        const int32_t from = s->from;
        const bool mapped = s->mapped;
        const int64_t takes = s->takes;
        const uint8_t* const mirrored = s->mirror;
        uint8_t* const spare = s->spare;
        const bool wait_freed = std::exchange(s->wait_freed, false);
        lk.unlock();
        const Layer& L = layers_[(size_t) s->il];
        const size_t e = (size_t) s->e, gu = L.gu_bytes, gu2 = L.gu2();
        uint8_t* const h = swap_pin_[b];
        cudaError_t err = cudaSuccess;
        // the expert whose slot this is, back into the mirror before the upload takes its place there
        if (wait_freed) err = cudaStreamWaitEvent(cs(swap_st_), s->freed, 0);
        if (err == cudaSuccess && spare)
            err = cudaMemcpyAsync(spare, s->slot, L.slot_bytes, cudaMemcpyDeviceToHost, cs(swap_st_));
        if (err == cudaSuccess && mirrored) {
            err = cudaMemcpyAsync(s->slot, mirrored, gu2 + L.d_bytes, cudaMemcpyHostToDevice, cs(swap_st_));
        } else if (err == cudaSuccess && (err = cudaEventSynchronize((cudaEvent_t) swap_pin_ev_[b])) == cudaSuccess) {
            // h's last upload has left it
            const auto get = [&](uint8_t* dst, const uint8_t* src, size_t n) {
                if (mapped || !read_file(dst, src, n)) std::memcpy(dst, src, n);
            };
            if (from >= 0) {
                std::memcpy(h, pool_at((size_t) from), gu2 + L.d_bytes);
            } else {
                get(h, L.exp_gate + e * L.gu_stride, gu);
                get(h + gu, L.exp_up + e * L.u_stride, L.u_bytes);
                get(h + gu2, L.exp_down + e * L.d_bytes, L.d_bytes);
            }
            err = cudaMemcpyAsync(s->slot, h, gu2 + L.d_bytes, cudaMemcpyHostToDevice, cs(swap_st_));
        }
        if (err == cudaSuccess) {
            if (takes >= 0) {
                // Its pages are copied: they go, and those of the expert that takes its place come, by the read
                // threads.  Here they held the uploads up while the cache was busiest, in the first answer.
                const Layer& T = layers_[(size_t) takes / (size_t) n_expert_];
                const int te = (int) (takes % n_expert_);
                std::vector<PoolJob> jobs;
                const auto as = [&](int32_t k) {
                    return [&jobs, k](const uint8_t* p, size_t n) { jobs.push_back({nullptr, p, n, k}); };
                };
                expert_bytes(L, (int) e, (int) e + 1, as(kDrop));
                expert_bytes(T, te, te + 1, as(kTouch));
                {
                    std::lock_guard<std::mutex> l(pool_m_);
                    pool_pages_.insert(pool_pages_.end(), jobs.begin(), jobs.end());
                    pool_n_.fetch_add((int) jobs.size(), std::memory_order_release);
                }
                pool_cv_.notify_all();
            }
        }
        if (err == cudaSuccess) err = cudaEventRecord(s->done, cs(swap_st_));
        if (err == cudaSuccess) err = cudaEventRecord((cudaEvent_t) swap_pin_ev_[b], cs(swap_st_));
        lk.lock();
        if (err != cudaSuccess) {
            swap_error_ = cudaGetErrorString(err);
            return;
        }
        s->at.store(Swap::kSent);
    }
}

// The cache follows what is written.  Called at the start of an eval, where nothing queued reads the cache, with
// the number of tokens the eval writes (0 for a prompt batch).  The experts whose upload has arrived get their
// slots.  Then, for written tokens, the experts used lately that the cache lacks start on their way, each to the
// slot of the cached expert of its size that was used least (among equals, the one the profile ranks last).
//
// A prompt batch neither counts nor moves anything.  What a prompt routes to says no more about the answer than
// the profile does, and moving the cache while the prompt was read cost 3% of the reading speed (measured).
void Engine::follow(int written) {
    {
        std::lock_guard<std::mutex> l(swap_m_);
        if (!swap_error_.empty()) throw std::runtime_error("expert upload: " + swap_error_);
    }
    // STRATA_FREEZE_EXPERTS: the split between the card's experts and the CPU's is the request's first token's
    // for the whole answer, so an arrival claims no slot and no trade is asked while one is being written.  A
    // frozen answer sees a cache that never changes, which makes it a function of the request alone.  The
    // uploads themselves go on: the followers fill their slots, and unfreeze_experts() lets the next request's
    // first token claim them.
    if (freeze_experts_) return;
    const size_t ne = (size_t) n_expert_;
#if !defined(STRATA_GPU_METAL)
    // card_far_ or the doorbell: the calls routed on the card counted their picks on the card (moe_card()).  The
    // doorbell counts only in the calls that rang (bell_counts_): with every expert in VRAM it never does, and this
    // copy, which waits for the card, would hold every token's launches until the one before it is done.
    if ((card_far_ || bell_counts_) && d_use_) {
        bell_counts_ = false;
        const size_t count = layers_.size() * ne;
        h_use_.resize(count);
        ck(cudaMemcpy(h_use_.data(), d_use_, count * sizeof(int32_t), cudaMemcpyDeviceToHost), "expert uses");
        ck(cudaMemsetAsync(d_use_, 0, count * sizeof(int32_t), cs(st_)), "expert uses");
        for (size_t i = 0; i < count; ++i) {
            if (!h_use_[i]) continue;
            use_[i] += h_use_[i];
            if (n_swap_ > 0 && i < recent_.size()) recent_[i] += (float) h_use_[i];
        }
    }
#endif
    int n_free = 0;
    for (int j = 0; j < n_swap_; ++j) {
        Swap& s = swap_[(size_t) j];
        if (s.at.load() == Swap::kSent) {
            const cudaError_t arrived = cudaEventQuery(s.done);
            if (arrived == cudaErrorNotReady) continue;
            ck(arrived, "expert upload");
            layers_[(size_t) s.il].slot[(size_t) s.e] = s.slot;
            coming_[(size_t) s.il * ne + (size_t) s.e] = 0;
            if (s.mirror && !(s.adds && refills_mirror())) mirror_give(s.il, s.e);
            if (s.spare) mirror_of_[(size_t) s.out] = s.spare;
            s.mirror = nullptr;
            s.spare = nullptr;
            s.out = -1;
            if (s.from >= 0) {   // the card has it now: its pool_ slot is free for another
                const size_t v = (size_t) s.from;
                --pool_pin_[v];
                pool_sizes_[pool_size_of_[(size_t) s.il]].nprot -= pool_prot_[v];
                pool_prot_[v] = 0;
                pool_of_[(size_t) s.il * ne + (size_t) s.e] = -1;
                pool_who_[v] = -1;
                s.from = -1;
                // the expert the cache gave up for it, if it is still the CPU's and of the slot's size
                const int64_t g = s.gets;
                if (g >= 0 && pool_of_[(size_t) g] < 0 && !kept_[(size_t) g] &&
                    !layers_[(size_t) g / ne].slot[(size_t) g % ne] &&
                    pool_size_of_[(size_t) g / ne] == pool_size_of_[(size_t) s.il]) {
                    pool_who_[v] = g;
                    pool_of_[(size_t) g] = (int32_t) v;
                    pool_tick_[v] = pool_now_;
                    pool_queue(layers_[(size_t) g / ne], (size_t) g % ne, (int32_t) v, false);
                }
                s.gets = -1;
            }
            cached_ += s.adds;
            s.adds = false;
            s.at.store(Swap::kFree);
            ++stats_.swapped_experts;
            staged_il_ = -1;   // the staging buffer holds what was not cached
        }
        n_free += s.at.load() == Swap::kFree;
    }
    if (written <= 0) return;
    const float keep = std::exp2(-(float) written / (float) kSwapHalf);
    for (float& r : recent_) r *= keep;
    const auto from_pool = [&](Swap& s) {   // the slot follow_loop() copies s from, held until s arrives
        s.from = -1;
        if (!pool_) return;
        const int32_t v = pool_of_[(size_t) s.il * ne + (size_t) s.e];
        if (v < 0 || pool_state_[(size_t) v].load(std::memory_order_acquire) != 0) return;
        s.from = v;
        ++pool_pin_[(size_t) v];
    };
    // The empty slots grow_cache() left to the uploads go first.
    bool asked = false;
    for (int j = 0; j < n_swap_ && !refill_.empty(); ++j) {
        Swap& s = swap_[(size_t) j];
        if (s.at.load() != Swap::kFree) continue;
        const Refill r = refill_.back();
        refill_.pop_back();
        s.il = r.il;
        s.e = r.e;
        s.slot = r.slot;
        s.adds = true;
        from_pool(s);
        s.mirror = mirror(s.il, s.e);
        s.mapped = !pool_ || kept_[(size_t) s.il * ne + (size_t) s.e];
        s.takes = -1;
        s.at.store(Swap::kAsked);
        asked = true;
        --n_free;
    }
    // card_far_: a trade's outgoing expert is copied on st_ (below), so a window asks kFarTrades at most.  At 12 GB,
    // forced, 16 a window took the mirror's spares faster than arrivals gave them back, and calls fell back to the
    // host; 1 a window kept the cache from following the answer.
    if (card_far_) n_free = std::max(0, std::min(n_free, kFarTrades));
    if (n_free == 0) {
        if (asked) {
            { std::lock_guard<std::mutex> l(swap_m_); }
            swap_cv_.notify_one();
        }
        return;
    }

    // One pass over the experts lists, for each slot size, the ones that may come in, the most used first, and
    // the cached ones that may go, the least used first: as many of each as uploads are free, which is all the
    // pairs below can take.  A pass for every upload was 0.6 ms of a window of GLM 5.3 Flash's.
    const size_t most = (size_t) n_free;
    in_.assign((size_t) n_trade_ * most, Pick{});
    out_.assign((size_t) n_trade_ * most, Pick{});
    // p to its place in a list, whose last one drops out; equals stay in the order they came
    const auto put = [most](Pick* list, const Pick& p, bool (*before)(const Pick&, const Pick&)) {
        if (list[most - 1].il >= 0 && !before(p, list[most - 1])) return;
        size_t at = most - 1;
        for (; at > 0 && (list[at - 1].il < 0 || before(p, list[at - 1])); --at) list[at] = list[at - 1];
        list[at] = p;
    };
    const auto more = [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    };
    const auto less = [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent < b.recent : a.prior < b.prior;
    };
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        Pick* in = in_.data() + (size_t) L.trade * most;
        Pick* out = out_.data() + (size_t) L.trade * most;
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * ne + (size_t) e;
            const float r = recent_[i];
            const int64_t p = prior_.empty() ? 0 : prior_[i];
            if (L.slot[(size_t) e]) put(out, {il, e, r, p}, less);
            else if (r >= kSwapMin && !coming_[i]) put(in, {il, e, r, p}, more);
        }
    }

    // An upload goes to the size whose next pair gains most.  A cached expert that has given its slot is no
    // candidate to come in: it was used no more than the cached ones left.
    trade_at_.assign((size_t) n_trade_, 0);
    for (; n_free > 0; --n_free) {
        int best = -1;
        float gain = 0.f;
        for (int c = 0; c < n_trade_; ++c) {
            const size_t k = (size_t) c * most + trade_at_[(size_t) c];
            const Pick& in = in_[k];
            const Pick& out = out_[k];
            if (in.il < 0 || out.il < 0 || in.recent < 2.f * out.recent + kSwapMin) continue;
            if (best < 0 || in.recent - out.recent > gain) {
                best = c;
                gain = in.recent - out.recent;
            }
        }
        if (best < 0) break;

        const size_t k = (size_t) best * most + trade_at_[(size_t) best]++;
        const Pick in = in_[k], out = out_[k];
        Swap* s = nullptr;
        for (int j = 0; j < n_swap_ && !s; ++j)
            if (swap_[(size_t) j].at.load() == Swap::kFree) s = &swap_[(size_t) j];
        uint8_t*& given = layers_[(size_t) out.il].slot[(size_t) out.e];
        s->il = in.il;
        s->e = in.e;
        s->slot = given;
        given = nullptr;
        staged_il_ = -1;
        coming_[(size_t) in.il * ne + (size_t) in.e] = 1;
        from_pool(*s);
        s->mirror = mirror(in.il, in.e);
        if (!mirror_of_.empty() && !mirror(out.il, out.e)) {
            if ((s->spare = mirror_take(out.il))) s->out = (int64_t) out.il * (int64_t) ne + out.e;
            else lock_queue(out.il, out.e);
        }
        // card_far_: the outgoing expert goes to its spare now, on st_, and is the mirror's at once, so that a call
        // routed on the card while the swap is on its way finds it there; the upload waits for that copy
        if (card_far_ && s->spare) {
            ck(cudaMemcpyAsync(s->spare, s->slot, layers_[(size_t) out.il].slot_bytes, cudaMemcpyDeviceToHost, cs(st_)),
               "expert upload");
            ck(cudaEventRecord(s->freed, cs(st_)), "expert upload");
            mirror_of_[(size_t) s->out] = s->spare;
            s->spare = nullptr;
            s->out = -1;
            s->wait_freed = true;
        }
        // Under the RAM budget the expert the cache gives up is the CPU's from now on.  The one that comes in leaves
        // kept_, and the one that goes out takes its place there, of the same size: the follower drops the first
        // one's pages once it has copied them and reads the other's in.  Left out of kept_ and the slots, an expert
        // the cache gave up was read from the files by the call that met it: the first answer after the load missed
        // 194 experts with every other one in RAM.
        const size_t xi = (size_t) in.il * ne + (size_t) in.e, xo = (size_t) out.il * ne + (size_t) out.e;
        s->mapped = !pool_ || kept_[xi];
        s->takes = -1;
        s->gets = -1;
        // One that comes from a slot leaves it to the outgoing one, read into it once the upload has arrived.  With
        // only kept_ handed over, the experts the cache gave up for slotted ones were 28% of a short chat's misses
        // after a 32K prompt on lab-pro6000.
        if (pool_ && s->from >= 0 && !kept_[xo]) s->gets = (int64_t) xo;
        if (pool_ && kept_[xi]) {
            kept_[xi] = 0;
            kept_[xo] = 1;
            s->takes = (int64_t) xo;
        }
        s->at.store(Swap::kAsked);
        asked = true;
    }
    if (asked) {
        { std::lock_guard<std::mutex> l(swap_m_); }
        swap_cv_.notify_one();
    }
}

void Engine::load(const Options& o) {
#if defined(STRATA_GPU_METAL)
    gpu::metal::set_optimize(o.metal_optimize);
    gpu::metal::set_scoped_residency(o.metal_balanced);
#endif
    opt_ = o;
    if ((!opt_.kv_grow || opt_.batch > 0) && opt_.kv_resident > 0) opt_.kv_resident = 0;
    asked_reserve_mib_ = o.vram_reserve_mib;
    if (opt_.max_context < 16) throw std::runtime_error("the context must be at least 16 tokens");
    if (opt_.prompt_batch < 1 || opt_.prompt_batch > 4096)
        throw std::runtime_error("a prompt batch is 1 to 4096 tokens");
    if (opt_.stage_mib < 0) throw std::runtime_error("the staging buffer must be at least 1 MiB (0: the default)");
    if (opt_.stage_ahead < 0) throw std::runtime_error("the batch size to stage ahead from cannot be negative");
    if (opt_.expert_swaps < -1 || opt_.expert_swaps > 64)
        throw std::runtime_error("0 to 64 experts can be on their way into the cache at a time (-1: the default)");
    if (opt_.window < 0 || opt_.window > 8)
        throw std::runtime_error("a window is 1 to 8 tokens, or 0 for the model's own");
    if (opt_.lock_experts && opt_.ram_experts_mib >= 0)
        throw std::runtime_error("--lock-experts locks every expert outside VRAM in RAM, so it does not go with "
                                 "--ram-experts-mib");
    if (opt_.tail_skip < 0) throw std::runtime_error("--tail-skip (STRATA_ROUTE_TAIL_SKIP) cannot be negative");
    if (opt_.route_top < 0) throw std::runtime_error("--route-top cannot be negative");
    if (!(opt_.skip_weight >= 0.f && opt_.skip_weight < 1.f))
        throw std::runtime_error("--skip-weight is a share of a token's experts' weight: 0 (off) up to below 1");
    if (!(opt_.cache_bias >= 0.f && opt_.cache_bias <= 16.f))
        throw std::runtime_error("--cache-bias is added to a router score: 0 (off) up to 16");
    if (opt_.dump || std::min(opt_.prompt_batch, opt_.max_context) == 1) opt_.window = 1;
    const char* stream_stage = std::getenv("STRATA_POLY_STREAM_STAGE");
    const bool requested_stream_stage = stream_stage && std::atoi(stream_stage) == 1;
    stream_stage_ = requested_stream_stage && stage_copy_streaming_available();
    if (requested_stream_stage)
        std::fprintf(stderr, "strata-poly: STRATA_POLY_STREAM_STAGE: %s\n",
                     stream_stage_ ? "streaming CPU stores for expert staging" : "unsupported CPU build; stock copies");
    gg_ = std::make_unique<Shards>(opt_.model);
    if (!opt_.draft_model.empty() && opt_.draft && opt_.window != 1) dg_ = std::make_unique<Shards>(opt_.draft_model);
#if defined(STRATA_GPU_METAL)
    if (opt_.kv_format == KvFmt::kAuto) {
        const MetaValue* arch = gg_->get("general.architecture");
        const MetaValue* heads = gg_->get("qwen35moe.attention.head_count_kv");
        const MetaValue* key = gg_->get("qwen35moe.attention.key_length");
        const MetaValue* value = gg_->get("qwen35moe.attention.value_length");
        // The Q8 attention kernels store a scale for each 32 values within each head.
        opt_.kv_format = arch && arch->s == "qwen35moe" && heads && heads->u > 0 &&
                         key && key->u > 0 && key->u % 32 == 0 && value && value->u == key->u
                             ? KvFmt::kQ8_0 : KvFmt::kFp16;
    }
#endif
    load_geometry();
#if defined(STRATA_GPU_METAL)
    std::fprintf(stderr, "strata-poly: Metal optimize=%s memory=%s kv=%s\n",
                 opt_.metal_optimize ? "on" : "off", opt_.metal_balanced ? "balanced" : "legacy",
                 opt_.kv_format == KvFmt::kQ8_0 ? "q8_0" : "fp16");
#endif
    if (block_ != dg_.get()) dg_.reset();
    hi_ = end_ < 0 ? n_layer_ : end_;

    // STRATA_POLY_STAGED_TILES=8,16,32: a prompt batch's staged experts go to llama.cpp's MMQ in size classes, the
    // experts of up to 8 rows in one launch, up to 16 in the next, up to 32 in the next, the rest in the last.  One
    // launch takes its tile width from its largest expert, and a tile computes that many columns whatever rows
    // are in them: with the median staged expert at 15 rows, a 64-column tile is 43% full.  The numbers are
    // ascending row counts.  The results are one launch's to the bit, except where llama.cpp's MMQ uses stream-k
    // (Mmq::split_exact()), which adds a row's partial sums in an order set by the launch's tile count.
    if (const char* v = std::getenv("STRATA_POLY_STAGED_TILES"); v && *v) {
        std::vector<int> t;
        bool ok = true;
        for (const char* p = v; *p && ok;) {
            char* end = nullptr;
            const long x = std::strtol(p, &end, 10);
            ok = end != p && x > 0 && x < 100000 && (t.empty() || x > t.back()) && (*end == ',' || !*end) &&
                 (int) t.size() < kMaxStagedTiles - 1;
            t.push_back((int) x);
            p = *end ? end + 1 : end;
        }
        if (ok) {
            staged_tiles_ = t;
        } else {
            std::fprintf(stderr, "strata-poly: STRATA_POLY_STAGED_TILES=%s is not ascending row counts a comma apart: "
                                 "ignored\n", v);
        }
    }
    // STRATA_FREEZE_EXPERTS=1: an answer's experts are the cache's as it was at the request's first token.  The
    // split between the card's experts and the CPU's (moe()) decides a layer's sums to the last bit - the two
    // paths add in different orders - so a greedy answer is a function of the request alone only while the split
    // is.  A freeze holds it still from the request's first token to its last: the followers' uploads are still
    // made and still arrive, but their slots are claimed only between requests, where nothing reads the cache.
    // What it costs: an expert the answer keeps using but that arrived after its first token is computed on the
    // CPU instead (the cache holds it, but the split is the first token's).  What it buys: greedy answers that
    // repeat bit for bit, which is what a test, a benchmark of quality or a comparison with another engine needs.
    freeze_experts_ = freeze_experts_asked_ = opt_.freeze_experts || std::getenv("STRATA_FREEZE_EXPERTS") != nullptr;
    // STRATA_GPU_EXPERTS=N, a test knob: the expert cache holds exactly the first N experts of the order it fills
    // in, frozen, with no uploads and no resize, for the whole run.  Two builds that leave different VRAM free (one
    // records more launches) cache different numbers of experts, and their sums differ wherever the split does; with
    // the same N (and the same --expert-profile, or none) they hold the same set, so that their logits can be
    // compared byte for byte.  An N the card cannot hold is an error, not a smaller set.
    if (const char* v = std::getenv("STRATA_GPU_EXPERTS"); v && *v) {
        char* end = nullptr;
        const long n = std::strtol(v, &end, 10);
        if (*end || n < 0 || n > (1L << 24))
            throw std::runtime_error(std::string("STRATA_GPU_EXPERTS=") + v + " is not a number of experts");
        if (split_ || opt_.expert_gpu >= 0)
            throw std::runtime_error("STRATA_GPU_EXPERTS is for one card's expert cache: not with --layer-split or "
                                     "--expert-gpu");
        expert_cap_ = (int) n;
        freeze_experts_ = freeze_experts_asked_ = true;
        opt_.expert_swaps = 0;
        std::fprintf(stderr, "strata-poly: STRATA_GPU_EXPERTS=%d: the expert cache holds the first %d experts of its "
                             "order, frozen, with no uploads and no resize, for the whole run\n",
                     expert_cap_, expert_cap_);
    } else if (freeze_experts_) {
        std::fprintf(stderr, "strata-poly: the experts' split is frozen for the engine's whole run "
                             "(STRATA_FREEZE_EXPERTS): greedy answers are a function of the request alone, "
                             "and the only cache change after load is the serve VRAM resize\n");
    }
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    st_ = st;
    for (void*& row : ev_row_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        row = ev;
    }
    ev_ = ev_row_[0];
    blas_ = blas_create(st_);
    // The card maps VRAM onto reserved addresses a piece at a time: the attention caches grow with the sequence,
    // and the expert cache has what a short one leaves (cover()).  With --kv-whole the attention caches are one
    // allocation each (kv_alloc()) and the expert cache still gives VRAM to other programs and takes it back
    // (mind_vram()): refilled in one allocation instead, it only ever got smaller.
    ck(cudaGetDevice(&device_), "cudaGetDevice");
#if defined(STRATA_USE_HIP) && defined(__linux__)
    // ROCm keeps a file descriptor open for each VMM handle (its dmabuf; librocdxg under WSL2): with --vram-pool's
    // 2 MiB handles the usual soft limit of 1,024 descriptors held an RX 9070 XT's pool to 2 GiB ("fail to import fd",
    // and the map fails as a card without VRAM does).  The soft limit goes up to the hard one (at most 2^20), as a
    // process may, and never down.
    if (rlimit r{}; getrlimit(RLIMIT_NOFILE, &r) == 0 && r.rlim_cur < r.rlim_max) {
        r.rlim_cur = std::max(r.rlim_cur, std::min<rlim_t>(r.rlim_max, rlim_t(1) << 20));
        (void) setrlimit(RLIMIT_NOFILE, &r);
    }
#endif
    CUdevice dev = 0;
    int vmm = 0;
    if (cuDeviceGet(&dev, device_) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) == CUDA_SUCCESS && vmm) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        vmm_ = cuMemGetAllocationGranularity(&grain_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
               grain_ > 0;
    }
    pool_on_ = opt_.vram_pool && vmm_ && opt_.kv_grow;
    if (opt_.vram_pool && !pool_on_)
        std::fprintf(stderr, "strata-poly: --vram-pool needs VMM and growing attention caches: off\n");
    if (pool_on_) pg_ = up_to(size_t(2) << 20, grain_);

    // the token table stays in the file, a row a token, and embed() reads a row in whatever type it has
    const TensorInfo* te = gg_->find("token_embd.weight");
    if (!te || te->shape.size() != 2 || (int64_t) te->shape[0] != n_embd_ || (int64_t) te->shape[1] != n_vocab_ ||
        type_block_values(te->type) <= 0 || n_embd_ % type_block_values(te->type) != 0 ||
        (te->type != kTypeF32 && !type_converts(te->type)))
        throw std::runtime_error("token_embd.weight must be [" + std::to_string(n_embd_) + ", " +
                                 std::to_string(n_vocab_) + "] in a type this engine converts to floats, it is " +
                                 (te ? shape_of(*te) : "missing"));
    tok_row_ = gg_->payload(*te) / (size_t) n_vocab_;
    tok_type_ = te->type;
    tok_embd_ = gg_->data(*te);

    // twice: the first pass adds up the arena, the second fills it
    for (int pass = 0; pass < 2; ++pass) {
        const int n_all = n_layer_ + (block_layer_ ? 1 : 0);
        layers_.assign((size_t) n_all, Layer{});
        n_moe_ = 0;
        dense_at_ = 0;
        begin_pass();
        for (int il = 0; il < n_all; ++il)
            if (runs(il)) load_layer(il);
        if (hi_ == n_layer_) {
            const TensorInfo* norm = gg_->find("output_norm.weight");
            if (const Names* a = norm ? nullptr : tensor_alias("output_norm.weight"))
                for (const char* other : a->other)
                    if (other && !norm) norm = gg_->find(other);
            out_norm_ = upload(norm, "output_norm.weight", n_embd_, 1, kF32);
            // a file without an output matrix ties the output to the token table
            const TensorInfo* head = gg_->find("output.weight");
            output_ = upload(head ? head : gg_->find("token_embd.weight"), "output.weight", n_embd_, n_vocab_,
                             kAnyMat);
        }
        if (draft_) load_drafter();
        if (pass == 0) {
            dense_bytes_ = dense_at_;
            dense_ = (uint8_t*) dmalloc(dense_bytes_ + 256, "the dense weights");
            zero_vram(dense_, dense_bytes_ + 256);   // between the tensors: kTail
        }
    }

    // A guess needs the recurrent layers' state as it was before it.  Where that is a copy in VRAM for every
    // guess of a window, a window is four tokens: the copies come out of what the experts would have.  A model
    // without such layers pays nothing for a longer one, and one whose layers leave the state after a few of a
    // window's tokens (snaps_spread()) no more than for four.  A repeat of the conversation can fill a window
    // of eight.  The layers are counted by now.
    if (opt_.window == 0) opt_.window = recurrent_floats() == 0 || snaps_spread() ? 8 : 4;
    opt_.window = std::min({opt_.window, opt_.prompt_batch, opt_.max_context});
    // A model that guesses its own block runs every layer for it, so on one card.  Off unless STRATA_BLOCK_GUESS=1:
    // a call of a few tokens costs about as much more as it has tokens, each with experts of its own, and the block is
    // one more such call a round.  With every expert on an RTX PRO 6000 Clover wrote 157 tokens/s after a 30K prompt
    // with its blocks and 204 without, and the policy asked for few (docs/BACKENDS.md).
    const char* bg = std::getenv("STRATA_BLOCK_GUESS");
    block_guess_ = ((bg && std::atoi(bg) == 1 && !draft_ && opt_.draft && block_mask() >= 0) || own_blocks()) &&
                   opt_.window > 1 && !split_;
    if (block_guess_) std::fprintf(stderr, "strata-poly: STRATA_BLOCK_GUESS: the model guesses its own blocks\n");
    alloc();
    if (!opt_.profile.empty()) read_profile(opt_.profile, (int) layers_.size(), n_expert_, prior_);
    fill_expert_cache();
#if !defined(STRATA_GPU_METAL)
    fill_expert_card();
#endif
    ensure_stage();
    if (mmq_ && !all_cached()) stage_pins();
    // The weights and the expert cache are written on st_ now (h2d(), zero_vram()): the load is not over until
    // those are done, so that everything after it - a read of what the cache holds, the first request's cover()
    // mapping the attention caches - finds what it expects.
    ck(cudaStreamSynchronize(cs(st_)), "the load");
    // A card of a split that has less free beside its layers than the reserve it keeps by default runs out of VRAM
    // in a later request, where one card alone can do nothing else.  It may be that another program held VRAM when
    // the split was made.  A larger reserve asked for (--vram-reserve-mib) only leaves the card without experts.
    if (split_ && !exp_) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const int need = default_reserve_mib(total_b, ram_tight_);
        if (free_b <= (size_t(need) << 20))
            throw std::runtime_error("layers " + std::to_string(lo_) + " to " + std::to_string(hi_ - 1) + " leave " +
                                     std::to_string(free_b >> 20) + " MiB of VRAM free, less than the " +
                                     std::to_string(need) + " MiB the card needs beside them: give it fewer layers "
                                     "(--layer-split), or free VRAM another program holds");
    }
    // A card that holds every expert has VRAM to spare, and the attention caches take the whole context's rows
    // there now when they fit: VRAM left free is what another program takes, and the caches would then grow out of
    // the expert cache (cover()), which does not get it back when that program ends.  A piece of the expert cache
    // (kArenaPiece) is the margin, so that cover() cuts nothing.
    if (!kv_.empty() && all_cached()) {
        size_t more = 0;
        for (const Kv& k : kv_) more += k.reserved - k.mapped;
        for (const Seq& q : seqs_)
            for (const Kv& k : q.kv) more += k.reserved - k.mapped;
        if ((int64_t) ((more + kArenaPiece) >> 20) + reserve_mib_ <= vram_free_mib()) {
            for (int s = sequences() - 1; s >= 0; --s) {   // ends with the first one bound
                if (s != bound_) {
                    park_bound();
                    take_bound(s);
                }
                cover(opt_.max_context);
            }
        }
    }
    const int threads = opt_.threads > 0 ? opt_.threads : default_threads();
    mind_memory();
    // the routed experts in the mapped files, and of them the ones the CPU reads through the mapping: those the
    // card does not have, as many as Options::ram_experts_mib keeps
    std::vector<std::pair<const uint8_t*, size_t>> spans, cpu;
    const auto to = [](auto& v) { return [&v](const uint8_t* p, size_t n) { v.emplace_back(p, n); }; };
    outside_bytes_ = 0;
    for (const Layer& L : layers_)
        if (L.moe)
            expert_runs(L, n_expert_, [&](int e) { return !L.slot[(size_t) e]; },
                        [&](const uint8_t*, size_t n) { outside_bytes_ += n; });
    // Options::ram_experts_mib below -1, the default: every expert outside VRAM in RAM when they fit in what is free,
    // with kRamHeadroom left for the rest of the process; else a budget of what is free less that.  Through the
    // mapping, experts that do not fit are read back a fault at a time: on the lab's machine held to 12 GB, Qwen3.6
    // UD-Q3_K_M wrote 51, 73 and 99 tokens/s (a chat, after a 24,000-token prompt, a chat again) with every expert
    // mapped and 61, 88 and 120 with a budget of 6 GiB.
    //
    // Under a cgroup's limit (a container) as well: on lab-241 under MemoryMax=8G, Qwen3.6 UD-Q3_K_M at 8 GB of VRAM
    // with 1.5 GiB left budgeted 6,389 MiB and peaked at 8,034 MiB; with the 4 GiB, 3,834 MiB, it peaked at 5,507
    // and still read 1.27 and wrote 2.13 times as fast as poly.10 with the same budget.
    {
        bool own = false;
        ram_available(&own);
        pool_uncached_ = own;   // read_file()
    }
    if (const char* dump = std::getenv("STRATA_COVER_DUMP"); dump && dump[0] == '1')   // measuring only
        for (int pct : {70, 75, 80, 85, 90, 93, 95, 96, 97, 98, 99})
            std::fprintf(stderr, "strata-poly: STRATA_COVER_DUMP: %d%% of the use outside VRAM in %zu MiB, of %zu MiB "
                                 "outside\n", pct, covering(pct) >> 20, outside_bytes_ >> 20);
    if (opt_.ram_experts_mib < -1) {
        constexpr size_t kRamHeadroom = size_t(4) << 30;
        size_t avail = ram_available();
#ifdef _WIN32
        // fill_expert_cache() let the card's experts go from the working set to the standby list, which Windows
        // counts as available: the budget would take for the CPU's experts the RAM the load just saved.  They are
        // counted as used, as they were before they went.  On the laptop Qwen3.6 UD-Q3_K_M saw 11.0 GiB free with
        // its 3.6 GiB of cached experts let go, and 7.3 GiB held.  Linux counts file pages as available either way.
        const size_t held = cached_file_bytes();
        if (avail) avail = avail > held ? avail - held : 1;   // 0 is "cannot tell"
#endif
        opt_.ram_experts_mib = -1;
        if (avail && !opt_.lock_experts && outside_bytes_ + kRamHeadroom > avail) {
            size_t budget = std::min(avail > kRamHeadroom ? avail - kRamHeadroom : 0, outside_bytes_);
            std::fprintf(stderr, "strata-poly: %.1f GiB of RAM free for %.1f GiB of experts outside VRAM: they do not fit, "
                                 "so the ones used most stay in RAM (--ram-experts-mib -1 maps them all)\n",
                         (double) avail / (1 << 30), (double) outside_bytes_ / (1 << 30));
#ifdef _WIN32
            // No more than the experts that take kCover of the profile's uses outside VRAM, and no less than three
            // quarters of the budget.  On the RTX 4070 laptop, with the profiles its runs had written, Qwen3.6
            // UD-Q3_K_M wrote as fast with 75% of the budget (96.3% of those uses) and 10% slower with 50% (83.0%);
            // LFM2.5 UD-Q4_K_XL, whose few experts are all used, wrote 26% slower at 75% of it.  Where the experts
            // fit, every one stays: Gemma 4 UD-Q3_K_M, put under a budget there, wrote 14-36% slower.
            //
            // kCover was 96.  In the first runs after an install LFM2.5's profile put 96% of its uses in 5,835 MiB
            // of a 7,100 MiB budget, and with that pool it wrote 42.2 tokens/s after a 30,156-token prompt where
            // poly.10 wrote 49.3; a few runs later the same profile wanted 7,400 MiB for 96%, no pool was cut, and
            // it wrote 54.4.  At 99 Qwen3.6's pool is still the three quarters (5,694 MiB cover 99% there, 6,103
            // MiB are three quarters), and LFM2.5 wants 8,171 MiB, more than its budget.
            constexpr int kCover = 99;
            const size_t most = std::max(covering(kCover), budget / 4 * 3);
            if (most < budget) {
                budget = most;
                std::fprintf(stderr, "strata-poly: %zu MiB of RAM for the experts that take %d%% of the use outside "
                                     "VRAM, or three quarters of the budget\n", most >> 20, kCover);
            }
#endif
            opt_.ram_experts_mib = std::max<int64_t>(1024, (int64_t) (budget >> 20));
        }
    }
    // Under a budget the experts swapped into VRAM come from the pool or the disk, as the CPU's own do, and the one
    // that goes out is read back for the CPU.  The cache that follows the answer is worth those reads all the same.
    // On lab-241 (RTX 4090 D, RAM held by a scope, the file's pages dropped first), the answers after four long
    // prompts found 30-76% of their experts in VRAM with none and 63-98% with 64, and wrote, tokens/s, with none
    // and with 64: Qwen3.6 UD-Q3_K_M at 8 GB of VRAM and 6,000 MiB for the pool under 8 GiB 120 and 174, at 4.8 GB
    // and 5,120 MiB 87 and 98, UD-Q4_K_M there 31 and 38, Gemma 4 UD-Q6_K_XL with 5,120 MiB under 11 GiB at 6.9 GB
    // 100 and 137, at 2.9 GB 42 and 78.  In unified memory a swap is a copy within RAM: on the M5 Ultra held to 6-7
    // GiB, 64 of them wrote 3-5% faster.
    if (opt_.expert_swaps < 0) opt_.expert_swaps = 64;
    if (opt_.ram_experts_mib >= 0) {
        // each size of slot takes the share of the budget its experts outside VRAM have of all of those
        pool_size_of_.assign(layers_.size(), 0);
        std::vector<size_t> outside;   // by size
        for (size_t il = 0; il < layers_.size(); ++il) {
            const Layer& L = layers_[il];
            if (!L.moe) continue;
            const size_t b = (L.gu2() + L.d_bytes + 63) / 64 * 64;
            size_t c = 0;
            while (c < pool_sizes_.size() && pool_sizes_[c].bytes != b) ++c;
            if (c == pool_sizes_.size()) {
                pool_sizes_.push_back({b, 0, 0, 0, 0});
                outside.push_back(0);
            }
            pool_size_of_[il] = (uint8_t) c;
            for (int e = 0; e < n_expert_; ++e) outside[c] += L.slot[(size_t) e] ? 0 : b;
        }
        // Under a cgroup's limit (pool_uncached_) the whole budget, so that kept_ (ram_keep()) has none of it: an
        // expert kept in the mapping is a page of the file cache, which the limit counts and the system takes back
        // when it is reached, and the CPU then faults it in a few pages at a time.  pool_'s slots are the process's
        // own memory and stay.  Elsewhere a quarter, and kept_ the rest: no limit charges the file cache, and pool_
        // is memory the process holds for the run where kept_'s pages are the file cache's.  With the whole budget in
        // pool_ the RTX 4070 laptop's engine and server committed 11,678 MiB where poly.10's 9,476, at the same
        // 3,072 MiB, and on the Mac held to 16 GB Qwen3.6 UD-Q2_K_XL's footprint had 1.1 GB more of it.
        size_t want = (size_t) opt_.ram_experts_mib << 20;
        if (!pool_uncached_) want /= 4;
        size_t all = 0, slots = 0;
        for (size_t b : outside) all += b;
        for (size_t c = 0; c < pool_sizes_.size(); ++c) {
            // a call's experts are of one layer, so one size; one that finds every slot of its size held by this
            // call and the next goes through the mapping (pool_take())
            PoolSize& z = pool_sizes_[c];
            const double part = all ? (double) outside[c] / (double) all : 0.0;
            z.n = std::max<size_t>((size_t) ((double) want * part) / z.bytes, 64);
            const size_t turn = std::min(z.n, (size_t) 2 * kCpuRows * kMaxUsed);   // a call's and the next one's
            z.prot_max = z.n - std::max(turn, z.n / 4);
            z.first = slots;
            z.at = pool_bytes_;
            slots += z.n;
            pool_bytes_ += z.n * z.bytes;
        }
        pool_.reset(new uint8_t[pool_bytes_]);
        pool_of_.assign(layers_.size() * (size_t) n_expert_, -1);
        pool_who_.assign(slots, -1);
        pool_tick_.assign(slots, 0);
        pool_ref_.assign(slots, 0);
        pool_prot_.assign(slots, 0);
        pool_pin_.assign(slots, 0);
        prompt_use_.assign(layers_.size() * (size_t) n_expert_, 0.f);
        pool_pending_.reset(new std::atomic<int>[slots]);
        pool_failed_.reset(new std::atomic<uint8_t>[slots]);
        for (size_t v = 0; v < slots; ++v) pool_pending_[v].store(0), pool_failed_[v].store(0);
        // 8: on the laptop 4 left a call waiting for its reads 8% longer; 12 were no better than 8
        for (int t = 0; t < 8; ++t) pool_threads_.emplace_back([this] { pool_loop(); });
        pool_state_.reset(new std::atomic<uint8_t>[slots]);
        for (size_t s = 0; s < slots; ++s) pool_state_[s].store(0);
        // the next layer's top 8: the best of 0, 8, 12 and 16, and of 8 with the layer after's top 8 as well
        prefetch_k_ = 8;
        const size_t part = 16 * (size_t) n_expert_;   // a part's rows: up to 16
        if (!ahead_dev_) {
            ahead_dev_ = (float*) dmalloc(8 * part * sizeof(float), "scratch");
            h_ahead_ = (float*) hmalloc(8 * part * sizeof(float), "pinned memory");
        }
    }
    const std::vector<uint8_t> keep = ram_keep();
    if (opt_.ram_experts_mib >= 0) {
        kept_ = keep;
        for (const Shards* sh : {gg_.get(), dg_.get()}) {
            if (!sh) continue;
            const auto v = sh->views();
            for (size_t i = 0; i < v.size(); ++i)
                if (v[i].first) views_.push_back({v[i].first, v[i].second, sh->paths()[i]});
        }
    }
    for (size_t il = 0; il < layers_.size(); ++il) {
        const Layer& L = layers_[il];
        if (!L.moe) continue;
        expert_bytes(L, 0, n_expert_, to(spans));
        const uint8_t* k = keep.data() + il * (size_t) n_expert_;
        expert_runs(L, n_expert_, [&](int e) { return !L.slot[(size_t) e] && k[e]; }, to(cpu));
    }
    if (!workers_) {   // a later card's engine has the first one's
        if (opt_.stay_near) {   // before any thread that reads the experts is started
            std::vector<std::string> files = gg_->paths();
            if (dg_) files.insert(files.end(), dg_->paths().begin(), dg_->paths().end());
            node_ = stay_near(spans, files, threads, node_share_);
        }
        workers_ = std::make_shared<Workers>(threads);
    }
    if (!aside_) aside_ = std::make_unique<Aside>();
    // save_state()'s pinned memory, and read()'s for the checkpoints inside a read, with the load: made at the
    // first checkpoint, it held a request up 32 ms
    if (!h_state_) {
        h_state_bytes_ = card_state_bytes();
        h_state_ = (uint8_t*) hmalloc(h_state_bytes_, "pinned memory");
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_saved_ = ev;
    }
    route_tmp_.assign((size_t) workers_->threads() * 2 * n_expert_, 0.f);
    if (opt_.cache_bias > 0.f) bias_rl_.assign((size_t) n_expert_, 0.f);
    start_swaps();
    reset();
    build_mirror();
    std::vector<std::pair<int, int>> outside;
    if (opt_.lock_experts && mirror_of_.empty()) {   // the mirror's pages are pinned already
        locked_.assign(layers_.size() * (size_t) n_expert_, 0);
        for (size_t il = 0; il < layers_.size(); ++il)
            for (int e = 0; layers_[il].moe && e < n_expert_; ++e)
                if (!layers_[il].slot[(size_t) e]) outside.emplace_back((int) il, e);
    }
    // nothing is read from the files when every expert is in VRAM, nor through the mapping for the mirror
    if (!all_cached() && mirror_of_.empty()) {
#if defined(STRATA_GPU_METAL)
        if (opt_.metal_balanced) {
            if (metal_warm_bytes_ > 0) {
                struct Warm {
                    int64_t used;
                    std::pair<const uint8_t*, size_t> gate, up, down;
                };
                std::vector<Warm> cold;
                for (size_t il = 0; il < layers_.size(); ++il) {
                    const Layer& l = layers_[il];
                    if (!l.moe) continue;
                    for (int e = 0; e < n_expert_; ++e) {
                        if (l.slot[(size_t) e]) continue;
                        const size_t i = il * n_expert_ + size_t(e);
                        const int64_t used = (prior_.empty() ? 0 : prior_[i]) + use_[i];
                        cold.push_back({used, {l.exp_gate + size_t(e) * l.gu_stride, l.gu_bytes},
                                             {l.exp_up + size_t(e) * l.u_stride, l.u_bytes},
                                             {l.exp_down + size_t(e) * l.d_bytes, l.d_bytes}});
                    }
                }
                std::stable_sort(cold.begin(), cold.end(), [](const Warm& a, const Warm& b) { return a.used > b.used; });
                const size_t warm_cap = metal_warm_bytes_;
                mapper_ = std::thread([this, cold = std::move(cold), warm_cap] {
                    constexpr size_t page = 16384, check_every = size_t(16) << 20;
                    size_t touched = check_every, total_warmed = 0;
                    bool warned = false;
                    volatile uint8_t seen = 0;
                    for (const Warm& expert : cold) for (const auto& span : {expert.gate, expert.up, expert.down}) {
                        for (size_t at = 0; at < span.second; at += page) {
                            if (map_quit_.load(std::memory_order_relaxed) || total_warmed >= warm_cap) return;
                            if (touched >= check_every) {
                                const auto available = strata::core::available_host_bytes();
                                if (available && *available <= kMetalHostReserve) return;
                                if (!available && !warned) {
                                    std::fprintf(stderr, "strata-poly: host memory is unavailable while warming uncached experts\n");
                                    warned = true;
                                }
                                touched = 0;
                            }
                            seen = uint8_t(seen + span.first[at]);
                            touched += page;
                            total_warmed += page;
                        }
                        if (span.second) seen = uint8_t(seen + span.first[span.second - 1]);
                    }
                });
            }
        } else
#endif
        mapper_ = std::thread([this, cpu = std::move(cpu), outside = std::move(outside)] {
            map_pages(cpu, *gg_, map_quit_);
            if (!outside.empty()) lock_outside(outside);
        });
    }
}

void Engine::load_drafter() {
    check_draft_block();
    auto part = [&](const char* n, int64_t a, int64_t b, Want w) {
        return upload(tensor(n_layer_, n), blk(n_layer_, n), a, b, w);
    };
    d_proj_ = part("nextn.eh_proj.weight", 2 * n_embd_, n_embd_, kAnyMat);
    d_enorm_ = part("nextn.enorm.weight", n_embd_, 1, kF32);
    d_hnorm_ = part("nextn.hnorm.weight", n_embd_, 1, kF32);
    // without a norm of its own, the block's output goes through the model's
    d_head_ = tensor(n_layer_, "nextn.shared_head_norm.weight")
                  ? part("nextn.shared_head_norm.weight", n_embd_, 1, kF32)
                  : out_norm_;
}

// Per-token buffers hold buf_rows_ tokens, row after row.
void Engine::shared_scratch() {
    const size_t n = (size_t) buf_rows_, e = n * n_embd_;
    const size_t ff = n * std::max({n_ff_, n_ff_shexp_, 2 * n_ff_exp_});
    // the draft block works in it too (draft_input()); read() keeps two calls' rows in it
    r_ = fbuf(std::max(streams(), draft_ ? 3 : 1) * (size_t) (read_rows_ > rows_ ? 2 : 1) * n * n_embd_);
    cur_ = fbuf(e);
    xn_ = fbuf(e);
    ao_ = fbuf(e);
    fo_ = fbuf(e);
    g_ = fbuf(ff);   // a routed expert's gate and up rows come out of one launch, back to back
    u_ = fbuf(ff);
    h_ = fbuf(ff);
    d_ = fbuf((size_t) card_rows() * n_embd_);   // the routed experts of a few tokens, until they are added up
    rl_ = fbuf(n * n_expert_);
    logits_ = fbuf((size_t) n_vocab_);
    if (buf_rows_ >= kStageRows && bf16_max_ > 0) wf_ = fbuf(bf16_max_);   // a BF16 matrix as floats, for lin()
}

void Engine::use_scratch(bool batch) {
    scratch_ = batch ? scratch_batch_ : scratch_few_;
    buf_rows_ = batch ? rows_ : kStageRows - 1;
    scratch_at_ = 0;
    shared_scratch();
    assign_scratch();
    prompt_scratch_ = batch;
}

// The sequence's state, the caches and every buffer, once the weights are known.
void Engine::alloc() {
    // --dump-layers prints one token's stages
    rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);
    const size_t most = std::min<size_t>(kReadBytes / ((size_t) streams() * n_embd_ * sizeof(float)), opt_.max_context);
    read_rows_ = rows_ < kStageRows ? rows_ : std::max(rows_, (int) most / rows_ * rows_);
    rec_n_ = recurrent_floats();
    state_n_ = rec_n_ + (draft_ ? (size_t) n_embd_ : 0);
    const size_t state_floats = state_n_ + (draft_ ? (size_t) rows_ * n_embd_ : 0);
    state_ = (float*) dmalloc(state_floats * sizeof(float), "the recurrent state");
    if (draft_) {
        hn_ = state_ + rec_n_;
        h_guess_ = (Likeliest*) hmalloc(sizeof(Likeliest), "pinned memory");   // likeliest() writes it
    }
    if (block_guess_) h_block_ = (Likeliest*) hmalloc((size_t) opt_.window * sizeof(Likeliest), "pinned memory");
    // window(): the state after tokens of it, for keep(), and every token's logits
    snap_places_ = snaps_spread() ? std::min(opt_.window - 1, kSnapPlaces) : opt_.window - 1;
    if (snap_places_ > 0)
        snap_ = (float*) dmalloc(size_t(snap_places_) * rec_n_ * sizeof(float), "the recurrent state");
    // step() with guesses: every sequence's state after its first row
    if (opt_.batch_mtp && opt_.batch > 0 && rec_n_ > 0)
        step_snap_ = (float*) dmalloc(size_t(opt_.batch + 1) * rec_n_ * sizeof(float), "the recurrent state");
    if (opt_.batch_mtp && opt_.batch > 0 && draft_) {
        step_h_ = (float*) dmalloc(size_t(opt_.batch + 1) * n_embd_ * sizeof(float), "scratch");
        h_guesses_ = (Likeliest*) hmalloc(size_t(opt_.batch + 1) * sizeof(Likeliest), "pinned memory");
    }
    // window()'s, step()'s (two rows a sequence with guesses)
    const size_t logit_rows = (size_t) std::max(opt_.window, (opt_.batch_mtp ? 2 : 1) * (opt_.batch + 1));
    wlogits_ = (float*) dmalloc(logit_rows * n_vocab_ * sizeof(float), "scratch");
    h_wlogits_ = (float*) hmalloc(logit_rows * n_vocab_ * sizeof(float), "pinned memory");
    d_picks_ = (Likeliest*) dmalloc(logit_rows * sizeof(Likeliest), "scratch");   // step_picks(), eval_picks()
    h_picks_ = (Likeliest*) hmalloc(logit_rows * sizeof(Likeliest), "pinned memory");
    alloc_caches();
    cover(1);   // a short conversation's rows, before the expert cache is sized
    // --batch: the other sequences, each with a state and caches of its own (bind())
    seqs_.resize((size_t) opt_.batch + 1);
    for (int s = 1; s <= opt_.batch; ++s) {
        park_bound();
        bound_ = s;
        pos_ = kv_rows_ = kv_reach_ = 0;
        ahead_pos_ = -1;
        state_ = (float*) dmalloc(state_floats * sizeof(float), "the recurrent state");
        ck(cudaMemset(state_, 0, state_n_ * sizeof(float)), "the recurrent state");
        if (draft_) hn_ = state_ + rec_n_;
        alloc_caches();
        cover(1);
    }
    if (bound_ != 0) {
        park_bound();
        take_bound(0);
    }

    // A prompt batch runs a layer's routed experts through MMQ when the card has a tile for each of their types
    // (batched()).  The model's layers: the draft block's experts never run a batch at a time.
    const size_t n = (size_t) rows_;
    bool tiles = false, mixed_tiles = false;
    size_t one_expert = 0;
    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;
        // a mixed layer multiplies gate and up apart (moe_rows()), each a matrix of n_ff rows
        const bool gu_fits = L.mixed() ? Mmq::fits((int) L.gu_type, n_ff_exp_, n_embd_) &&
                                             Mmq::fits((int) L.u_type, n_ff_exp_, n_embd_)
                                       : Mmq::fits((int) L.gu_type, 2 * n_ff_exp_, n_embd_);
        L.mmq = opt_.gpu_experts && rows_ >= kStageRows && gu_fits && Mmq::fits((int) L.d_type, n_embd_, n_ff_exp_);
        tiles = tiles || L.mmq;
        mixed_tiles = mixed_tiles || (L.mmq && L.mixed());
        one_expert = std::max(one_expert, L.gu_step + L.d_bytes);
    }
    if (tiles) mmq_ = std::make_unique<Mmq>();

    buf_rows_ = rows_;
    shared_scratch();   // the first pass only counts
    assign_scratch();
    const size_t batch_floats = scratch_at_;
    // The scratch in two layouts where the prompt's buffers are lent (below) and the architecture can: the one for
    // a prompt's batch is lent with them, and a call of a few tokens has a layout of its own.  At a prompt batch of
    // 2,048 Qwen3.6's scratch is 450 MB, of which the few tokens' layout keeps some 70 (the attention's scores).
    const int most_step = (opt_.batch + 1) * (opt_.batch_mtp ? 2 : 1);
    // Whether the prompt's buffers are lent (prompt_lends_, below).  Not on ROCm on Linux, which gives VRAM back only
    // with its addresses (see below): there nothing is lent, so the scratch keeps one layout, and a call of a few
    // tokens never finds the prompt's layout left in place (which kept graph replay off for every later step).
#ifdef STRATA_VMM_RELEASE_WITH_ADDRESSES
    const bool lends = false;
#else
    const bool lends = vmm_ && !split_ && std::max(opt_.window, most_step) < kStageRows;
#endif
    two_scratch_ = scratch_layouts() && mmq_ && vmm_ && !split_ && rows_ >= kStageRows &&
                   std::max(opt_.window, most_step) < kStageRows && lends;
    if (two_scratch_) {
        buf_rows_ = kStageRows - 1;
        scratch_at_ = 0;
        shared_scratch();   // counts the few tokens' layout
        assign_scratch();
        scratch_few_ = (float*) dmalloc(scratch_at_ * sizeof(float), "scratch");
    } else {
        scratch_ = (float*) dmalloc(batch_floats * sizeof(float), "scratch");
        scratch_at_ = 0;
        shared_scratch();
        assign_scratch();
    }

    const int q_max = std::max({n_embd_, n_ff_, n_ff_exp_, n_ff_shexp_, widest(), 256});
    xq_ = dmalloc(n * kernels::native_q8_1_bytes(n_embd_), "scratch");
    q8_ = dmalloc(n * kernels::native_q8_1_bytes((q_max + 255) / 256 * 256), "scratch");

    h_r_ = (float*) hmalloc(n * streams() * n_embd_ * sizeof(float), "pinned memory");
    const size_t at_rows = (size_t) std::max(opt_.window, opt_.batch + 1);   // window()'s tokens, step()'s
    at_ = (long long*) dmalloc(at_rows * 2 * sizeof(long long), "scratch");
    h_at_ = (long long*) hmalloc(at_rows * 2 * sizeof(long long), "pinned memory");
    h_rl_ = (float*) hmalloc(n * n_expert_ * sizeof(float), "pinned memory");
    h_x_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_cpu_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_logits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");

    if (mmq_) {
        const size_t rr = n * n_used_,
                     tab = (3 * rr + 3 * (size_t) n_expert_ + 3 + kClasses * ((size_t) n_expert_ + 1) +
                            2 * kMaxStagedTiles * (2 * (size_t) n_expert_ + 2)) * sizeof(int32_t);
        // What the products take beside their arguments (Mmq::work(): mmq_own.cu's FP16 and BLAS paths): the
        // experts', whose rows go by ids, and lin()'s, with the quantized matrices upload() kept, the output's only
        // where a step can have kStageRows rows.  Made by the products themselves, it came out of the reserve the
        // expert cache leaves, while a prompt was read: with less than that free, the first long prompt failed.
        // Where the prompt's buffers are lent, it has addresses of its own (work_map_), mapped only while a prompt
        // is read (take_prompt()), as those buffers are: the expert cache has it until the first prompt and between
        // prompts.  Elsewhere it is one of those buffers, for good.
        Mmq::Work need;
#if !defined(STRATA_GPU_METAL)
        const auto take = [&](uint32_t type, int64_t w_rows, int64_t w_cols, int64_t rows, bool by_ids, int64_t ld) {
            const Mmq::Work w = Mmq::work((int) type, w_rows, w_cols, rows, by_ids, ld);
            need.floats = std::max(need.floats, w.floats);
            need.halves = std::max(need.halves, w.halves);
        };
        for (const Layer& L : layers_) {
            if (!L.mmq) continue;
            if (L.mixed()) {
                take(L.gu_type, n_ff_exp_, n_embd_, (int64_t) rr, true, 2 * n_ff_exp_);
                take(L.u_type, n_ff_exp_, n_embd_, (int64_t) rr, true, 2 * n_ff_exp_);
            } else {
                take(L.gu_type, 2 * n_ff_exp_, n_embd_, (int64_t) rr, true, 2 * n_ff_exp_);
            }
            take(L.d_type, n_embd_, n_ff_exp_, (int64_t) rr, true, n_embd_);
        }
        const bool wide_steps = std::max(opt_.window, most_step) >= kStageRows;
        for (const Dev& d : mats_)
            if (d.n_in <= q_max && (d.p != output_.p || wide_steps))
                take(d.type, d.n_out, d.n_in, rows_, false, d.n_out);
#endif
        const size_t floats_bytes = (need.floats * sizeof(float) + 255) / 256 * 256,
                     work = floats_bytes + (need.halves * 2 + 255) / 256 * 256;
        // The buffers only a call of kStageRows rows or more uses (moe_rows(), lin_raw()'s products), side by side:
        // where they can be mapped apart (VMM) and the answer's calls never have that many rows, the expert cache
        // has their VRAM between prompts (lend_prompt()), 300 MB of Qwen3.6's at a prompt batch of 2,048.  Not on ROCm
        // on Linux, which gives VRAM back only with its addresses (vmm_unmap()), as the staging buffer's widening is
        // not: with the buffers lent, strata-poly crashed (SIGSEGV inside hipLaunchKernel, the dequantize launch of a
        // prompt's dense product into the lent work memory) in 15 of 16 runs of Inkling 2 Mini IQ4_NL on an RX 9070 XT
        // with 2,443 of its 3,072 experts in VRAM, ROCm 7.2.4.
        prompt_lends_ = lends;
        const size_t sizes[] = {rr * 2 * n_ff_exp_ * sizeof(float), rr * n_ff_exp_ * sizeof(float),
                                rr * n_embd_ * sizeof(float), rr * sizeof(float),
                                Mmq::q8_bytes((int64_t) rr, n_embd_), Mmq::q8_bytes((int64_t) rr, n_ff_exp_),
                                mixed_tiles ? Mmq::q8_bytes((int64_t) rr, n_embd_) : 0, Mmq::q8_bytes(rows_, q_max),
                                two_scratch_ ? batch_floats * sizeof(float) : 0, prompt_lends_ ? 0 : work};
        size_t at[10], total = 0;
        for (int i = 0; i < 10; ++i) {
            at[i] = total;
            total += (sizes[i] + 255) / 256 * 256;
        }
        uint8_t* base = nullptr;
        if (prompt_lends_) {
            vmm_reserve(prompt_map_, total);
            vmm_map(prompt_map_, total, SIZE_MAX, "a prompt's buffers");
            base = (uint8_t*) prompt_map_.base;
            prompt_bytes_ = total;
        } else {
            base = (uint8_t*) dmalloc(total, "scratch");
        }
        gu_ = (float*) (base + at[0]);
        hx_ = (float*) (base + at[1]);
        dx_ = (float*) (base + at[2]);
        hf_ = (float*) (base + at[3]);
        mq_ = base + at[4];
        mq2_ = base + at[5];
        if (mixed_tiles) mqu_ = base + at[6];
        mqx_ = base + at[7];
        if (two_scratch_) {
            scratch_batch_ = (float*) (base + at[8]);
            use_scratch(false);
        }
#if !defined(STRATA_GPU_METAL)
        uint8_t* w = base + at[9];
        if (prompt_lends_ && work > 0) {
            work_bytes_ = up_to(work, grain_);
            vmm_reserve(work_map_, work_bytes_);
            w = (uint8_t*) work_map_.base;
            work_lent_ = true;
        }
        mmq_->give(need.floats ? (float*) w : nullptr, need.floats, need.halves ? w + floats_bytes : nullptr,
                   need.halves, blas_);
#endif
        d_ids_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        d_wts_ = (float*) dmalloc(rr * sizeof(float), "scratch");
        h_ids_ = (int32_t*) hmalloc(rr * sizeof(int32_t), "pinned memory");
        // the weights of step()'s rows, back from the card where a token alone routes there (the doorbell)
        if (bells() && opt_.batch > 0) h_wts_ = (float*) hmalloc(rr * sizeof(float), "pinned memory");
        d_slots_ = (void**) dmalloc((size_t) n_layer_ * n_expert_ * sizeof(void*), "scratch");
#if !defined(STRATA_GPU_METAL)
        d_far_ = (void**) dmalloc((size_t) n_layer_ * n_expert_ * sizeof(void*), "scratch");
        d_tab_ = (void**) dmalloc((size_t) n_expert_ * sizeof(void*), "scratch");
        d_fetch_ = (FarFetch*) dmalloc((size_t) card_rows() * sizeof(FarFetch), "scratch");
        d_nfetch_ = (int*) dmalloc(sizeof(int), "scratch");
        d_use_ = (int32_t*) dmalloc(layers_.size() * (size_t) n_expert_ * sizeof(int32_t), "scratch");
        ck(cudaMemset(d_use_, 0, layers_.size() * (size_t) n_expert_ * sizeof(int32_t)), "expert uses");
#if !defined(STRATA_USE_HIP)
        // the doorbell (bell_on()): every layer's experts by picks on the card, or none of them; STRATA_POLY_BELL=0: none
        const char* bell_env = std::getenv("STRATA_POLY_BELL");
        bool bell = bells() && !split_ && !(bell_env && bell_env[0] == '0');
        size_t most = 0;
        for (int il = 0; il < n_layer_ && bell; ++il) {
            const Layer& L = layers_[(size_t) il];
            if (!L.moe) continue;
            bell = kernels::native_mmvq_ids_supported((int) L.gu_type) &&
                   kernels::native_mmvq_ids_supported((int) L.u_type) && kernels::native_mmvq_ids_supported((int) L.d_type);
            most = std::max(most, L.gu2() + L.d_bytes);   // (slot_bytes is not known before fill_expert_cache())
        }
        if (bell && most > 0) {
            h_bell_ = (ExpertBell*) hmalloc(sizeof(ExpertBell), "pinned memory");
            std::memset((void*) h_bell_, 0, sizeof(ExpertBell));
            d_bell_seq_ = (uint32_t*) dmalloc(2 * sizeof(uint32_t), "scratch");
            ck(cudaMemset(d_bell_seq_, 0, 2 * sizeof(uint32_t)), "the doorbell");
            d_zslots_ = (void**) dmalloc((size_t) n_layer_ * n_expert_ * sizeof(void*), "scratch");
            most = (most + 4096 + kTail) / 256 * 256;   // past an expert's bytes, as far as a slot's padding goes
            zero_slot_ = (uint8_t*) dmalloc(most, "scratch");
            ck(cudaMemset(zero_slot_, 0, most), "the doorbell");
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            ev_bell_ = ev;
        }
#endif
#endif
        tab_ = (int32_t*) dmalloc(tab, "scratch");
        h_tab_ = (int32_t*) hmalloc(tab, "pinned memory");
        if (expert_tiles_) {
            d_slot_at_ = (int64_t*) dmalloc((size_t) n_expert_ * sizeof(int64_t), "scratch");
            h_slot_at_ = (int64_t*) hmalloc((size_t) n_expert_ * sizeof(int64_t), "pinned memory");
        }
        if (mmq_) mmq_->mirror(tab_, h_tab_, tab);   // the bounds moe_rows() hands mul() are in tab_, written from h_tab_
        // the ids of the gate and up products, with Mmq::kIdsTail behind the call's last row (d_place, the down
        // products' ids, has the rest of tab_ behind it)
        const size_t ids = rr + (size_t) Mmq::kIdsTail;
        iota_ = (int32_t*) dmalloc(ids * sizeof(int32_t), "scratch");
        d_ex_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        h_ex_ = (int32_t*) hmalloc(rr * sizeof(int32_t), "pinned memory");
        std::vector<int32_t> ramp(ids);
        std::iota(ramp.begin(), ramp.end(), 0);
        ck(cudaMemcpy(iota_, ramp.data(), ids * sizeof(int32_t), cudaMemcpyHostToDevice), "scratch");
        // the next MoE layer's router on a layer's input (moe_router()), for vram_ahead() or pool_ahead(): here, so
        // that the expert cache is sized without it
        const size_t part = 16 * (size_t) n_expert_;
        ahead_dev_ = (float*) dmalloc(8 * part * sizeof(float), "scratch");
        h_ahead_ = (float*) hmalloc(8 * part * sizeof(float), "pinned memory");

        // One layer's experts pass through it at a time, so it never holds more than a layer has.  A model of
        // small experts has a few hundred MiB a layer, and the rest of the default would be VRAM that the
        // expert cache of a small card does without.
        //
        // By default, where a read can widen it (VMM, one card), it is 128 MiB and a read of several batches
        // widens it to a layer: it is the expert cache's VRAM while an answer is written.  On the laptop Gemma 4
        // cached 1,166 experts in place of 1,013 and Qwen3.6 2,712 in place of 2,444; prompts of 3,000 to 6,000
        // tokens read up to 2% slower.  A batch alone (up to --prompt-batch tokens) reads with the narrow one: 300
        // and 1,000 tokens no slower.  Without the widening, 128 MiB read 6,000 tokens 20% slower.  ROCm keeps the old
        // default: it gives back VRAM only with its addresses (vmm_unmap()), and a read's widening is untested there.
        const bool widens = vmm_ && !split_;
#ifdef STRATA_VMM_RELEASE_WITH_ADDRESSES
        const bool lend = false;
#else
        const bool lend = widens;
#endif
        const int stage_mib = opt_.stage_mib > 0 ? opt_.stage_mib : lend ? 128 : 2048;
        const int read_mib = opt_.stage_read_mib >= 0 ? opt_.stage_read_mib : opt_.stage_mib > 0 || !lend ? 0 : 2048;
        stage_bytes_ = std::max(std::min((size_t) stage_mib << 20, (size_t) n_expert_ * one_expert), one_expert);
        chunk_bytes_ = std::max(kChunkBytes, one_expert);
        // With stage_read_mib the buffer's addresses are reserved for the larger size, and a read maps the rest.
        stage_narrow_ = stage_wide_ = stage_bytes_;
        if (widens && read_mib > 0)
            stage_wide_ = std::max(stage_bytes_, std::min((size_t) read_mib << 20, (size_t) n_expert_ * one_expert));
#if defined(STRATA_GPU_METAL)
        if (!opt_.metal_balanced)
#endif
            ensure_stage();
        for (void** e : {&ev_stage_[0], &ev_stage_[1], &ev_ahead_, &ev_free_}) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t up = nullptr;
        ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
        up_ = up;
        ids_.resize(rr);
        first_.resize((size_t) n_expert_);

        for (void** e : {&ev_in_[0], &ev_in_[1], &ev_out_[0], &ev_out_[1]}) {   // read()
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t rs = nullptr;
        ck(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking), "cudaStreamCreate");
        rs_ = rs;
    }
    use_.assign(layers_.size() * n_expert_, 0);
}

void Engine::save_profile() const {
    if (opt_.profile.empty()) return;
    const std::string tmp = opt_.profile + ".tmp";
    // every card's engine counts its own layers' experts, and the last one's profile has the draft block's too
    const Engine* last = this;
    while (last->next_) last = last->next_.get();
    std::vector<int64_t> all = last->prior_;
    for (const Engine* e = this; e; e = e->next_.get()) {
        if (all.size() < e->use_.size()) all.resize(e->use_.size(), 0);
        for (size_t i = 0; i < e->use_.size(); ++i) all[i] += e->use_[i];
    }
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    for (size_t i = 0; i < all.size(); ++i) {
        const int64_t n = all[i];
        if (n > 0)
            std::fprintf(f, "%d %d %lld\n", (int) (i / (size_t) n_expert_), (int) (i % (size_t) n_expert_),
                         (long long) n);
    }
    const bool written = std::fclose(f) == 0;
    std::error_code ec;
    if (written) std::filesystem::rename(tmp, opt_.profile, ec);   // std::rename replaces no file on Windows
    if (!written || ec) throw std::runtime_error("cannot write " + opt_.profile);
}

void Engine::reset() {
    OnCard here(device_, split_);
    ck(cudaMemsetAsync(state_, 0, state_n_ * sizeof(float), cs(st_)), "reset");
    pos_ = 0;
    ahead_pos_ = -1;
    if (next_) next_->reset();
}

// STRATA_FREEZE_EXPERTS: the freeze is for the engine's whole run, not one request — a greedy answer
// has to be a function of the request alone, and with a cache that changes between requests it is
// not.  The followers' uploads pile up in their slots (kSent) and are claimed by the one call below,
// which the operator's VRAM resize makes between requests; after it, unfreeze stays off and the
// cache is as the resize left it for every request after.
void Engine::unfreeze_experts() {
    freeze_experts_ = expert_cap_ >= 0;   // STRATA_GPU_EXPERTS: frozen for the whole run
    if (next_) next_->unfreeze_experts();
}

void Engine::refreeze_experts() {
    if (!freeze_experts_asked_) return;
    freeze_experts_ = true;
    if (next_) next_->refreeze_experts();
}

// The layers copy their state to pinned memory, which the card writes at its own speed, and the CPU's threads copy
// that to dst.  Straight into dst, which is pageable, Qwen3.6's 66 MB took 44 ms with the card idle, three times in
// a 4,833-token prompt (a conversation's checkpoints).
void Engine::save_state(uint8_t* dst) {
    OnCard here(device_, split_);
    const size_t bytes = card_state_bytes();
    if (h_state_bytes_ < bytes) {
        hfree(h_state_);
        h_state_ = nullptr;
        h_state_ = (uint8_t*) hmalloc(bytes, "pinned memory");
        h_state_bytes_ = bytes;
    }
    for (int il = 0; il < (int) layers_.size(); ++il)
        if (runs(il)) save_layer(il, pos_, h_state_);
    save_last(h_state_);
    ck(cudaStreamSynchronize(cs(st_)), "save state");
    state_out(h_state_, dst);
    if (next_) next_->save_state(dst + bytes);
}

// A card's state from pinned memory to its place, a piece a CPU thread.
void Engine::state_out(const uint8_t* from, uint8_t* to) {
    constexpr size_t kPiece = size_t(1) << 20;
    const size_t bytes = card_state_bytes();
    const int pieces = (int) ((bytes + kPiece - 1) / kPiece);
    const auto piece = [&](int i) {
        const size_t at = (size_t) i * kPiece;
        std::memcpy(to + at, from + at, std::min(kPiece, bytes - at));
    };
    if (workers_) workers_->run(pieces, piece);
    else
        for (int i = 0; i < pieces; ++i) piece(i);
}

// What the state has that is no layer's: the model's output for the last token, which the draft block pairs the
// next one with.  It lies behind the layers' state.
void Engine::save_last(uint8_t* dst) {
    if (!draft_) return;
    ck(cudaMemcpyAsync(dst + rec_n_ * sizeof(float), hn_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost,
                       cs(st_)),
       "save state");
}

void Engine::load_state(const uint8_t* src, int64_t pos) {
    if (pos < 0 || pos > opt_.max_context) throw std::logic_error("load_state: the position is outside the context");
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "load state");
    ck(cudaMemcpy(state_, src, state_n_ * sizeof(float), cudaMemcpyHostToDevice), "load state");
    load_extra(src + state_n_ * sizeof(float), pos);
    pos_ = pos;
    ahead_pos_ = -1;
    if (next_) next_->load_state(src + card_state_bytes(), pos);
}

void Engine::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    OnCard here(device_, split_);
    save_card_rows(first, n, dst);
    if (next_) next_->save_rows(first, n, dst + (size_t) n * card_row_bytes());
}

void Engine::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    OnCard here(device_, split_);
    load_card_rows(src, from, count, first, n);
    if (next_) next_->load_rows(src + (size_t) count * card_row_bytes(), from, count, first, n);
}

// The CPU reads an expert the card has only after an answer moves it out of the cache (follow() gives away the
// slot of the least used one), and a page brought back while an answer is written costs that answer: with every
// cached expert's pages dropped at load, LFM2.5's first short chat moved 624 of its 838 cached experts out and
// wrote at 70 tokens/s in place of 80.  So of the card's experts only the half used most lately go, the last to
// move out (the same chat at 90 tokens/s, as with nothing dropped).  The CPU's experts past the RAM budget go too,
// and the pinned rows of a long read: 256 MiB after Qwen3.6's 32K prompt, held for the next one that long.
// It runs once after the load as well.  Before an answer only the profile's counts order the card's experts, and
// that half still held LFM2.5's first short chat to 61 tokens/s in place of 67, so it goes then only under the RAM
// budget: Qwen3.6 held 5.3 GiB after the load with --ram-experts-mib 3072 in place of 7.1.  Under the budget all of
// the card's experts go: one the cache moves out comes to the CPU through pool_, read from the files.
// Options::lock_experts, on mapper_ once map_pages() is done: the experts outside VRAM at the load, locked one by
// one so that relock() knows which are.  Windows locks no more pages than the working-set minimum leaves room
// for, so it is raised once, by those experts and 512 MiB for the rest of the process.  It says what it got.
void Engine::lock_outside(const std::vector<std::pair<int, int>>& outside) {
    size_t want = 0, got = 0;
    for (const auto& [il, e] : outside)
        expert_bytes(layers_[(size_t) il], e, e + 1, [&](const uint8_t*, size_t n) { want += n; });
#ifdef _WIN32
    const HANDLE self = GetCurrentProcess();
    SIZE_T lo = 0, hi = 0;
    DWORD flags = 0;
    if (GetProcessWorkingSetSizeEx(self, &lo, &hi, &flags)) {
        const SIZE_T min = lo + want + ((SIZE_T) 512 << 20);
        SetProcessWorkingSetSizeEx(self, min, std::max(hi, min + ((SIZE_T) 512 << 20)),
                                   QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    }
#endif
    bool refused = false;
    for (const auto& [il, e] : outside) {
        if (map_quit_.load(std::memory_order_relaxed)) return;
        bool ok = true;
        size_t n = 0;
        expert_bytes(layers_[(size_t) il], e, e + 1, [&](const uint8_t* p, size_t b) {
            ok = ok && lock_pages(p, b, true);
            n += b;
        });
        if (!ok) {
            refused = true;
            break;
        }
        locked_[(size_t) il * n_expert_ + e] = 1;
        got += n;
    }
    const double gib = 1.0 / (1ull << 30);
    if (!refused) {
        std::fprintf(stderr, "strata-poly: %.1f GiB of experts outside VRAM locked in RAM (--lock-experts)\n",
                     want * gib);
    } else {
#ifdef _WIN32
        const unsigned long err = GetLastError();
        const std::string why = "VirtualLock refused the rest (error " + std::to_string(err) + ")";
#else
        const std::string kib = memlock_kib();
        const std::string why = kib.empty() ? std::string("mlock refused the rest")
                                            : "mlock refused the rest: `ulimit -l` is " + kib +
                                                  " (KiB); raise it before strata-poly starts, with `ulimit -l "
                                                  "unlimited` where `ulimit -Hl` allows it, else a memlock line in "
                                                  "/etc/security/limits.conf";
#endif
        std::fprintf(stderr, "strata-poly: --lock-experts locked %.1f of the %.1f GiB of experts outside VRAM; %s, "
                             "so the rest stays in the file cache\n", got * gib, want * gib, why.c_str());
    }
    locks_ready_.store(true, std::memory_order_release);
}

// Options::lock_experts, between requests: the experts the cache took since are unlocked, then the ones it gave
// up are locked, until the system refuses one.  The split changes while an answer is written (follow()) and when
// the cache is cut or sized again; this keeps the locked ones the CPU's without a lock at each of those.  Nothing
// before lock_outside() is done.
void Engine::relock() {
    if (!opt_.lock_experts || !locks_ready_.load(std::memory_order_acquire)) return;
    for (int pass = 0; pass < 2; ++pass)
        for (size_t il = 0; il < layers_.size(); ++il) {
            const Layer& L = layers_[il];
            if (!L.moe) continue;
            for (int e = 0; e < n_expert_; ++e) {
                uint8_t& k = locked_[il * (size_t) n_expert_ + e];
                const bool outside = !L.slot[(size_t) e];
                if (pass == 0 && k && !outside) {
                    expert_bytes(L, e, e + 1, [](const uint8_t* p, size_t n) { lock_pages(p, n, false); });
                    k = 0;
                } else if (pass == 1 && !k && outside) {
                    bool ok = true;
                    expert_bytes(L, e, e + 1, [&](const uint8_t* p, size_t n) { ok = ok && lock_pages(p, n, true); });
                    if (!ok) return;
                    k = 1;
                }
            }
        }
}

size_t Engine::cached_file_bytes() const {
    size_t n = 0;
    for (const Layer& L : layers_)
        if (L.moe)
            expert_runs(L, n_expert_, [&](int e) { return L.slot[(size_t) e] != nullptr; },
                        [&](const uint8_t*, size_t b) { n += b; });
    return n + (next_ ? next_->cached_file_bytes() : 0);
}

// The bytes of the experts outside VRAM that take pct% of their uses in the profile (prior_), the most used first;
// all of them where there is no profile.
size_t Engine::covering(int pct) const {
    std::vector<std::pair<int64_t, size_t>> outs;   // uses, bytes
    int64_t all = 0;
    for (size_t il = 0; il < layers_.size(); ++il) {
        const Layer& L = layers_[il];
        if (!L.moe) continue;
        for (int e = 0; e < n_expert_; ++e) {
            if (L.slot[(size_t) e]) continue;
            const int64_t u = prior_.empty() ? 0 : prior_[il * (size_t) n_expert_ + (size_t) e];
            size_t b = 0;
            expert_bytes(L, e, e + 1, [&](const uint8_t*, size_t n) { b += n; });
            outs.push_back({u, b});
            all += u;
        }
    }
    if (all <= 0) return outside_bytes_;
    std::sort(outs.begin(), outs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    size_t bytes = 0;
    int64_t got = 0;
    for (const auto& [u, b] : outs) {
        if (got * 100 >= all * pct) break;
        got += u;
        bytes += b;
    }
    return bytes;
}

void Engine::trim() {
    relock();   // first: under Linux the pages dropped below must not be locked
    struct Held {
        float recent;
        int64_t used;
        int il, e;
    };
    std::vector<Held> held;
    const bool answered = std::any_of(use_.begin(), use_.end(), [](int64_t u) { return u > 0; });
    for (int il = 0; il < (int) layers_.size() && (answered || opt_.ram_experts_mib >= 0); ++il)
        if (layers_[(size_t) il].moe)
            for (int e = 0; e < n_expert_; ++e)
                if (layers_[(size_t) il].slot[(size_t) e]) {
                    const size_t i = (size_t) il * n_expert_ + e;
                    held.push_back({recent_.empty() ? 0.f : recent_[i], (prior_.empty() ? 0 : prior_[i]) + use_[i],
                                    il, e});
                }
    const size_t n = pool_ ? held.size() : held.size() / 2;
    std::partial_sort(held.begin(), held.begin() + (ptrdiff_t) n, held.end(), [](const Held& a, const Held& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.used > b.used;
    });
    for (size_t i = 0; i < n; ++i) expert_bytes(layers_[(size_t) held[i].il], held[i].e, held[i].e + 1, drop_pages);
    if (pool_)   // and the CPU's outside kept_, which come from pool_; kept_'s stay for the next answer
        for (size_t il = 0; il < layers_.size(); ++il) {
            const Layer& L = layers_[il];
            const uint8_t* k = kept_.data() + il * (size_t) n_expert_;
            if (L.moe) expert_runs(L, n_expert_, [&](int e) { return !L.slot[(size_t) e] && !k[e]; }, drop_pages);
        }
    hfree(h_res_);
    h_res_ = nullptr;
    h_res_rows_ = 0;
#if !defined(STRATA_GPU_METAL)
    if (h_bell_ && n_swap_ == 0 && d_use_) {   // the doorbell's picks into the profile, which follow() does with swaps
        const size_t count = layers_.size() * (size_t) n_expert_;
        h_use_.resize(count);
        ck(cudaMemcpy(h_use_.data(), d_use_, count * sizeof(int32_t), cudaMemcpyDeviceToHost), "expert uses");
        ck(cudaMemset(d_use_, 0, count * sizeof(int32_t)), "expert uses");
        for (size_t i = 0; i < count; ++i) use_[i] += h_use_[i];
    }
#endif
    if (pool_ && !answered) pool_fill();
    if (pool_) {
        std::fill(prompt_use_.begin(), prompt_use_.end(), 0.f);
        prompt_rows_ = 0;
    }
    if (lock_thread_.joinable()) {   // between requests: lock_loop()'s turn
        {
            std::lock_guard<std::mutex> l(lock_m_);
            lock_idle_.store(true, std::memory_order_relaxed);
        }
        lock_cv_.notify_one();
    }
    if (next_) next_->trim();
    else drop_images();   // once, by the last card's engine
}

// n bytes of the mapped files from src into dst, read through the file cache instead of the mapping, so that the
// pages stay out of this process's working set; false when src is in none of the files or the read fails.  A
// thread keeps a handle to each file.  For the experts the RAM budget leaves out (Options::ram_experts_mib), which
// a prompt batch copies to the card: through the mapping they came back into the working set a fault at a time.
bool Engine::read_file(uint8_t* dst, const uint8_t* src, size_t n, bool copy) const {
    const FileView* v = nullptr;
    for (const FileView& f : views_)
        if (src >= f.base && src + n <= f.base + f.bytes) v = &f;
    if (!v) return false;
    const uint64_t off = (uint64_t) (src - v->base);
#ifdef _WIN32
    struct Handles {
        std::map<std::string, HANDLE> h;
        ~Handles() {
            for (auto& [path, fh] : h) CloseHandle(fh);
        }
    };
    thread_local Handles mine;
    HANDLE& fh = mine.h[v->path];
    if (!fh) fh = CreateFileA(v->path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return false;
    for (size_t done = 0; done < n;) {
        OVERLAPPED o{};
        o.Offset = (DWORD) (off + done);
        o.OffsetHigh = (DWORD) ((off + done) >> 32);
        DWORD got = 0;
        const DWORD want = (DWORD) std::min<size_t>(n - done, size_t(1) << 30);
        if (!ReadFile(fh, dst + done, want, &got, &o) || got == 0) return false;
        done += got;
    }
#else
    struct Fds {
        std::map<std::string, int> fd;
        ~Fds() {
            for (auto& [path, f] : fd)
                if (f >= 0) ::close(f);
        }
    };
    thread_local Fds mine;
    // Under a RAM budget (pool_) inside a cgroup's limit, the experts it leaves out are read past the file cache: the
    // cgroup is charged for those pages too, so they took the budget's RAM a second time, and the system made room for
    // them by dropping the pages of the ones it keeps.  With no such limit the cache is the machine's free RAM, not
    // this process's: the pages it has room for stay there, and a miss found there is no disk read.
    //
    // `copy`: the pinned copy of the experts outside VRAM (build_mirror(), lock_loop()), read past the file cache under
    // a cgroup's limit too.  Through it, the copy's pages were charged twice while it was made: on an RTX 5090 pod held
    // to 90 GiB, DeepSeek V4 Flash's 56.8 GiB copy took the container to its limit and the pod ended the engine with
    // SIGKILL at each of four starts (no OOM kill counted), where with no copy it started.
    if ((pool_ || copy) && pool_uncached_) {
        thread_local Fds raw;
        auto it = raw.fd.find(v->path);
        if (it == raw.fd.end()) it = raw.fd.emplace(v->path, open_uncached(v->path.c_str())).first;
        if (it->second >= 0 && read_uncached(it->second, dst, off, n)) return true;
    }
    auto it = mine.fd.find(v->path);
    if (it == mine.fd.end()) it = mine.fd.emplace(v->path, ::open(v->path.c_str(), O_RDONLY | O_CLOEXEC)).first;
    if (it->second < 0) return false;
    for (size_t done = 0; done < n;) {
        const ssize_t got = ::pread(it->second, dst + done, n - done, (off_t) (off + done));
        if (got <= 0) return false;
        done += (size_t) got;
    }
#endif
    return true;
}

const Stats& Engine::stats() const {
    if (!next_) return stats_;
    all_stats_ = stats_;   // the tokens are counted once
    for (const Engine* e = next_.get(); e; e = e->next_.get()) {
        const Stats& s = e->stats_;
        all_stats_.gpu_experts += s.gpu_experts;
        all_stats_.cpu_experts += s.cpu_experts;
        all_stats_.batch_cached += s.batch_cached;
        all_stats_.batch_staged += s.batch_staged;
        all_stats_.swapped_experts += s.swapped_experts;
        all_stats_.tail_experts += s.tail_experts;
        all_stats_.ahead_sent += s.ahead_sent;
        all_stats_.ahead_used += s.ahead_used;
        all_stats_.cpu_ms += s.cpu_ms;
        all_stats_.route_wait_ms += s.route_wait_ms;
    }
    return all_stats_;
}

void Engine::embed(int32_t token, float* dst) const {
    const uint8_t* row = tok_embd_ + (size_t) token * tok_row_;
    if (tok_type_ == kTypeF32) std::memcpy(dst, row, (size_t) n_embd_ * sizeof(float));
    else type_to_floats(tok_type_, row, dst, n_embd_);
}

void Engine::images(const Image* im, int n) {
    for (int i = 0; i < n; ++i) {
        const Image& a = im[i];
        if (a.n < 1 || a.nx < 1 || a.n % a.nx != 0 || !a.rows || a.at < (i ? im[i - 1].at + im[i - 1].n : 0))
            throw std::logic_error("images: a picture's rows are not a grid, or the pictures are out of order");
    }
    images_.assign(im, im + n);
    ++images_set_;
    if (next_) next_->images(im, n);   // positions and masks depend on where the pictures are
}

const Image* Engine::image_at(int64_t row) const {
    for (const Image& im : images_)
        if (row >= im.at && row < im.at + im.n) return &im;
    return nullptr;
}

// What the model reads at a row of the sequence: a picture's row there, else the token's embedding.  True for a
// picture's row, which goes in as the encoder made it (llama.cpp scales token embeddings only).
bool Engine::input(int32_t token, int64_t row, float* dst) const {
    if (const Image* im = image_at(row)) {
        std::memcpy(dst, im->rows + (size_t) (row - im->at) * n_embd_, (size_t) n_embd_ * sizeof(float));
        return true;
    }
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    embed(token, dst);
    return false;
}

// What the model reads at rows [pos, pos + n): input()'s, scaled as the model scales token embeddings, and copied to
// every residual stream.  A row is a dequantized row of the embeddings: on one thread a 2,048-token call's rows
// took 5-9 ms of Qwen3.6's, which the card waited out, so the CPU threads take a row each (0.4-0.7 ms).
void Engine::embed_rows(const int32_t* tokens, int64_t pos, int n, float* rows) const {
    for (int t = 0; t < n; ++t)
        if ((tokens[t] < 0 || tokens[t] >= n_vocab_) && !image_at(pos + t))
            throw std::runtime_error("a token id is outside the vocabulary");
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    const float scale = embed_scale();
    const auto one = [&](int t) {
        float* r = rows + (size_t) t * r_row;
        if (!input(tokens[t], pos + t, r) && scale != 1.f)
            for (size_t i = 0; i < e_row; ++i) r[i] *= scale;
        for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
    };
    if (n >= 64 && workers_) workers_->run(n, one);
    else
        for (int t = 0; t < n; ++t) one(t);
}

void Engine::dbg(const char* name, int il, const float* dev, int n) {
    if (!opt_.dump) return;
    std::vector<float> h((size_t) n);
    ck(cudaStreamSynchronize(cs(st_)), "dump");
    ck(cudaMemcpy(h.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    double sum = 0, abs = 0;
    for (float v : h) {
        sum += v;
        abs += std::fabs(v);
    }
    std::fprintf(stderr, "dump pos=%lld blk=%d %-8s n=%d sum=%.6f abs=%.6f first=%.6f %.6f %.6f\n", (long long) pos_,
                 il, name, n, sum, abs, h[0], h[1], h[2]);
}

// y = w x for n tokens.  xq is x already in Q8_1, when the caller has it.  rows_per_x below n_out gives each group
// of rows its own input (see gemv_f32); only the float types take it.
void Engine::lin(const Dev& w, const float* x, float* y, int n, const void* xq, int rows_per_x) {
    lin_raw(w, x, y, n, xq, rows_per_x);
    if (w.scale != 1.0f) axpy(y, w.scale, y, n * w.n_out, false, st_);
}

// Two Q8_0 matrices on one input, a few tokens' worth (Unsloth's files keep attention, the recurrent layers'
// projections and the shared expert in Q8_0): one launch, each output the bits of its own launch, the second
// matrix's rows after the first's (a recurrent layer's 8,192 rows of queries, keys and values and 4,096 of its
// output gate).  A small matrix alone leaves most of the card idle for its launch.
void Engine::lin_pair(const Dev& w1, const Dev& w2, const float* x, float* y1, float* y2, int n, const void* xq) {
#if !defined(STRATA_GPU_METAL)
    // F32 or BF16, the few tokens' products of gemv_f32() or gemv_bf16(): a recurrent layer's alpha and beta, a
    // router and the next layer's (moe_router())
    if (w1.type == w2.type && (w1.type == kTypeF32 || w1.type == kTypeBF16) && w1.n_in == w2.n_in && n < kStageRows &&
        w1.scale == 1.0f && w2.scale == 1.0f) {
        gemv_pair(w1.type == kTypeBF16 ? 1 : 0, w1.p, w2.p, x, y1, y2, w1.n_in, w1.n_out, w2.n_out, n, st_);
        return;
    }
    const bool pair = xq && n >= 2 && n <= 8 && w1.type == w2.type && w1.n_in == w2.n_in && w1.scale == 1.0f &&
                      w2.scale == 1.0f && !(mmq_ && n >= kStageRows);
    if (pair && kernels::native_mmvq_pair((int) w1.type, w1.p, w2.p, xq, y1, y2, w1.n_in, w1.n_out, n, st_, w2.n_out))
        return;
#endif
    lin(w1, x, y1, n, xq);
    lin(w2, x, y2, n, xq);
}

ExpertScales Engine::expert_scales(const Layer& L, const int32_t* ids) const {
    ExpertScales s;
    if (L.h_s_gate.empty() && L.h_s_up.empty()) return s;
    s.gate = L.h_s_gate.empty() ? nullptr : fp(L.s_gate);
    s.up = L.h_s_up.empty() ? nullptr : fp(L.s_up);
    s.ids = ids;
    return s;
}

void Engine::lin_raw(const Dev& w, const float* x, float* y, int n, const void* xq, int rows_per_x) {
    if (w.type == kTypeF32 || w.type == kTypeBF16) {
        const int per = rows_per_x ? rows_per_x : w.n_out;
        const size_t count = (size_t) w.n_in * w.n_out;
        // A prompt batch: one matrix product.  The kernel below has a block for every output of every token,
        // and GLM 5.3 Flash's per-head projections are 32,768 outputs a token.
        if (n >= kStageRows && (w.type == kTypeF32 || (wf_ && count <= bf16_max_))) {
            const float* wf = fp(w);
            if (w.type == kTypeBF16) {
                bf16_to_f32((const uint16_t*) w.p, wf_, count, st_);
                wf = wf_;
            }
            return gemm_f32(blas_, wf, x, y, w.n_in, w.n_out, per, n);
        }
        if (w.type == kTypeF32) return gemv_f32(fp(w), x, y, w.n_in, w.n_out, per, n, st_);
        return gemv_bf16((const uint16_t*) w.p, x, y, w.n_in, w.n_out, per, n, st_);
    }
    // a prompt batch: llama.cpp's matrix-matrix kernels read the weights once for all the tokens
    if (mmq_ && n >= kStageRows && Mmq::fits((int) w.type, w.n_out, w.n_in)) {
        if (prompt_lent_ || work_lent_)
            throw std::logic_error("a prompt's product with its buffers lent to the expert cache");
        // A row too large for the halves its blocks keep their sums in goes in divided by a power of two, and x is
        // left as it was for whatever reads it next (both exact): its product comes out divided by the same.
        const bool guard = half_sum_guard_ && Mmq::half_sums((int) w.type);
        float* xs = const_cast<float*>(x);
        if (guard) guard_half_sums(xs, w.n_in, w.n_in, n, hf_, st_);
        Mmq::quantize(x, nullptr, mqx_, (int) w.type, w.n_in, w.n_in, n, st_);
        if (guard) scale_rows(xs, w.n_in, w.n_in, n, nullptr, hf_, false, st_);
        mmq_->mul(w.p, (int) w.type, w.n_out, w.n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, w.n_out, st_);
        if (guard) scale_rows(y, w.n_out, w.n_out, n, nullptr, hf_, false, st_);
        return;
    }
    if (!xq) {
        // the tokens' blocks are back to back, so they quantize as one long vector
        kernels::native_quantize_q8_1(x, q8_, w.n_in * n, 1, st_);
        xq = q8_;
    }
    // a launch takes up to 8 tokens, and each comes out as it would alone
    const size_t token = kernels::native_q8_1_bytes(w.n_in);
    for (int at = 0; at < n; at += 8)
        kernels::native_mmvq((int) w.type, w.p, (const uint8_t*) xq + (size_t) at * token,
                             y + (size_t) at * w.n_out, w.n_in, w.n_out, std::min(8, n - at), st_);
}

void Engine::router_scores(const Layer& L, int n) { lin(L.gate_inp, xn_, rl_, n, xq_); }

bool Engine::batched(const Layer& L, int n) const { return mmq_ && L.mmq && n >= kStageRows; }

// The routed experts' rows a call routed on the card can have: window()'s with guesses, and step()'s, every
// sequence's token and its guess.  Sized for kSumRows only, a step of more than four sequences was routed on the
// host and waited for the card at every layer.
int Engine::card_rows() const {
    const int step_rows = (opt_.batch_mtp ? 2 : 1) * (opt_.batch + 1);
    return std::max({kSumRows, (block_guess_ ? opt_.window : 1) * n_used_, step_rows * n_used_});
}

// The router's scores, and the copies of them and of the experts' input to the host.
void Engine::moe_router(const Layer& L, int n) {
    const size_t ne = (size_t) n_expert_;
    int j = (int) (&L - layers_.data()) + 1;   // the next MoE layer, whose router on this layer's input guesses ahead
    while (j < n_layer_ && !layers_[(size_t) j].moe) ++j;
    size_t part = 0;
    for (size_t q = 0; q < 8; ++q)
        if (ev_ == ev_row_[q]) part = q;
    const bool ahead = prefetch_k_ > 0 && !block_pass_ && n <= 16 && j < n_layer_;
#if !defined(STRATA_GPU_METAL)
    // vram_ahead(): both routers in one launch where the router reads xn_, and the guesses' scores go to the host in
    // the router's copy, a kernel's as the router's own: a copy engine's copy on st_ waits for the launches around it
    if (ahead && ahead_slots_ > 0) {
        float* d = ahead_dev_ + part * 16 * ne;
        if (plain_router()) {
            lin_pair(L.gate_inp, layers_[(size_t) j].gate_inp, xn_, rl_, d, n, xq_);
        } else {
            router_scores(L, n);
            lin(layers_[(size_t) j].gate_inp, xn_, d, n, xq_);
        }
        copy3(rl_, h_rl_, n * n_expert_, xn_, h_x_, n * n_embd_, d, h_ahead_ + part * 16 * ne, n * n_expert_, st_);
        return;
    }
#endif
    router_scores(L, n);
    if (!ahead && !router_rows_to_host(n)) return;
    copy2(rl_, h_rl_, n * n_expert_, xn_, h_x_, n * n_embd_, st_);
    if (!ahead) return;
    float* d = ahead_dev_ + part * 16 * ne;   // pool_ahead()
    lin(layers_[(size_t) j].gate_inp, xn_, d, n, xq_);
    ck(cudaMemcpyAsync(h_ahead_ + part * 16 * ne, d, (size_t) n * ne * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "read ahead");
}

// Layer j's likeliest prefetch_k_ experts of each of n tokens, as moe_router() guessed them before ev_: those the CPU
// would read from the files get a slot now, and pool_threads_ read them while this layer is computed, every token's
// best guess first, then every token's second, and so on.  A slot read ahead is held through the calls up to its
// layer's (a window's halves are a call each): held for the next call alone, it could go to the other half's misses
// before its own half came to it.  It is on trial: one its layer does not use goes when the clock comes to it.
void Engine::pool_ahead(int j, int n, size_t part, int k) {
    const Layer& L = layers_[(size_t) j];
    const size_t ne = (size_t) n_expert_;
    k = std::min(k, 16);
    n = std::min(n, 16);
    int32_t top[16][16];
    int m[16];
    for (int r = 0; r < n; ++r) {
        // the k best in one pass: a sort of all of them was a fifth of a millisecond a token on the laptop
        const float* g = h_ahead_ + part * 16 * ne + (size_t) r * ne;
        int32_t* t = top[r];
        float best[16];
        m[r] = 0;
        for (int e = 0; e < (int) ne; ++e) {
            const float s = g[e];
            if (m[r] == k && s <= best[k - 1]) continue;
            int at = m[r] < k ? m[r]++ : k - 1;
            for (; at > 0 && best[at - 1] < s; --at) best[at] = best[at - 1], t[at] = t[at - 1];
            best[at] = s;
            t[at] = e;
        }
    }
    for (int q = 0; q < k; ++q)
        for (int r = 0; r < n; ++r) {
            if (q >= m[r]) continue;
            const size_t e = (size_t) top[r][q], x = (size_t) j * ne + e;
            if (L.slot[e] || kept_[x] || pool_of_[x] >= 0) continue;
            const size_t v = pool_victim(j);
            if (v == SIZE_MAX) return;   // no slot to spare; what was queued is still read
            if (pool_who_[v] >= 0) pool_of_[(size_t) pool_who_[v]] = -1;
            pool_who_[v] = (int64_t) x;
            pool_of_[x] = (int32_t) v;
            pool_tick_[v] = pool_now_ + 4;
            pool_queue(L, e, (int32_t) v, false);
        }
}

// Layer j's likeliest experts that the card lacks, kAheadGuesses of each of n tokens, every token's best guess
// first, up from the mirror into the next group of the staging buffer's slots while the CPU computes this layer's
// experts (moe()'s launch).  The group's last reader is done before they land (its `read`).  A guess another group
// already has for j is not sent twice.
//
// The guesses are the next layer's router on this layer's input (moe_router()).  Qwen3.6 UD-Q3_K_M on lab-pro6000
// GPU2 held to 6 GB (--vram-pool): of the experts a call uploaded, 1.05 to 1.1 uses each with 4 guesses a token;
// with 6 the card took 84% of the experts in place of 81%, but more uploads were late and the answer slower; with
// 8 a third of them went unused.
void Engine::vram_ahead(int j, int n, size_t part) {
    // up_ copies in order: guesses put behind uploads it has not finished land after their layer has run, and a
    // call's own miss (moe()) waits behind them with the card.  A bus that keeps up is done before the next layer.
    // Qwen3.6 UD-Q4_K_M on a 4090 D held to 11 GiB, PCIe 4.0 x8: the first answer after a load wrote 102-109
    // tokens/s with them queued, 173-185 without; UD-Q3_K_M on the PRO 6000, PCIe 5.0 x16: 300-358 and 334-347.
    if (up_busy()) return;
    const Layer& L = layers_[(size_t) j];
    const size_t ne = (size_t) n_expert_;
    constexpr int k = kAheadGuesses;
    n = std::min(n, 16);
    const int me = ahead_turn_;
    Ahead& g = ahead_[me];
    g.il = -1;
    g.n = 0;
    int32_t top[16][k];
    int m[16];
    for (int r = 0; r < n; ++r) {   // the k best in one pass, as pool_ahead()
        const float* s = h_ahead_ + part * 16 * ne + (size_t) r * ne;
        int32_t* t = top[r];
        float best[k];
        m[r] = 0;
        for (int e = 0; e < (int) ne; ++e) {
            if (m[r] == k && s[e] <= best[k - 1]) continue;
            int at = m[r] < k ? m[r]++ : k - 1;
            for (; at > 0 && best[at - 1] < s[e]; --at) best[at] = best[at - 1], t[at] = t[at - 1];
            best[at] = s[e];
            t[at] = e;
        }
    }
    const auto sent = [&](int32_t e) {
        for (int h = 0; h < kAheadGroups; ++h) {
            const Ahead& o = ahead_[h];
            if (o.il == j && std::find(o.e, o.e + o.n, e) != o.e + o.n) return true;
        }
        return false;
    };
    for (int q = 0; q < k && g.n < ahead_slots_; ++q)
        for (int r = 0; r < n && g.n < ahead_slots_; ++r) {
            if (q >= m[r]) continue;
            const int32_t e = top[r][q];
            if (L.slot[(size_t) e] || !mirror(j, e) || std::find(g.e, g.e + g.n, e) != g.e + g.n || sent(e)) continue;
            g.e[g.n++] = e;
        }
    if (g.n == 0) return;
    ahead_turn_ = (ahead_turn_ + 1) % kAheadGroups;
    staged_il_ = -1;   // the staging buffer holds other experts now
    ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) g.read, 0), "upload ahead");
    uint8_t* at = stage_ + (size_t) me * ahead_slots_ * ahead_stride_;
    for (int i = 0; i < g.n; ++i, at += ahead_stride_) {
        ck(cudaMemcpyAsync(at, mirror(j, g.e[i]), L.gu2() + L.d_bytes, cudaMemcpyHostToDevice, cs(up_)), "upload ahead");
        ck(cudaEventRecord((cudaEvent_t) g.arrived[i], cs(up_)), "upload ahead");   // usable on its own
    }
    ck(cudaEventRecord((cudaEvent_t) g.up, cs(up_)), "upload ahead");
    up_last_ = g.up;
    g.here = 0;
    g.il = j;
    stats_.ahead_sent += g.n;
}

bool Engine::up_busy() const { return up_last_ && cudaEventQuery((cudaEvent_t) up_last_) == cudaErrorNotReady; }

// Expert e of layer il in a group vram_ahead() uploaded, once it has arrived, or null.  `read` gets the group's bit:
// moe() marks the group's last reader after its launches.
const uint8_t* Engine::ahead_of(int il, int e, unsigned& read) {
    for (int h = 0; h < kAheadGroups; ++h) {
        Ahead& g = ahead_[h];
        if (g.il != il) continue;
        const int32_t* p = std::find(g.e, g.e + g.n, e);
        if (p == g.e + g.n) continue;
        const int i = (int) (p - g.e);
        if (!(g.here >> i & 1u)) {
            const cudaError_t q = cudaEventQuery((cudaEvent_t) g.arrived[i]);
            if (q == cudaErrorNotReady) return nullptr;
            ck(q, "upload ahead");
            g.here |= 1u << i;
        }
        read |= 1u << h;
        return stage_ + ((size_t) h * ahead_slots_ + (size_t) (p - g.e)) * ahead_stride_;
    }
    return nullptr;
}

int Engine::first_moe_layer() const {
    int il = 0;
    while (il < n_layer_ && !layers_[(size_t) il].moe) ++il;
    return il;
}

// The CPU's experts this process keeps in its working set, by (layer, expert): the most used lately, then by the
// profile's and this run's counts, as many as Options::ram_experts_mib holds past pool_; every one without a budget.
// The card's are none of them.
std::vector<uint8_t> Engine::ram_keep(const std::vector<float>* lately) const {
    const size_t ne = (size_t) n_expert_;
    std::vector<uint8_t> keep(layers_.size() * ne, 1);
    if (opt_.ram_experts_mib < 0) return keep;
    std::fill(keep.begin(), keep.end(), 0);
    struct Cand {
        float recent;
        int64_t used;
        size_t i;
    };
    std::vector<Cand> cands;
    for (size_t il = 0; il < layers_.size(); ++il) {
        if (!layers_[il].moe) continue;
        for (size_t e = 0; e < ne; ++e) {
            const size_t i = il * ne + e;
            if (layers_[il].slot[e]) continue;
            cands.push_back({lately ? (*lately)[i] : recent_.empty() ? 0.f : recent_[i],
                             (prior_.empty() ? 0 : prior_[i]) + use_[i], i});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.used > b.used;
    });
    size_t room = (size_t) opt_.ram_experts_mib << 20;
    room -= std::min(room, pool_bytes_);
    for (const Cand& c : cands) {
        size_t bytes = 0;
        expert_bytes(layers_[c.i / ne], 0, 1, [&](const uint8_t*, size_t n) { bytes += n; });
        if (bytes > room) break;
        room -= bytes;
        keep[c.i] = 1;
    }
    return keep;
}

// After a long prompt (with pool_reseed()): kept_ by the prompt's own uses, then the counts.  The pages of the
// experts that leave it go, and those that come in are read into the working set, by the read threads.
void Engine::rekeep() {
    std::vector<uint8_t> was = ram_keep(&prompt_use_);
    {
        std::lock_guard<std::mutex> l(swap_m_);   // the follower looks at it (follow_loop())
        kept_.swap(was);
    }
    // The page work goes to the read threads behind their reads, in pieces of 4 MiB, and the answer starts at once:
    // its first call waited for it.  An expert of the new kept_ that a call meets first is faulted in by it.
    static constexpr size_t kPiece = size_t(4) << 20;
    std::vector<PoolJob> jobs;
    const auto as = [&jobs](int32_t k) {
        return [&jobs, k](const uint8_t* p, size_t n) {
            for (size_t at = 0; at < n; at += kPiece) jobs.push_back({nullptr, p + at, std::min(kPiece, n - at), k});
        };
    };
    for (size_t il = 0; il < layers_.size(); ++il) {
        const Layer& L = layers_[il];
        if (!L.moe) continue;
        const uint8_t* a = was.data() + il * (size_t) n_expert_;
        const uint8_t* b = kept_.data() + il * (size_t) n_expert_;
        expert_runs(L, n_expert_, [&](int e) { return a[e] && !b[e] && !L.slot[(size_t) e]; }, as(kDrop));
        expert_runs(L, n_expert_, [&](int e) { return !a[e] && b[e] && !L.slot[(size_t) e]; }, as(kTouch));
    }
    {
        std::lock_guard<std::mutex> l(pool_m_);
        pool_pages_.insert(pool_pages_.end(), jobs.begin(), jobs.end());
        pool_n_.fetch_add((int) jobs.size(), std::memory_order_release);
    }
    pool_cv_.notify_all();
}

// After the load: the slots take the experts the profile has used most of those outside kept_ and the card, read
// on every CPU thread, so that the first answer finds them.
void Engine::pool_fill() {
    const size_t ne = (size_t) n_expert_;
    std::vector<std::pair<int64_t, size_t>> cands;
    for (size_t il = 0; il < layers_.size(); ++il)
        if (layers_[il].moe)
            for (size_t e = 0; e < ne; ++e)
                if (!layers_[il].slot[e] && !kept_[il * ne + e])
                    cands.push_back({prior_.empty() ? 0 : prior_[il * ne + e], il * ne + e});
    std::sort(cands.rbegin(), cands.rend());
    struct Fill {
        size_t x;
        PoolRead r;
    };
    std::vector<Fill> fills;
    std::vector<size_t> next;   // by size: the next slot to fill
    for (const PoolSize& z : pool_sizes_) next.push_back(z.first);
    for (const auto& cand : cands) {
        const size_t x = cand.second, c = pool_size_of_[x / ne], k = next[c];
        PoolSize& z = pool_sizes_[c];
        if (k == z.first + z.n) continue;
        ++next[c];
        if (pool_who_[k] >= 0) pool_of_[(size_t) pool_who_[k]] = -1;
        pool_who_[k] = (int64_t) x;
        pool_of_[x] = (int32_t) k;
        pool_prot_[k] = k - z.first < z.prot_max;   // the profile's most used start held
        fills.push_back({x, {x % ne, (int32_t) k, false}});
    }
    workers_->run((int) fills.size(), [&](int t) { pool_read(layers_[fills[(size_t) t].x / ne], fills[(size_t) t].r); });
    for (const Fill& f : fills)
        if (!f.r.ok) {
            pool_of_[f.x] = -1;
            pool_who_[(size_t) f.r.s] = -1;
            pool_prot_[(size_t) f.r.s] = 0;
        }
    for (PoolSize& z : pool_sizes_)
        z.nprot = (size_t) std::count(pool_prot_.begin() + (ptrdiff_t) z.first,
                                      pool_prot_.begin() + (ptrdiff_t) (z.first + z.n), 1);
}

// After a prompt of 4,096 tokens or more, at the answer's first call: the held part of each size of slot (prot_max)
// takes the experts the prompt used most, read on every CPU thread; the slots of experts it used less give way.
// The answer to a new document routes to its experts, not to the last answer's: after a 32K prompt the slots
// missed 4.5 experts a token, and waiting for them was the CPU's whole extra time.
void Engine::pool_reseed() {
    prompt_rows_ = 0;
    const size_t ne = (size_t) n_expert_;
    std::vector<std::pair<float, size_t>> cands;
    for (size_t il = 0; il < layers_.size(); ++il)
        if (layers_[il].moe)
            for (size_t e = 0; e < ne; ++e)
                if (!layers_[il].slot[e] && !kept_[il * ne + e] && prompt_use_[il * ne + e] > 0.f)
                    cands.push_back({prompt_use_[il * ne + e], il * ne + e});
    std::fill(prompt_use_.begin(), prompt_use_.end(), 0.f);
    size_t held = 0;
    for (const PoolSize& z : pool_sizes_) held += z.prot_max;
    const size_t want = std::min(cands.size(), held);
    std::partial_sort(cands.begin(), cands.begin() + (ptrdiff_t) want, cands.end(), std::greater<>());
    std::vector<uint8_t> wanted(layers_.size() * ne, 0);
    for (size_t k = 0; k < want; ++k) wanted[cands[k].second] = 1;
    // by size: the slots that may take them, the last one first, and how many more may be held
    std::vector<std::vector<size_t>> free(pool_sizes_.size());
    std::vector<size_t> room;
    for (size_t c = 0; c < pool_sizes_.size(); ++c) {
        const PoolSize& z = pool_sizes_[c];
        room.push_back(z.prot_max);
        for (size_t v = z.first + z.n; v-- > z.first;)
            if (!pool_pin_[v] && pool_state_[v].load(std::memory_order_acquire) != 1 &&
                (pool_who_[v] < 0 || !wanted[(size_t) pool_who_[v]]))
                free[c].push_back(v);
    }
    struct Fill {
        size_t x;
        PoolRead r;
    };
    std::vector<Fill> fills;
    for (size_t k = 0; k < want; ++k) {
        const size_t x = cands[k].second, c = pool_size_of_[x / ne];
        if (room[c] == 0) continue;
        --room[c];
        if (pool_of_[x] >= 0) {
            pool_prot_[(size_t) pool_of_[x]] = 1;
            continue;
        }
        if (free[c].empty()) continue;
        const size_t v = free[c].back();
        free[c].pop_back();
        if (pool_who_[v] >= 0) pool_of_[(size_t) pool_who_[v]] = -1;
        pool_who_[v] = (int64_t) x;
        pool_of_[x] = (int32_t) v;
        pool_prot_[v] = 1;
        pool_tick_[v] = pool_ref_[v] = 0;
        fills.push_back({x, {x % ne, (int32_t) v, false}});
    }
    workers_->run((int) fills.size(), [&](int t) { pool_read(layers_[fills[(size_t) t].x / ne], fills[(size_t) t].r); });
    for (const Fill& f : fills)
        if (!f.r.ok) {
            pool_of_[f.x] = -1;
            pool_who_[(size_t) f.r.s] = -1;
            pool_prot_[(size_t) f.r.s] = 0;
        }
    for (PoolSize& z : pool_sizes_)
        z.nprot = (size_t) std::count(pool_prot_.begin() + (ptrdiff_t) z.first,
                                      pool_prot_.begin() + (ptrdiff_t) (z.first + z.n), 1);
}

uint8_t* Engine::pool_at(size_t v) const {
    size_t c = 0;
    while (v >= pool_sizes_[c].first + pool_sizes_[c].n) ++c;
    return pool_.get() + pool_sizes_[c].at + (v - pool_sizes_[c].first) * pool_sizes_[c].bytes;
}

// The slot of layer il's size that p is in, or SIZE_MAX: p is in the mapping.
size_t Engine::pool_slot_of(int il, const uint8_t* p) const {
    const PoolSize& z = pool_sizes_[pool_size_of_[(size_t) il]];
    const uintptr_t a = (uintptr_t) pool_.get() + z.at, q = (uintptr_t) p;
    if (q < a || q >= a + z.n * z.bytes) return SIZE_MAX;
    return z.first + (size_t) (q - a) / z.bytes;
}

// The slot of layer il's size that the next expert takes: the hand of that size's clock goes past those this call or
// the next one holds, that are being read or that an upload copies, and past the held ones (pool_prot_), clearing
// the mark of those used since it last came by and putting the others back on trial while too many are held.  It
// takes the first trial slot.  A search of every slot for the least lately used one was a few million comparisons a
// step on the thread that computes the CPU's experts.  SIZE_MAX: none after three rounds.
size_t Engine::pool_victim(int il) {
    PoolSize& z = pool_sizes_[pool_size_of_[(size_t) il]];
    for (size_t k = 0; k < 3 * z.n; ++k) {
        const size_t v = z.first + z.hand++ % z.n;
        if (pool_tick_[v] >= pool_now_ || pool_pin_[v] || pool_state_[v].load(std::memory_order_acquire) == 1)
            continue;
        if (pool_prot_[v]) {
            if (pool_ref_[v]) {
                pool_ref_[v] = 0;
            } else if (z.nprot > z.prot_max) {
                pool_prot_[v] = 0;
                --z.nprot;
            }
            continue;
        }
        return v;
    }
    return SIZE_MAX;
}

void Engine::pool_run(const PoolJob& job) {
    if (job.s < 0) {   // the pages of an expert that changed places in kept_ (follow_loop()): read in, or let go
        if (job.s == kTouch) touch_pages(job.src, job.n);
        else drop_pages(job.src, job.n);
        return;
    }
    const size_t v = (size_t) job.s;
    if (!read_file(job.dst, job.src, job.n)) pool_failed_[v].store(1, std::memory_order_relaxed);
    if (pool_pending_[v].fetch_sub(1, std::memory_order_acq_rel) == 1)   // its last piece
        pool_state_[v].store(pool_failed_[v].load(std::memory_order_relaxed) ? 2 : 0, std::memory_order_release);
}

// A read thread without a job spins 2 ms before it sleeps: an answer queues reads every layer, a fraction of a
// millisecond apart.
void Engine::pool_loop() {
    for (;;) {
        const Clock::time_point t0 = Clock::now();
        while (pool_n_.load(std::memory_order_acquire) == 0 && Clock::now() - t0 < std::chrono::milliseconds(2))
            _mm_pause();
        PoolJob job;
        {
            std::unique_lock<std::mutex> l(pool_m_);
            pool_cv_.wait(l, [&] { return pool_quit_ || !pool_jobs_.empty() || !pool_pages_.empty(); });
            if (pool_quit_) return;
            std::deque<PoolJob>& q = pool_jobs_.empty() ? pool_pages_ : pool_jobs_;
            job = q.front();
            q.pop_front();
            pool_n_.fetch_sub(1, std::memory_order_relaxed);
        }
        pool_run(job);
    }
}

// A thread that waits for a read takes the first piece queued (those of the reads a call waits for are first);
// false when none is.  Spinning, the CPU's threads left a call's few late experts to the read threads alone.
bool Engine::pool_help() {
    PoolJob job;
    {
        std::lock_guard<std::mutex> l(pool_m_);
        if (pool_jobs_.empty()) return false;
        job = pool_jobs_.front();
        pool_jobs_.pop_front();
        pool_n_.fetch_sub(1, std::memory_order_relaxed);
    }
    pool_run(job);
    return true;
}

// Expert e of L into slot s by the read threads, in pieces of 256 KiB: ahead of what is queued with `first`, for a
// call that is about to compute it.  The threads that wait for it share its pieces; in pieces of 64 KiB a piece took
// 89 to 109 us on the laptop while the others read (one of 256 KiB 143 to 154), and the call waited longer.
void Engine::pool_queue(const Layer& L, size_t e, int32_t s, bool first) {
    constexpr int kMost = 64;
    uint8_t* p = pool_at((size_t) s);
    const std::pair<uint8_t*, const uint8_t*> part[3] = {{p, L.exp_gate + e * L.gu_stride},
                                                         {p + L.gu_bytes, L.exp_up + e * L.u_stride},
                                                         {p + L.gu2(), L.exp_down + e * L.d_bytes}};
    const size_t bytes[3] = {L.gu_bytes, L.u_bytes, L.d_bytes};
    // pieces of the larger experts grow so that each matrix takes at most a third of jobs
    const size_t most = (std::max({bytes[0], bytes[1], bytes[2]}) + kMost / 3 - 1) / (kMost / 3);
    const size_t piece = std::max(size_t(256) << 10, (most + 4095) / 4096 * 4096);
    PoolJob jobs[kMost];
    int n = 0;
    for (int k = 0; k < 3; ++k)
        for (size_t at = 0; at < bytes[k]; at += piece)
            jobs[n++] = {part[k].first + at, part[k].second + at, std::min(piece, bytes[k] - at), s};
    pool_failed_[(size_t) s].store(0, std::memory_order_relaxed);
    pool_pending_[(size_t) s].store(n, std::memory_order_relaxed);
    pool_state_[(size_t) s].store(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> l(pool_m_);
        if (first) pool_jobs_.insert(pool_jobs_.begin(), jobs, jobs + n);
        else pool_jobs_.insert(pool_jobs_.end(), jobs, jobs + n);
        pool_n_.fetch_add(n, std::memory_order_release);
    }
    pool_cv_.notify_all();
}

// qwen35moe's: the router's scores with Options::cache_bias added to the ones of the experts in VRAM choose the
// experts; the weights are the router's own, a softmax of the scores it gave the chosen ones.
void Engine::route_cached(const Layer& L, int il, const float* scores, int32_t token, int* ids, float* w, float* tmp) {
    for (int e = 0; e < n_expert_; ++e) bias_rl_[(size_t) e] = scores[e] + (L.slot[(size_t) e] ? opt_.cache_bias : 0.f);
    route(il, bias_rl_.data(), token, ids, w, tmp);
    float mx = scores[ids[0]], sum = 0.f;
    for (int k = 1; k < n_used_; ++k) mx = std::max(mx, scores[ids[k]]);
    for (int k = 0; k < n_used_; ++k) sum += w[k] = std::exp(scores[ids[k]] - mx);
    for (int k = 0; k < n_used_; ++k) w[k] /= sum;
}

// The call's experts from pool_'s slots.  One not there takes the slot pool_victim() gives and goes on
// pool_reads_, which moe() hands the read threads ahead of everything else.
void Engine::pool_take(const Layer& L, int il, CpuExpert* host, int n) {
    const size_t ne = (size_t) n_expert_;
    ++pool_now_;
    pool_reads_.clear();
    for (int i = 0; i < n; ++i) {
        const size_t e = (size_t) (host[i].gate - L.exp_gate) / L.gu_stride;
        if (kept_[(size_t) il * ne + e]) continue;   // through the mapping
        int32_t& s = pool_of_[(size_t) il * ne + e];
        if (s >= 0 && !pool_prot_[(size_t) s]) {   // used after it was read: held
            pool_prot_[(size_t) s] = 1;
            ++pool_sizes_[pool_size_of_[(size_t) il]].nprot;
        }
        if (s < 0) {
            const size_t v = pool_victim(il);
            if (v == SIZE_MAX) continue;   // every slot held by this call or the next: the mapping, then
            if (pool_who_[v] >= 0) pool_of_[(size_t) pool_who_[v]] = -1;
            pool_who_[v] = (int64_t) ((size_t) il * ne + e);
            s = (int32_t) v;
            pool_reads_.push_back({e, s, false});
        }
        pool_tick_[(size_t) s] = pool_now_;
        pool_ref_[(size_t) s] = 1;
        uint8_t* p = pool_at((size_t) s);
        host[i].gate = p;
        host[i].up = p + L.gu_bytes;
        host[i].down = p + L.gu2();
    }
}

// Expert e of layer il's bytes in its pool_ slot, or nullptr: none, or still being read.
const uint8_t* Engine::pool_has(int il, int e) const {
    const int32_t v = pool_ ? pool_of_[(size_t) il * n_expert_ + (size_t) e] : -1;
    if (v < 0 || pool_state_[(size_t) v].load(std::memory_order_acquire) != 0) return nullptr;
    return pool_at((size_t) v);
}

void Engine::pool_read(const Layer& L, PoolRead& r) {
    uint8_t* p = pool_at((size_t) r.s);
    r.ok = read_file(p, L.exp_gate + r.e * L.gu_stride, L.gu_bytes) &&
           read_file(p + L.gu_bytes, L.exp_up + r.e * L.u_stride, L.u_bytes) &&
           read_file(p + L.gu2(), L.exp_down + r.e * L.d_bytes, L.d_bytes);
}

// The routed experts of tokens that come a few at a time.  moe_router() is on its way, with ev_ behind it.  True
// when experts were computed on the CPU: their sum waits in h_cpu_ for moe_add().  `first` is run once, by the
// thread that hands the cached experts to the card and ahead of them.
bool Engine::moe(Layer& L, int il, int n, const std::function<void()>* first) {
    const size_t e_row = (size_t) n_embd_;
    Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    stats_.route_wait_ms += ms_since(t0);
    if (block_pass_) route_block(il, h_rl_, n);
    if (pool_ && prompt_rows_ >= 4096 && !block_pass_) {
        rekeep();
        pool_reseed();
    }
    int ahead_j = -1;   // vram_ahead()'s layer: launch() sends its guesses
    size_t ahead_part = 0;
    const bool aheads = ahead_slots_ > 0 && !freeze_experts_;   // a frozen answer's experts go where its first's did
    if (prefetch_k_ > 0 && !block_pass_ && n <= 16) {   // as moe_router() asked for it
        size_t part = 0;
        for (size_t q = 0; q < 8; ++q)
            if (ev_ == ev_row_[q]) part = q;
        int j = il + 1;
        while (j < n_layer_ && !layers_[(size_t) j].moe) ++j;
        if (j < n_layer_ && pool_) pool_ahead(j, n, part, prefetch_k_);
        else if (j < n_layer_ && aheads) ahead_j = j, ahead_part = part;
    }
    unsigned read = 0;   // the groups of vram_ahead() this call reads

    const float limit = clamp_exp_[(size_t) il];
    const size_t xq_row = kernels::native_q8_1_bytes(n_embd_);
    bool any_host = false;
    // A window's tokens together: the CPU reads an expert that several of them chose once.
    for (int at = 0; at < n; at += kCpuRows) {
        const int m = std::min(kCpuRows, n - at);
        struct Cached {
            const uint8_t* slot;
            float w;
            int row;
            int e;
        };
        CpuExpert host[kCpuRows * kMaxUsed];
        Cached cached[kCpuRows * kMaxUsed];
        int32_t host_e[kCpuRows * kMaxUsed], host_k[kCpuRows * kMaxUsed];   // a host entry's expert and its place
#if !defined(STRATA_GPU_METAL)
        ExpertCard::Job card2[kCpuRows * kMaxUsed];   // --expert-gpu: the experts the second card holds
        int n_far = 0;
#endif
        float host_rel[kCpuRows * kMaxUsed];   // and its weight's share of its token's experts' weight
        int n_host = 0, n_cached = 0;
        // A row's routing is a softmax over every expert and a search for the likeliest: 3.4 us a row of Qwen3.6's
        // 256, which on this thread was 27 us of a step's half of 8 rows.  The workers take a row each, as a prompt's
        // rows (moe_rows()); --cache-bias's scores are in one buffer, so its rows go one by one here.
        int ids_of[kCpuRows][kMaxUsed];
        float w_of[kCpuRows][kMaxUsed];
        // route_tmp_ has a share for each host thread: the workers' rows take a share each, rows on this thread the first
        const bool spread = m > 1 && opt_.cache_bias <= 0.f && route_tmp_.size() >= (size_t) m * 2 * n_expert_;
        const auto route_row = [&](int t) {
            int* ids = ids_of[t];
            float* w = w_of[t];
            if (step_ids_) {   // step()'s rows: the card's picks and weights (step_ids_)
                const size_t from = (size_t) (at + t) * n_used_;
                std::copy(step_ids_ + from, step_ids_ + from + n_used_, ids);
                std::copy(h_wts_ + from, h_wts_ + from + n_used_, w);
                return;
            }
            const float* scores = h_rl_ + (size_t) (at + t) * n_expert_;
            float* tmp = route_tmp_.data() + (size_t) (spread ? t : 0) * 2 * n_expert_;
            if (opt_.cache_bias > 0.f) route_cached(L, il, scores, tok_ ? tok_[at + t] : -1, ids, w, tmp);
            else route(il, scores, tok_ ? tok_[at + t] : -1, ids, w, tmp);
        };
        if (spread)
            workers_->run(m, route_row);
        else
            for (int t = 0; t < m; ++t) route_row(t);
        for (int t = 0; t < m; ++t) {
            const int* ids = ids_of[t];
            const float* w = w_of[t];
            float w_all = 0.f;
            for (int k = 0; k < n_used_; ++k) w_all += w[k];
            for (int k = 0; k < n_used_; ++k) {
                // a block of guesses routed among fewer experts than a token takes: the rest weigh nothing
                if (block_pass_ && w[k] == 0.f) continue;
                if (opt_.route_top > 0 && k >= opt_.route_top) {   // --route-top: the token's last places
                    ++stats_.tail_experts;
                    continue;
                }
                host_rel[n_host] = w_all > 0.f ? w[k] / w_all : 1.f;
                const size_t e = (size_t) ids[k];
                // a block's guesses are not the answer's tokens: the cache is not filled by them
                if (!block_pass_) {
                    ++use_[(size_t) il * n_expert_ + e];
                    if (n_swap_ > 0) recent_[(size_t) il * n_expert_ + e] += 1.f;
                }
                // NVFP4's down scale goes on the weight (ExpertScales)
                const float we = L.h_s_down.empty() ? w[k] : w[k] * L.h_s_down[e];
                if (const uint8_t* s = L.slot[e]) {
                    cached[n_cached++] = {s, we, at + t, (int) e};
                    continue;
                }
                if (const uint8_t* s = aheads ? ahead_of(il, (int) e, read) : nullptr) {
                    cached[n_cached++] = {s, we, at + t, (int) e};
                    ++stats_.ahead_used;
                    continue;
                }
#if !defined(STRATA_GPU_METAL)
                if (const uint8_t* s = ecard_ ? ecard_->slot(il, (int) e) : nullptr) {
                    card2[n_far++] = {s, we, t, (int) e};
                    continue;
                }
#endif
                host_e[n_host] = (int32_t) e;
                host_k[n_host] = k;
                const float gs = L.h_s_gate.empty() ? 1.f : L.h_s_gate[e], us = L.h_s_up.empty() ? 1.f : L.h_s_up[e];
                if (const uint8_t* m = mirror(il, (int) e))
                    host[n_host++] = {m, m + L.gu_bytes, m + L.gu2(), we, t, gs, us};
                else
                    host[n_host++] = {L.exp_gate + e * L.gu_stride, L.exp_up + e * L.u_stride, L.exp_down + e * L.d_bytes,
                                      we, t, gs, us};
            }
        }
        // Strata's STRATA_ROUTE_TAIL_SKIP: an expert outside VRAM that every token here ranks at tail_skip or later
        // adds little to any of them, so it is not computed.  The others keep the weights the router gave them.
        // Options::skip_weight in place of the places: one whose weight is under that share of its token's experts'
        // weight, for every token here.  It keeps an expert the router weighs high at any place, and leaves out one it
        // weighs low at any place, so for the same change of the answers it leaves out more (docs/BACKENDS.md).
        int32_t kept_e[kCpuRows * kMaxUsed];   // host_e of the ones kept
        if (opt_.skip_weight > 0.f) {
            int kept = 0;
            for (int i = 0; i < n_host; ++i) {
                bool low = true;
                for (int j = 0; j < n_host && low; ++j) low = host_e[j] != host_e[i] || host_rel[j] < opt_.skip_weight;
                if (low) ++stats_.tail_experts;
                else kept_e[kept] = host_e[i], host[kept++] = host[i];
            }
            n_host = kept;
            std::copy(kept_e, kept_e + kept, host_e);
        } else if (opt_.tail_skip > 0) {
            int kept = 0;
            for (int i = 0; i < n_host; ++i) {
                bool low = true;
                for (int j = 0; j < n_host && low; ++j) low = host_e[j] != host_e[i] || host_k[j] >= opt_.tail_skip;
                if (low) ++stats_.tail_experts;
                else kept_e[kept] = host_e[i], host[kept++] = host[i];
            }
            n_host = kept;
            std::copy(kept_e, kept_e + kept, host_e);
        }
        // The miss the most tokens here chose goes up from the mirror now, ahead of the card's experts, and the card
        // computes it in the CPU's place: the CPU's share of a call costs it most where it is one expert more, and
        // the card waits for one upload less than the CPU takes for it.  At 6 GB (vram_ahead()) 85% of the experts
        // went to the card in place of 81%, 182-189 tokens/s in place of 175-187; two or three a call left the card
        // waiting for their uploads, and were slower than none.
        // Not while up_ still copies the guesses for this layer or earlier ones: the card would wait for all of them.
        int own = -1;   // that expert, which launch() uploads; -1: none
        if (aheads && n_host > 0 && at == 0 && !up_busy()) {
            int best = 0, most = 0;
            for (int i = 0; i < n_host; ++i) {
                if (host[i].gate != mirror(il, host_e[i])) continue;
                int uses = 0;
                for (int q = 0; q < n_host; ++q) uses += host_e[q] == host_e[i];
                if (uses > most) best = host_e[i], most = uses;
            }
            if (most > 0) {
                Ahead& g = ahead_[kAheadGroups];
                uint8_t* slot = stage_ + (size_t) kAheadGroups * ahead_slots_ * ahead_stride_;
                g.il = il;
                g.n = 1;
                g.e[0] = best;
                own = best;
                read |= 1u << kAheadGroups;
                ++stats_.ahead_sent;
                int kept = 0;
                for (int i = 0; i < n_host; ++i) {
                    if (host_e[i] == best) {
                        cached[n_cached++] = {slot, host[i].weight, at + host[i].row, best};
                        ++stats_.ahead_used;
                    } else {
                        host[kept] = host[i];
                        host_e[kept++] = host_e[i];
                    }
                }
                n_host = kept;
            }
        }
        if (pool_) pool_take(L, il, host, n_host);
        stats_.gpu_experts += n_cached;
#if !defined(STRATA_GPU_METAL)
        if (n_far > 0) {   // on the second card while the first one and the CPU do theirs
            ecard_->start(L, il, h_x_ + (size_t) at * e_row, m, card2, n_far, limit, gelu_, situ_beta_, situ_lb_);
            stats_.gpu_experts += n_far;
        }
#endif
        stats_.cpu_experts += n_host;
        // The launches cost this thread more than they cost the card, so each one less counts.  The cached experts
        // go kSlots at a time, a launch a stage for all of them: gate and up, the activation, its quantization,
        // down.  Then one launch adds up to kSumRows of them into their tokens' rows, in the order they came.  A
        // format without a kernel for several matrices gets the four launches an expert.
        // Nothing else goes to the stream until the CPU experts are done, so one of their threads hands these to
        // the card meanwhile: on this thread they held the CPU experts up.
        const std::function<void()> launch = [&] {
            if (split_) cudaSetDevice(device_);   // one of the experts' threads, which every card's engine shares
            if (first && at == 0) (*first)();
            if (own >= 0) {   // the miss this call took for the card, up before its experts run
                const Ahead& g = ahead_[kAheadGroups];
                ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) g.read, 0), "upload ahead");
                ck(cudaMemcpyAsync(stage_ + (size_t) kAheadGroups * ahead_slots_ * ahead_stride_, mirror(il, own),
                                   L.gu2() + L.d_bytes, cudaMemcpyHostToDevice, cs(up_)),
                   "upload ahead");
                ck(cudaEventRecord((cudaEvent_t) g.up, cs(up_)), "upload ahead");
                up_last_ = g.up;
            }
            // The upload's uses come last in cached: the experts before them go at once, and only the launches with
            // its uses wait for it.  When every launch waited for it, the card was still busy after the CPU's experts
            // in three calls of four at 6 GB, 43 us on average.  The sums are added in the same order either way.
            int own_at = n_cached;
            while (own >= 0 && own_at > 0 && cached[own_at - 1].e == own) --own_at;
            const size_t ff = (size_t) n_ff_exp_, q8_row = kernels::native_q8_1_bytes(n_ff_exp_);
            // the scratch holds rows_ experts' values, and a launch's down rows stay inside one add_scaled's kSumRows
            const int most = kernels::native_mmvq_slots_wide() ? kSlots : 8;
            const int per = rows_ >= most ? most : rows_ >= 8 ? 8 : 1;
            ScaledRows sum{};
            for (int i0 = 0, m; i0 < n_cached; i0 += m) {
                // nor past the end of add_scaled()'s rows: d_'s rows from the start are the ones not added yet
                m = std::min({per, (i0 < own_at ? own_at : n_cached) - i0, kSumRows - i0 % kSumRows});
                if (i0 == own_at)
                    ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ahead_[kAheadGroups].up, 0), "upload ahead");
                kernels::MmvqSlots gate_up{}, up{}, down{};
                gate_up.lanes = up.lanes = down.lanes = lane_slots_;
                for (int j = 0; j < m; ++j) {
                    const Cached& c = cached[i0 + j];
                    gate_up.w[j] = c.slot;
                    gate_up.x[j] = (const uint8_t*) xq_ + (size_t) c.row * xq_row;
                    gate_up.y[j] = g_ + (size_t) j * 2 * ff;
                    up.w[j] = c.slot + L.gu_bytes;
                    up.x[j] = gate_up.x[j];
                    up.y[j] = gate_up.y[j] + ff;
                    down.w[j] = c.slot + L.gu2();
                    down.x[j] = (const uint8_t*) q8_ + (size_t) j * q8_row;
                    down.y[j] = d_ + (size_t) ((i0 + j) % kSumRows) * e_row;
                }
                // a mixed layer's gate and up are each a matrix of n_ff rows of its own type, up's into the second half
                const int gu_rows = L.mixed() ? n_ff_exp_ : 2 * n_ff_exp_;
                const bool one = m > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, m, n_embd_, gu_rows, st_);
                for (int j = 0; j < m && !one; ++j)
                    kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], n_embd_, gu_rows, 1,
                                         st_);
                if (L.mixed()) {
                    const bool one_u = m > 1 && kernels::native_mmvq_slots((int) L.u_type, up, m, n_embd_, n_ff_exp_, st_);
                    for (int j = 0; j < m && !one_u; ++j)
                        kernels::native_mmvq((int) L.u_type, up.w[j], up.x[j], up.y[j], n_embd_, n_ff_exp_, 1, st_);
                }
                ExpertScales es = expert_scales(L);
                for (int j = 0; j < m && (es.gate || es.up); ++j) es.e[j] = cached[i0 + j].e;
                if (situ_beta_ > 0) situ_rows(g_, h_, n_ff_exp_, m, situ_beta_, situ_lb_, st_, nullptr, &es);
                else if (gelu_) geglu_rows(g_, h_, n_ff_exp_, m, st_, &es);
                else swiglu_clamp_rows(g_, h_, n_ff_exp_, m, limit, st_, &es);
                kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, m, st_);
                if (!(m > 1 && kernels::native_mmvq_slots((int) L.d_type, down, m, n_ff_exp_, n_embd_, st_)))
                    for (int j = 0; j < m; ++j)
                        kernels::native_mmvq((int) L.d_type, down.w[j], down.x[j], down.y[j], n_ff_exp_, n_embd_, 1,
                                             st_);
                for (int j = 0; j < m; ++j) {
                    const Cached& c = cached[i0 + j];
                    const int k = (i0 + j) % kSumRows;
                    sum.x[k] = down.y[j];
                    sum.acc[k] = fo_ + (size_t) c.row * e_row;
                    sum.w[k] = c.w;
                    if (k == kSumRows - 1 || i0 + j == n_cached - 1) add_scaled(sum, k + 1, n_embd_, st_);
                }
            }
#if defined(STRATA_GPU_METAL)
            if (opt_.metal_optimize && n_host > 0)
                ck(gpu::metal::submit(static_cast<gpu::Stream>(st_)), "submitting cached experts");
#endif
            // the next layer's guesses, up while the CPU computes this layer's experts
            if (ahead_j >= 0 && at == 0) vram_ahead(ahead_j, n, ahead_part);
        };

        // The next layer's router wait is what makes h_cpu_ free to write again.
        float* cpu = h_cpu_ + (size_t) at * e_row;
        if (n_host == 0) {
            launch();
            std::fill(cpu, cpu + (size_t) m * e_row, 0.f);
#if !defined(STRATA_GPU_METAL)
            if (n_far > 0) {   // the second card's sums take the CPU's place in moe_add()
                ecard_->finish_into(cpu);
                any_host = true;
            }
#endif
            continue;
        }
        any_host = true;
        t0 = Clock::now();
        const std::function<void()>* meanwhile =
            n_cached > 0 || (first && at == 0) || (ahead_j >= 0 && at == 0) ? &launch : nullptr;
        // The experts still being read (missed now, or read ahead and not there yet) are computed last, by
        // cpu_experts(), and the others meanwhile: read before them, a call's reads held its whole CPU part up.
        uint8_t late[kCpuRows * kMaxUsed];
        bool any_late = false;
        for (const PoolRead& r : pool_reads_) pool_queue(L, r.e, r.s, true);
        for (int i = 0; i < n_host && pool_; ++i) {
            const size_t v = pool_slot_of(il, host[i].gate);
            late[i] = v != SIZE_MAX && pool_state_[v].load(std::memory_order_acquire) != 0;
            any_late = any_late || late[i];
        }
        // A read that failed is made once more by one of the waiting threads, then through the mapping.
        const std::function<void(int)> wait = [&](int j) {
            const size_t v = pool_slot_of(il, host[j].gate);
            uint8_t st;
            while ((st = pool_state_[v].load(std::memory_order_acquire)) != 0) {
                if (st == 2 && pool_state_[v].compare_exchange_strong(st, 3)) {
                    const size_t x = (size_t) pool_who_[v], e = x % (size_t) n_expert_;
                    PoolRead r{e, (int32_t) v, false};
                    pool_read(L, r);
                    if (!r.ok) {
                        uint8_t* p = pool_at(v);
                        std::memcpy(p, L.exp_gate + e * L.gu_stride, L.gu_bytes);
                        std::memcpy(p + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes);
                        std::memcpy(p + L.gu2(), L.exp_down + e * L.d_bytes, L.d_bytes);
                    }
                    pool_state_[v].store(0, std::memory_order_release);
                    break;
                }
                if (!pool_help()) std::this_thread::yield();
            }
        };
        // The launches go on a thread of their own: as a task of the CPU's first products, every one of those waited
        // for them.  `first` moves the buffers to the other half's rows while it runs, so this call's are taken now.
        const float* x = h_x_ + (size_t) at * e_row;
        const bool aside = meanwhile && aside_;
        if (aside) aside_->start(*meanwhile);
        try {
            cpu_experts(*workers_, host, n_host, m, (int) L.gu_type, (int) L.u_type, (int) L.d_type, n_embd_, n_ff_exp_,
                        x, limit, gelu_, cpu, aside ? nullptr : meanwhile, any_late ? late : nullptr, &wait,
                        situ_beta_, situ_lb_);
        } catch (...) {
            if (aside) try {
                    aside_->finish();
                } catch (...) {
                }
            throw;
        }
        if (aside) aside_->finish();
#if !defined(STRATA_GPU_METAL)
        if (n_far > 0) ecard_->finish_into(cpu);
#endif
        pool_reads_.clear();
        stats_.cpu_ms += ms_since(t0);
    }
    for (int h = 0; read; ++h, read >>= 1)
        if (read & 1u) ck(cudaEventRecord((cudaEvent_t) ahead_[h].read, cs(st_)), "upload ahead");
    return any_host;
}

// The CPU's experts of n tokens into the routed experts' sum.
void Engine::moe_add(int n) { axpy(h_cpu_, 1.f, fo_, n * n_embd_, true, st_); }

// With every routed expert in VRAM, a few tokens' experts are routed on the card and found there by its picks
// (moe_card()), so a layer never waits for the host.  The cache's slots go to the card again when they changed.
bool Engine::card_moe(int n) {
    const bool bell = bell_on(n);
    if (!(card_routes() || bell) || !d_slots_ || n * n_used_ > card_rows() || opt_.dump) return false;
#if !defined(STRATA_GPU_METAL)
    if (ecard_) return false;   // --expert-gpu: the host routes, and hands the second card its experts (moe())
#endif
    const bool mirrored = !all_cached() && !bell;   // with the doorbell the misses are the host's, not the mirror's
    const auto far_of = [&](int il, int e) -> void* {
        return layers_[(size_t) il].slot[(size_t) e] ? nullptr : (void*) mirror(il, e);
    };
    bool same = !h_slots_.empty() && (!mirrored || !h_far_.empty());
    for (int il = lo_; il < hi_ && same; ++il) {
        const Layer& L = layers_[(size_t) il];
        for (int e = 0; e < n_expert_ && same && L.moe; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            same = h_slots_[i] == L.slot[(size_t) e] && (!mirrored || h_far_[i] == far_of(il, e));
        }
    }
    if (!same) {
        h_slots_.assign((size_t) n_layer_ * n_expert_, nullptr);
        if (mirrored) h_far_.assign((size_t) n_layer_ * n_expert_, nullptr);
        for (int il = lo_; il < hi_; ++il) {
            if (!layers_[(size_t) il].moe) continue;
            std::copy(layers_[(size_t) il].slot.begin(), layers_[(size_t) il].slot.end(),
                      h_slots_.begin() + (ptrdiff_t) il * n_expert_);
            for (int e = 0; e < n_expert_ && mirrored; ++e) {
                const size_t i = (size_t) il * n_expert_ + e;
                if (!h_slots_[i] && !(h_far_[i] = far_of(il, e))) {   // in neither VRAM nor the mirror
                    h_slots_.clear();
                    h_far_.clear();
                    return false;
                }
            }
        }
        ck(cudaMemcpy(d_slots_, h_slots_.data(), h_slots_.size() * sizeof(void*), cudaMemcpyHostToDevice),
           "expert slots");
        if (d_zslots_) {   // the doorbell's table: a miss reads the expert of zeros
            std::vector<void*> z(h_slots_);
            for (void*& p : z)
                if (!p) p = zero_slot_;
            ck(cudaMemcpy(d_zslots_, z.data(), z.size() * sizeof(void*), cudaMemcpyHostToDevice), "expert slots");
        }
        if (mirrored)
            ck(cudaMemcpy(d_far_, h_far_.data(), h_far_.size() * sizeof(void*), cudaMemcpyHostToDevice),
               "expert slots");
        slots_moved();
#if defined(STRATA_GPU_METAL)
        ck(gpu::metal::register_indirect(d_slots_, h_slots_.data(), h_slots_.size()), "expert dependencies");
#endif
    }
    return true;
}

// moe()'s launches for a few tokens whose experts are all in VRAM, with the card's picks for the host's: routed on
// the card, an expert's slot read from d_slots_ by its pick.  False, with nothing launched: this layer is routed
// on the host, or its formats have no kernel for picks on the card.
bool Engine::moe_card(Layer& L, int il, int n) {
    if (L.card == 0) return false;
    if (!kernels::native_mmvq_ids_supported((int) L.gu_type) || !kernels::native_mmvq_ids_supported((int) L.u_type) ||
        !kernels::native_mmvq_ids_supported((int) L.d_type) || !route_card(il, n, d_ids_, d_wts_)) {
        L.card = 0;
        return false;
    }
    L.card = 1;
    const int rr = n * n_used_;
    const void* const* table = d_slots_ + (size_t) il * n_expert_;
    [[maybe_unused]] const bool rings = bell_ && n == 1;
#if !defined(STRATA_GPU_METAL)
    if (rings) {   // the doorbell: the misses to the host, a token's place under --tail-skip's
        if (!block_pass_) count_picks(d_ids_, rr, n_expert_, d_use_ + (size_t) il * n_expert_, st_);
        bell_ring(table, d_ids_, d_wts_, n_used_, opt_.tail_skip > 0 ? opt_.tail_skip : n_used_, il, xn_, h_x_, n_embd_,
                  h_bell_, d_bell_seq_, st_);
        table = d_zslots_ + (size_t) il * n_expert_;
    } else if (!all_cached()) {   // card_far_: the experts it lacks, from the mirror into the staging buffer first
        far_plan(table, d_far_ + (size_t) il * n_expert_, d_ids_, rr, n_expert_, stage_, ahead_stride_, d_tab_,
                 d_fetch_, d_nfetch_, st_);
        far_pull(d_fetch_, d_nfetch_, rr, (L.gu2() + L.d_bytes + 15) / 16 * 16, st_);
        if (!block_pass_) count_picks(d_ids_, rr, n_expert_, d_use_ + (size_t) il * n_expert_, st_);   // follow()
        table = d_tab_;
    }
#endif
    const size_t xq_row = kernels::native_q8_1_bytes(n_embd_), ff = (size_t) n_ff_exp_;
    // a mixed layer's gate and up are each a matrix of n_ff rows of its own type, up's into the second half
    const kernels::MmvqIds gate_up{table, d_ids_, 0, xq_, xq_row, n_used_, g_, 2 * ff, lane_slots_};
    kernels::native_mmvq_ids((int) L.gu_type, gate_up, rr, n_embd_, L.mixed() ? n_ff_exp_ : 2 * n_ff_exp_, st_);
    if (L.mixed()) {
        const kernels::MmvqIds up{table, d_ids_, L.gu_bytes, xq_, xq_row, n_used_, g_ + ff, 2 * ff, lane_slots_};
        kernels::native_mmvq_ids((int) L.u_type, up, rr, n_embd_, n_ff_exp_, st_);
    }
    const ExpertScales es = expert_scales(L, d_ids_);
    // SiTU's rows come out quantized too (situ_rows()), the bytes native_quantize_q8_1() gives them
    if (situ_beta_ > 0) situ_rows(g_, h_, n_ff_exp_, rr, situ_beta_, situ_lb_, st_, q8_, &es);
    else if (gelu_) geglu_rows(g_, h_, n_ff_exp_, rr, st_, &es);
    else swiglu_clamp_rows(g_, h_, n_ff_exp_, rr, clamp_exp_[(size_t) il], st_, &es);
    // the down product's scales on the weights; a family that sums the experts later multiplies them there
    moe_left_scale_ = moe_sum_later(n) && !L.h_s_down.empty() ? fp(L.s_down) : nullptr;
    if (!L.h_s_down.empty() && !moe_left_scale_) expert_weights(d_wts_, fp(L.s_down), d_ids_, rr, st_);
    if (!(situ_beta_ > 0)) kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_ * rr, 1, st_);   // the rows back to back, as one vector
    const kernels::MmvqIds down{table, d_ids_, L.gu2(), q8_, kernels::native_q8_1_bytes(n_ff_exp_), 1, d_,
                                (size_t) n_embd_, lane_slots_};
    kernels::native_mmvq_ids((int) L.d_type, down, rr, n_ff_exp_, n_embd_, st_);
    if (moe_sum_later(n)) moe_left_ = true;
    else moe_sum(d_, d_wts_, fo_, n_embd_, n_used_, n, st_);
#if !defined(STRATA_GPU_METAL)
    if (rings) bell_add(h_bell_, d_bell_seq_, h_cpu_, fo_, n_embd_, st_);   // (bells(): no sum later then)
#endif
    return true;
}

bool Engine::bell_on(int n) const {
#if defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP)
    (void) n;
    return false;
#else
    return n == 1 && h_bell_ && !all_cached() && !card_far_ && !pool_ && !split_ && !block_pass_ && !opt_.dump;
#endif
}

// The doorbell's host side, after a token's launches: each ring is a layer's experts that the card lacks, computed by
// the workers as moe() computes its CPU experts (the token's input in h_x_, their sum into h_cpu_) and answered, until
// the card is past the token's last launch.  Here, not on threads of its own: the token's launches are all handed
// over by then, and this thread would wait for the card anyway.
void Engine::bell_serve() {
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    ck(cudaEventRecord((cudaEvent_t) ev_bell_, cs(st_)), "the doorbell");
    for (;;) {
        const uint32_t r = h_bell_->ring;
        if (r != bell_seen_) {
            std::atomic_thread_fence(std::memory_order_acquire);
            bell_seen_ = r;
            try {
                bell_job();
            } catch (...) {
                h_bell_->abort = 1;   // the card's waits end; what it adds is spoiled, and the call throws
                cudaStreamSynchronize(cs(st_));
                h_bell_->abort = 0;
                bell_seen_ = h_bell_->ring;
                throw;
            }
            std::atomic_thread_fence(std::memory_order_release);
            h_bell_->ack = r;
            continue;
        }
        const cudaError_t q = cudaEventQuery((cudaEvent_t) ev_bell_);
        if (q == cudaSuccess) return;
        if (q != cudaErrorNotReady) ck(q, "the doorbell");
        _mm_pause();
    }
#endif
}

void Engine::bell_job() {
    const int il = h_bell_->layer, n = std::min<int>(h_bell_->n, kBellJobs);
    const Layer& L = layers_[(size_t) il];
    CpuExpert host[kBellJobs];
    for (int j = 0; j < n; ++j) {
        const size_t e = (size_t) h_bell_->e[j];
        const float w = L.h_s_down.empty() ? h_bell_->w[j] : h_bell_->w[j] * L.h_s_down[e];   // NVFP4's down scale
        const float gs = L.h_s_gate.empty() ? 1.f : L.h_s_gate[e], us = L.h_s_up.empty() ? 1.f : L.h_s_up[e];
        if (const uint8_t* m = mirror(il, (int) e))
            host[j] = {m, m + L.gu_bytes, m + L.gu2(), w, 0, gs, us};
        else
            host[j] = {L.exp_gate + e * L.gu_stride, L.exp_up + e * L.u_stride, L.exp_down + e * L.d_bytes, w, 0, gs, us};
    }
    const Clock::time_point t0 = Clock::now();
    cpu_experts(*workers_, host, n, 1, (int) L.gu_type, (int) L.u_type, (int) L.d_type, n_embd_, n_ff_exp_, h_x_,
                clamp_exp_[(size_t) il], gelu_, h_cpu_, nullptr, nullptr, nullptr, situ_beta_, situ_lb_);
    stats_.cpu_ms += ms_since(t0);
    stats_.cpu_experts += n;
    stats_.gpu_experts -= n;   // run() counted every pick of the token as the card's
}

// m experts from the mapped file into the staging buffer: gate and up of each back to back from the start, the
// downs after them.  The two pinned buffers take turns, so the workers fill one while the other uploads.  `alone`:
// stagers_ fill them, while the workers compute experts.
void Engine::stage(const Layer& L, const int32_t* experts, int m, void* stream, bool alone) {
    staged_il_ = -1;   // whatever the buffer held
    for (Ahead& g : ahead_) {   // vram_ahead()'s groups go, once the calls that read them are done
        if (g.il < 0) continue;
        ck(cudaStreamWaitEvent(cs(stream), (cudaEvent_t) g.read, 0), "expert staging");
        g.il = -1;
    }
    const size_t step = L.gu_step, down_at = pad((size_t) m * step);
#if !defined(STRATA_GPU_METAL)
    if (!mirror_of_.empty()) {
        const int il = (int) (&L - layers_.data());
        // The copy ahead of the routing runs beside the layer's work, where the gather's blocks, waiting on the bus,
        // held SMs that work needed (DeepSeek V4 Flash's moe_sum took 9 times as long beside it): the copy engines
        // take that one.
        const bool dma = stream == up_;
        GatherPiece* q = dma ? nullptr : gather_slot();
        int k = 0;
        const auto piece = [&](const void* src, void* dst, size_t n) {
            if (!dma && (((uintptr_t) src | (uintptr_t) dst) % 16) == 0 && n % 16 == 0) q[k++] = {src, dst, n, 0};
            else ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, cs(stream)), "expert staging");
        };
        stage_rest_.clear();
        for (int j = 0; j < m; ++j) {
            const uint8_t* src = mirror(il, experts[j]);
            if (!src) {
                stage_rest_.push_back(j);
                continue;
            }
            piece(src, stage_ + (size_t) j * step, L.gu2());
            piece(src + L.gu2(), stage_ + down_at + (size_t) j * L.d_bytes, L.d_bytes);
        }
        if (!dma) {
            gather_pieces(q, k, stream);
            ck(cudaGetLastError(), "expert staging");
            ck(cudaEventRecord((cudaEvent_t) gather_ev_[gather_at_], cs(stream)), "expert staging");
        }
        if (!stage_rest_.empty()) stage_rest(L, experts, stage_rest_, m, stream, alone);
        ck(cudaMemsetAsync(stage_ + down_at + (size_t) m * L.d_bytes, 0, kTail, cs(stream)), "expert staging");
        return;
    }
#endif
    stage_pins();
    const int per = (int) (chunk_bytes_ / (step + L.d_bytes));
    for (int j0 = 0; j0 < m; j0 += per) {
        const size_t c = (size_t) std::min(per, m - j0);
        stage_turn_ ^= 1;
        uint8_t* h = h_stage_[stage_turn_];
        const cudaEvent_t left = (cudaEvent_t) ev_stage_[stage_turn_];
        ck(cudaEventSynchronize(left), "expert staging");
        // Under the RAM budget an expert outside kept_ comes from its pool_ slot, or else from the file, not the
        // mapping (read_file()).  Nothing takes a slot while a prompt batch is read.
        const int il = (int) (&L - layers_.data());
        const auto put = [&](void* dst, const void* src, size_t bytes) {
            if (stream_stage_) stage_copy_streaming(dst, src, bytes);
            else std::memcpy(dst, src, bytes);
        };
        const auto copy = [&](int i) {
            const size_t j = (size_t) i / 3, e = (size_t) experts[(size_t) j0 + j];
            const uint8_t* in = pool_has(il, (int) e);
            const bool mapped = !pool_ || kept_[(size_t) il * n_expert_ + e];
            const auto get = [&](uint8_t* dst, const uint8_t* src, size_t n, size_t at) {
                if (in) put(dst, in + at, n);
                else if (mapped || !read_file(dst, src, n)) put(dst, src, n);
            };
            if (i % 3 == 0) get(h + j * step, L.exp_gate + e * L.gu_stride, L.gu_bytes, 0);
            else if (i % 3 == 1) get(h + j * step + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes, L.gu_bytes);
            else get(h + c * step + j * L.d_bytes, L.exp_down + e * L.d_bytes, L.d_bytes, L.gu2());
        };
        (alone ? *stagers_ : *workers_).run((int) (3 * c), copy);
        ck(cudaMemcpyAsync(stage_ + (size_t) j0 * step, h, c * step, cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        ck(cudaMemcpyAsync(stage_ + down_at + (size_t) j0 * L.d_bytes, h + c * step, c * L.d_bytes,
                           cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        ck(cudaEventRecord(left, cs(stream)), "expert staging");
    }
    ck(cudaMemsetAsync(stage_ + down_at + (size_t) m * L.d_bytes, 0, kTail, cs(stream)), "expert staging");
}

// stage()'s experts `which` (their places in experts[0..m)) that the mirror lacks: through the pinned buffers one by
// one, each to its place.
void Engine::stage_rest(const Layer& L, const int32_t* experts, const std::vector<int>& which, int m, void* stream,
                        bool alone) {
    const size_t step = L.gu_step, down_at = pad((size_t) m * step), one = L.gu2() + L.d_bytes;
    stage_pins();
    const int per = (int) std::max<size_t>(chunk_bytes_ / one, 1);
    for (size_t i0 = 0; i0 < which.size(); i0 += (size_t) per) {
        const size_t c = std::min((size_t) per, which.size() - i0);
        stage_turn_ ^= 1;
        uint8_t* h = h_stage_[stage_turn_];
        ck(cudaEventSynchronize((cudaEvent_t) ev_stage_[stage_turn_]), "expert staging");
        (alone ? *stagers_ : *workers_).run((int) c, [&](int i) {
            const size_t e = (size_t) experts[which[i0 + (size_t) i]];
            uint8_t* d = h + (size_t) i * one;
            std::memcpy(d, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            std::memcpy(d + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes);
            std::memcpy(d + L.gu2(), L.exp_down + e * L.d_bytes, L.d_bytes);
        });
        for (size_t i = 0; i < c; ++i) {
            const size_t j = (size_t) which[i0 + i];
            ck(cudaMemcpyAsync(stage_ + j * step, h + i * one, L.gu2(), cudaMemcpyHostToDevice, cs(stream)), "expert staging");
            ck(cudaMemcpyAsync(stage_ + down_at + j * L.d_bytes, h + i * one + L.gu2(), L.d_bytes, cudaMemcpyHostToDevice,
                               cs(stream)),
               "expert staging");
        }
        ck(cudaEventRecord((cudaEvent_t) ev_stage_[stage_turn_], cs(stream)), "expert staging");
    }
}

// The next of the gathers' piece lists, once the card has read it for the gather before (they take turns).
GatherPiece* Engine::gather_slot() {
    constexpr int kRing = (int) (sizeof(gather_ev_) / sizeof(gather_ev_[0]));
    const size_t per = 2 * (size_t) n_expert_;
    if (!gather_ring_) {
        gather_ring_ = (GatherPiece*) hmalloc(kRing * per * sizeof(GatherPiece), "pinned memory");
        for (void*& e : gather_ev_) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            e = ev;
        }
    }
    gather_at_ = (gather_at_ + 1) % kRing;
    ck(cudaEventSynchronize((cudaEvent_t) gather_ev_[gather_at_]), "expert staging");
    return gather_ring_ + (size_t) gather_at_ * per;
}

// ---- the pinned mirror
// With every routed expert outside VRAM in RAM (no RAM budget) on an NVIDIA card, those experts get a copy in pinned
// host memory, in a card slot's layout (gate | up | down; a slot's bytes rounded up to 64), made at the end of the
// load.  A read then stages them by one gather on the card a call, which reads them over the bus itself, where the
// copy was the CPU's into a pinned buffer and an upload from there; a swap uploads one with no copy of the CPU's; the
// CPU experts read them there.  The mapping's pages of the routed experts then leave the working set: the RAM the
// process holds for its experts is the mirror, the ones the card does not have.
//
// Qwen3.6 UD-Q3_K_M on lab-pro6000 (GPU2, shared), the card held to its budget by --vram-pool, a prompt of 4,833
// tokens read first and second after the load, two runs each, tokens/s, and the process's RAM after them:
//   card held to   6 GB: 3,629-3,664 / 3,768-3,809 -> 4,324-4,434 / 4,381-4,659, 13.8 -> 13.3 GB
//                8.9 GB: 3,403-3,446 / 3,478-3,486 -> 4,810-5,082 / 4,507-4,678, 12.2 -> 10.3 GB
//                 12 GB: 4,059-4,535 / 4,114-4,293 -> 5,820-5,852 / 5,217-5,499, 10.6 ->  7.1 GB
//                 16 GB: 5,765 / 6,285-6,295 -> 5,785-6,199 / 6,608-7,212, 8.0 -> 2.9 GB
// The answers write as fast as before, and the load takes 1.4-2 s more.  Made on a thread of its own after the
// load, the mirror held the first answer to 91-123 tokens/s where 145-245.
//
// An expert that leaves the card (shrink_cache(), a swap's outgoing one) is copied back into a spare slot, so that it
// needs no read of the file; one that comes onto it gives its slot back.  64 spares, for the swaps; one a wider
// staging buffer takes from the cache goes back to the mapping.  STRATA_POLY_MIRROR=0 makes none; on Windows there is
// none unless STRATA_POLY_MIRROR=1 or --lock-experts.
//
// Options::lock_experts takes the mirror as its copy of the experts outside VRAM: pinned pages are never paged out
// or dropped, and the NVIDIA driver pins them past RLIMIT_MEMLOCK, which is 8 MiB on many Linux desktops and kept
// mlock() of the mapping (lock_outside()) from holding anything there (issue #79).  An expert that leaves the card
// with no spare left is copied into the mirror too, by lock_loop()'s thread, so it comes to stay in RAM as well.
// Where there is no mirror, the mapping's pages are locked as before.
void Engine::build_mirror() {
#if defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP)
    return;
#else
    const char* asked = std::getenv("STRATA_POLY_MIRROR");
    int unified = 0;   // the card reads host memory at the host's addresses (the gather)
    cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, device_);
    if ((asked && asked[0] == '0') || !unified || opt_.ram_experts_mib != -1 || pool_ || !mmq_ || !stage_ ||
        all_cached() || split_ || n_trade_ < 1) {
        if (opt_.lock_experts && !all_cached())
            std::fprintf(stderr, "strata-poly: --lock-experts: no pinned copy of the experts outside VRAM in this setup, "
                                 "so their pages in the model file are locked\n");
        return;
    }
#ifdef _WIN32
    // None on Windows unless STRATA_POLY_MIRROR=1 asks or --lock-experts takes it: the copy is memory the process holds,
    // where the mapping's pages are the file cache's.  On the RTX 4070 laptop Gemma 4 UD-Q3_K_M with its 7.2 GiB copy
    // read 1,881 tokens/s and wrote 68.2 where poly.10, mapped, read 1,906 and wrote 76.6, and committed 17,227 MiB
    // against 9,513.  Whether it was made turned on a few hundred MiB: the copy and 4 GiB beside it needed 11.2 GiB
    // free, one start on that laptop found 11.0 and made none, another made it.
    if (!(asked && asked[0] == '1') && !opt_.lock_experts) {
        std::fprintf(stderr, "strata-poly: the experts outside VRAM stay in the mapped files: no pinned copy of them on "
                             "Windows (STRATA_POLY_MIRROR=1 makes one)\n");
        return;
    }
#endif
    const size_t ne = (size_t) n_expert_;
    struct Want {
        int il, e;
    };
    std::vector<std::vector<Want>> want((size_t) n_trade_);
    std::vector<size_t> stride((size_t) n_trade_, 0), cached((size_t) n_trade_, 0), spares((size_t) n_trade_, 0);
    size_t cached_all = 0;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        stride[(size_t) L.trade] = (L.slot_bytes + 63) / 64 * 64;
        for (int e = 0; e < n_expert_; ++e) {
            if (L.slot[(size_t) e]) {
                ++cached[(size_t) L.trade];
                ++cached_all;
            } else {
                want[(size_t) L.trade].push_back({il, e});
            }
        }
    }
    size_t total = 0;
    for (size_t c = 0; c < (size_t) n_trade_; ++c) {
        if (!stride[c]) continue;
        spares[c] = 1 + (size_t) (64.0 * (cached_all ? (double) cached[c] / (double) cached_all : 1.0));
        total += (want[c].size() + spares[c]) * stride[c];
    }
    constexpr size_t kHeadroom = size_t(4) << 30;   // as the RAM budget's (load())
    size_t avail = ram_available();
#ifdef _WIN32
    if (avail) avail -= std::min(avail, cached_file_bytes());   // as the budget counts them (load())
#endif
    if (!avail || total + kHeadroom > avail) {
        std::fprintf(stderr, "strata-poly: %.1f GiB of RAM available for a %.1f GiB pinned copy of the experts outside VRAM: "
                             "prompts are staged from the mapped files%s\n",
                     (double) avail / (1 << 30), (double) total / (1 << 30),
                     opt_.lock_experts ? ", and --lock-experts locks their pages in the model file" : "");
        return;
    }
    // the copies are read from the files (read_file()), through the file cache, or past it under a cgroup's limit:
    // through the mapping every page of them would be faulted in, and on Windows a fault reads a few pages at a
    // time (map_pages())
    if (views_.empty())
        for (const Shards* sh : {gg_.get(), dg_.get()}) {
            if (!sh) continue;
            const auto v = sh->views();
            for (size_t i = 0; i < v.size(); ++i)
                if (v[i].first) views_.push_back({v[i].first, v[i].second, sh->paths()[i]});
        }
#ifdef __linux__
    // Under a cgroup's limit the file cache is the container's: the pages the load read from the files stay charged
    // to it beside the copy.  On an RTX 5090 pod held to 90 GiB, GLM 5.3 Flash's load left 54 GiB of them, its
    // 80.9 GiB copy took the container to its limit, and the pod ended the engine with SIGKILL while the copy was
    // made or while the next long prompt was read.  A page still mapped stays in the cache, so the mapping lets
    // them go first; what is read from it later is read again.
    if (pool_uncached_ && !opt_.lock_experts)
        for (const FileView& f : views_) {
            drop_pages(f.base, f.bytes);
            if (const int fd = ::open(f.path.c_str(), O_RDONLY | O_CLOEXEC); fd >= 0) {
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                ::close(fd);
            }
        }
#endif
    const Clock::time_point t0 = Clock::now();
    // The memory is the process's own, its pages placed by the threads that fill it and then pinned
    // (cudaHostRegister()): cudaMallocHost()'s came from the card's NUMA node, and the CPU experts, read there by
    // threads on both of lab-pro6000's nodes, wrote 80-102 tokens/s where 127-149 through the mapping.  Where the
    // card cannot read registered memory at the host's addresses, cudaMallocHost() it is.
    int same = 0;
    cudaDeviceGetAttribute(&same, cudaDevAttrCanUseHostPointerForRegisteredMem, device_);
    // Pieces of at most 256 MiB, each allocated, filled and pinned by one of the CPU experts' threads.
    constexpr size_t kPiece = size_t(256) << 20;
    struct Piece {
        size_t c, first, n;   // class c's slots [first, first + n): its experts, then its spares
    };
    std::vector<Piece> pieces;
    for (size_t c = 0; c < want.size(); ++c) {
        if (!stride[c]) continue;
        const size_t all = want[c].size() + spares[c], per = std::max<size_t>(1, kPiece / stride[c]);
        for (size_t i = 0; i < all; i += per) pieces.push_back({c, i, std::min(per, all - i)});
    }
    std::vector<MirrorPiece> got(pieces.size());
    std::atomic<bool> failed{false};
    workers_->run((int) pieces.size(), [&](int k) {
        const Piece& p = pieces[(size_t) k];
        MirrorPiece m{nullptr, p.n * stride[p.c], same != 0};
        if (m.registered) m.base = host_alloc(m.bytes);
        else if (cudaMallocHost((void**) &m.base, m.bytes) != cudaSuccess) m.base = nullptr;
        if (!m.base) {
            failed = true;
            return;
        }
        for (size_t i = p.first; i < p.first + p.n && i < want[p.c].size(); ++i) {
            const Layer& L = layers_[(size_t) want[p.c][i].il];
            const size_t e = (size_t) want[p.c][i].e;
            uint8_t* to = m.base + (i - p.first) * stride[p.c];
            const auto get = [&](uint8_t* dst, const uint8_t* src, size_t n) {
                if (!read_file(dst, src, n, true)) std::memcpy(dst, src, n);
            };
            get(to, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            get(to + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes);
            get(to + L.gu2(), L.exp_down + e * L.d_bytes, L.d_bytes);
        }
        // portable: a worker has no card of its own (cudaSetDevice())
        if (m.registered && cudaHostRegister(m.base, m.bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) !=
                                cudaSuccess) {
            host_free(m.base);
            failed = true;
            return;
        }
        got[(size_t) k] = m;
    });
    if (failed.load()) {
        cudaGetLastError();
        free_pieces(got);
        std::fprintf(stderr, "strata-poly: no pinned copy of the experts outside VRAM: no memory for it%s\n",
                     opt_.lock_experts ? ", so --lock-experts locks their pages in the model file" : "");
        return;
    }
    std::vector<uint8_t*> table(layers_.size() * ne, nullptr);
    std::vector<std::vector<uint8_t*>> spare((size_t) n_trade_);
    size_t n = 0;
    for (size_t k = 0; k < pieces.size(); ++k) {
        const Piece& p = pieces[k];
        for (size_t i = p.first; i < p.first + p.n; ++i) {
            uint8_t* s = got[k].base + (i - p.first) * stride[p.c];
            if (i < want[p.c].size()) {
                table[(size_t) want[p.c][i].il * ne + (size_t) want[p.c][i].e] = s;
                ++n;
            } else {
                spare[p.c].push_back(s);
            }
        }
    }
    // the mapping's pages of every routed expert leave the working set: the mirror has the ones outside VRAM
    for (const Layer& L : layers_)
        if (L.moe) expert_bytes(L, 0, n_expert_, drop_pages);
    mirror_blocks_ = std::move(got);
    mirror_of_ = std::move(table);
    mirror_spare_ = std::move(spare);
    mirror_stride_ = std::move(stride);
    mirror_registered_ = same != 0;
    std::fprintf(stderr, "strata-poly: the %zu experts outside VRAM in %.1f GiB of pinned memory in %.1f s\n", n,
                 (double) total / (1 << 30), ms_since(t0) / 1000.0);
    if (opt_.lock_experts) {
        std::fprintf(stderr, "strata-poly: --lock-experts: copied the %zu experts outside VRAM into %.1f GiB of RAM and "
                             "locked all of it: pinned for the card, which the system does not page out%s; one that "
                             "leaves VRAM later is copied in too, from the model file, between requests\n",
                     n, (double) total / (1 << 30), memlock_note().c_str());
        lock_asked_.assign(layers_.size() * ne, 0);
        lock_thread_ = std::thread([this] { lock_loop(); });
    }

    // card_far_: an answer's few tokens route on the card as with every expert cached, and the card copies the
    // picks it lacks from the mirror itself: a layer does not wait for the host, and a window is one recording.  The
    // staging buffer holds a call's picks.  Its answers are the ones every expert in VRAM gives.
    //
    // A token routed on the card copies what the cache lacks of its picks over the bus; routed on the host, it costs
    // each MoE layer a round trip: the router's scores to the host, the card waiting for the host's launches.  Both
    // are measured here, the copy as far_pull() makes it, the round trip as moe_router() and moe() make it, and the
    // way that takes less a token is the one taken.  The bytes are the profile's: each layer's picks times the share
    // of its picks the cache lacks.  With no profile, nothing to say what an answer misses: on the host.  On lab-241's
    // RTX 4090 D (PCIe 4.0 x8: 12.6 GB/s, round trips of 11-12 us) Qwen3.6 UD-Q3_K_M goes to the host with 95% and
    // 97% of the profile's picks cached, where the card wrote 10-25% slower, and to the card with 99%, where it wrote
    // as fast or faster.  STRATA_POLY_CARD_FAR=1 takes the card, any other value the host.
    ahead_stride_ = (slot_max_ + 4095) / 4096 * 4096;
    {
        double far_bytes = 0;   // a token's, by the profile
        int moe_layers = 0, il_moe = -1;
        size_t bytes = 0;       // a far_pull() of the measure's: the largest slot's
        for (int il = 0; il < n_layer_; ++il) {
            const Layer& L = layers_[(size_t) il];
            if (!L.moe) continue;
            ++moe_layers;
            if (il_moe < 0) il_moe = il;
            bytes = std::max(bytes, (L.gu2() + L.d_bytes + 15) / 16 * 16);
            double all = 0, lacks = 0;
            for (int e = 0; e < n_expert_; ++e) {
                const double u = prior_.empty() ? 0.0 : (double) prior_[(size_t) il * n_expert_ + e];
                all += u;
                if (!L.slot[(size_t) e]) lacks += u;
            }
            if (all > 0) far_bytes += n_used_ * lacks / all * (double) (L.gu2() + L.d_bytes);
        }
        // the copy's picks: experts of the mirror of that size, each once, so that no copy reads what one before it read
        constexpr int kPicks = 4;   // a call's misses, as a layer of an answer has them
        std::vector<FarFetch> fetch;
        for (int il = 0; il < n_layer_ && (int) fetch.size() < card_rows(); ++il) {
            const Layer& L = layers_[(size_t) il];
            for (int e = 0; L.moe && (L.gu2() + L.d_bytes + 15) / 16 * 16 == bytes && e < n_expert_ &&
                            (int) fetch.size() < card_rows(); ++e)
                if (const uint8_t* m = !L.slot[(size_t) e] ? mirror(il, e) : nullptr)
                    fetch.push_back({m, stage_ + (fetch.size() % kPicks) * ahead_stride_});
        }
        const char* cf = std::getenv("STRATA_POLY_CARD_FAR");
        const bool fits = d_far_ && opt_.batch == 0 && (size_t) card_rows() * ahead_stride_ <= stage_narrow_;
        const int groups = (int) fetch.size() / kPicks;
        if (fits && !cf && !prior_.empty() && groups >= 2) {
            // each group of kPicks timed on its own, after one that wakes the bus; the median
            ck(cudaMemcpy(d_fetch_, fetch.data(), fetch.size() * sizeof(fetch[0]), cudaMemcpyHostToDevice), "far copy");
            ck(cudaMemcpy(d_nfetch_, &kPicks, sizeof(int), cudaMemcpyHostToDevice), "far copy");
            std::vector<cudaEvent_t> ev((size_t) groups);
            for (cudaEvent_t& e : ev) ck(cudaEventCreate(&e), "cudaEventCreate");
            far_pull(d_fetch_, d_nfetch_, kPicks, bytes, st_);
            ck(cudaEventRecord(ev[0], cs(st_)), "far copy");
            for (int g = 1; g < groups; ++g) {
                far_pull(d_fetch_ + (size_t) g * kPicks, d_nfetch_, kPicks, bytes, st_);
                ck(cudaEventRecord(ev[(size_t) g], cs(st_)), "far copy");
            }
            ck(cudaEventSynchronize(ev.back()), "far copy");
            std::vector<float> copy;
            for (int g = 1; g < groups; ++g) {
                float ms = 0.f;
                ck(cudaEventElapsedTime(&ms, ev[(size_t) g - 1], ev[(size_t) g]), "far copy");
                copy.push_back(ms);
            }
            std::sort(copy.begin(), copy.end());
            const float copy_ms = copy[copy.size() / 2];
            // the round trip: the router's copy to the host, the host waiting for it, routing the token, and on to the
            // next launch
            ck(cudaMemsetAsync(rl_, 0, (size_t) n_expert_ * sizeof(float), cs(st_)), "round trip");
            std::vector<float> tmp(2 * (size_t) n_expert_);
            int ids[kMaxUsed];
            float w[kMaxUsed];
            std::vector<double> trip;
            for (int k = 0; k < 36; ++k) {
                const Clock::time_point t0 = Clock::now();
                copy2(rl_, h_rl_, n_expert_, xn_, h_x_, n_embd_, st_);
                ck(cudaEventRecord(ev[0], cs(st_)), "round trip");
                ck(cudaEventSynchronize(ev[0]), "round trip");
                route(il_moe, h_rl_, -1, ids, w, tmp.data());
                if (k >= 4) trip.push_back(ms_since(t0));
            }
            for (cudaEvent_t e : ev) cudaEventDestroy(e);
            std::sort(trip.begin(), trip.end());
            const double gbs = copy_ms > 0.f ? (double) kPicks * (double) bytes / copy_ms / 1e6 : 0.0;
            const double far_ms = gbs > 0 ? far_bytes / gbs / 1e6 : 0.0, host_ms = moe_layers * trip[trip.size() / 2];
            card_far_ = gbs > 0 && far_ms < host_ms;
            std::fprintf(stderr, "strata-poly: a token's experts the card lacks: %.1f MiB by the profile, %.2f ms at "
                                 "%.1f GB/s; routed on the host, %d round trips of %.0f us, %.2f ms: routed on the %s\n",
                         far_bytes / (1 << 20), far_ms, gbs, moe_layers, 1000.0 * trip[trip.size() / 2], host_ms,
                         card_far_ ? "card" : "host");
        } else {
            card_far_ = fits && cf && cf[0] == '1';
            if (card_far_)
                std::fprintf(stderr, "strata-poly: experts the card lacks: copied by the card from the mirror "
                                     "(STRATA_POLY_CARD_FAR=1)\n");
        }
    }

    // vram_ahead(): one sequence's answer (step() mixes the staging buffer with other sequences' prompt reads).
    // Whether an upload has arrived decides where an expert is computed, so a cache asked to stay as it is
    // (--expert-swaps 0, the regression tests' way to get the same answer twice) gets none.  STRATA_POLY_AHEAD=0
    // turns it off.
    const char* ahead = std::getenv("STRATA_POLY_AHEAD");
    ahead_slots_ = std::min<int>(Ahead::kSlots, (int) (stage_narrow_ / ahead_stride_ / (kAheadGroups + 1)));
    if ((ahead && ahead[0] == '0') || opt_.batch > 0 || opt_.expert_swaps == 0 || card_far_ || !up_ || !ahead_dev_ ||
        ahead_slots_ < 4) {
        ahead_slots_ = 0;
        return;
    }
    for (Ahead& g : ahead_) {
        for (void** e : {&g.up, &g.read}) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        for (void*& e : g.arrived) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            e = ev;
        }
    }
    prefetch_k_ = kAheadGuesses;   // moe_router() scores the next layer's router
#endif
}

uint8_t* Engine::mirror_take(int il) {
    const int c = layers_[(size_t) il].trade;
    if (mirror_spare_.empty() || c < 0 || mirror_spare_[(size_t) c].empty()) return nullptr;
    uint8_t* s = mirror_spare_[(size_t) c].back();
    mirror_spare_[(size_t) c].pop_back();
    return s;
}

// Options::lock_experts: an expert that leaves the card with no spare slot goes back to the mapping, as without the
// option, and lock_loop() copies it from the files into a slot of its own; lock_settle() makes the copy the mirror's.
// Made where the expert left, the pinned memory for it and the copy back from the card took the read that took the
// cache's VRAM: on a 4090 held to 12 GiB the first reads of Qwen3.6 UD-Q3_K_M were 5-30% slower than the default's.
void Engine::lock_queue(int il, int e) {
    const size_t k = (size_t) il * (size_t) n_expert_ + (size_t) e;
    if (!lock_thread_.joinable() || lock_asked_[k]) return;
    lock_asked_[k] = 1;
    {
        std::lock_guard<std::mutex> l(lock_m_);
        lock_want_.push_back((int64_t) k);
    }
    lock_cv_.notify_one();
}

// The experts lock_queue() asks for, read from the files into slots of pieces of 32 MiB, pinned as the mirror's own
// are, between requests only: from trim() to the next read() or run().  cudaHostRegister() holds the CUDA calls of
// every other thread while it pins: a launch and a sync on another thread waited out all of it, 80-100 ms for 256
// MiB on lab-pro6000 and 128-130 ms on lab-241, 11-14 ms for 32 MiB.  A piece begun as a request comes holds the
// request up no longer than that.  Only this thread touches lock_blocks_ until the engine joins it; it reads none
// of what the main thread changes (the slots, mirror_of_).
void Engine::lock_loop() {
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    cudaSetDevice(device_);
    const size_t ne = (size_t) n_expert_;
    std::vector<std::vector<uint8_t*>> slots((size_t) n_trade_);
    size_t grown = 0, copied = 0, left = 0;
    bool failed = false;
    for (;;) {
        int64_t k = -1;
        {
            std::unique_lock<std::mutex> l(lock_m_);
            if ((copied || left) && (lock_want_.empty() || !lock_idle_.load(std::memory_order_relaxed))) {
                size_t wait = 0;
                for (const int64_t w : lock_want_) wait += mirror_stride_[(size_t) layers_[(size_t) w / ne].trade];
                std::fprintf(stderr, "strata-poly: --lock-experts: %.0f MiB of experts that left VRAM copied from the "
                                     "model file into locked RAM (%.1f GiB more in all)%s%s\n",
                             (double) copied / (1 << 20), (double) grown / (1 << 30),
                             left ? ("; " + std::to_string(left >> 20) + " MiB of them stay in the model file, with no "
                                     "more pinned memory for them").c_str() : "",
                             wait ? ("; " + std::to_string(wait >> 20) + " MiB more wait for the end of the request "
                                     "under way").c_str() : "");
                copied = left = 0;
            }
            lock_cv_.wait(l, [&] {
                return lock_quit_ || (lock_idle_.load(std::memory_order_relaxed) && !lock_want_.empty());
            });
            if (lock_quit_) return;
            k = lock_want_.back();
            lock_want_.pop_back();
        }
        {
            const Layer& L = layers_[(size_t) k / ne];
            const size_t e = (size_t) k % ne, c = (size_t) L.trade, stride = mirror_stride_[c];
            if (slots[c].empty() && !failed) {
                const size_t n = std::max<size_t>(1, (size_t(32) << 20) / stride);
                MirrorPiece m{nullptr, n * stride, mirror_registered_};
                if (m.registered) {
                    m.base = host_alloc(m.bytes);
                    if (m.base && cudaHostRegister(m.base, m.bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) !=
                                      cudaSuccess) {
                        host_free(m.base);
                        m.base = nullptr;
                    }
                } else if (cudaMallocHost((void**) &m.base, m.bytes) != cudaSuccess) {
                    m.base = nullptr;
                }
                if (m.base) {
                    lock_blocks_.push_back(m);
                    for (size_t i = n; i > 0; --i) slots[c].push_back(m.base + (i - 1) * stride);
                    grown += m.bytes;
                } else {
                    cudaGetLastError();
                    failed = true;
                }
            }
            if (slots[c].empty()) {
                left += stride;
                continue;
            }
            uint8_t* to = slots[c].back();
            slots[c].pop_back();
            const auto get = [&](uint8_t* dst, const uint8_t* src, size_t n) {
                if (!read_file(dst, src, n, true)) std::memcpy(dst, src, n);
            };
            get(to, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            get(to + L.gu_bytes, L.exp_up + e * L.u_stride, L.u_bytes);
            get(to + L.gu2(), L.exp_down + e * L.d_bytes, L.d_bytes);
            copied += stride;
            std::lock_guard<std::mutex> l(lock_m_);
            lock_done_.emplace_back(k, to);
            lock_done_n_.store((int) lock_done_.size(), std::memory_order_release);
        }
    }
#endif
}

// On the main thread, between calls, where the mirror is read: each copy lock_loop() made is its expert's slot in the
// mirror, on the card again or not (refills_mirror()), unless it got one meanwhile (a spare when it left the card
// again), and then a spare, for the next one that leaves.
void Engine::lock_settle() {
    if (lock_done_n_.load(std::memory_order_acquire) == 0) return;
    std::vector<std::pair<int64_t, uint8_t*>> done;
    {
        std::lock_guard<std::mutex> l(lock_m_);
        done.swap(lock_done_);
        lock_done_n_.store(0, std::memory_order_relaxed);
    }
    for (const auto& [k, s] : done) {
        lock_asked_[(size_t) k] = 0;
        if (!mirror_of_[(size_t) k]) mirror_of_[(size_t) k] = s;
        else mirror_spare_[(size_t) layers_[(size_t) k / (size_t) n_expert_].trade].push_back(s);
    }
}

void Engine::mirror_give(int il, int e) {
    uint8_t*& m = mirror_of_[(size_t) il * (size_t) n_expert_ + (size_t) e];
    const int c = layers_[(size_t) il].trade;
    if (m && c >= 0) mirror_spare_[(size_t) c].push_back(m);
    m = nullptr;
}

// Queued on the main stream before what takes the VRAM slot.
void Engine::mirror_keep(int il, int e, const uint8_t* vram) {
    if (mirror_of_.empty() || mirror(il, e)) return;
    uint8_t* s = mirror_take(il);
    if (!s) return lock_queue(il, e);
    ck(cudaMemcpyAsync(s, vram, layers_[(size_t) il].slot_bytes, cudaMemcpyDeviceToHost, cs(st_)), "the expert cache");
    mirror_of_[(size_t) il * (size_t) n_expert_ + (size_t) e] = s;
}

// A MoE layer for a prompt batch: the experts on the GPU, by llama.cpp's MMQ.  A row is a token and one
// of the experts it routes to.  The rows are sorted by expert, the experts of the VRAM cache first; the other
// experts pass through the staging buffer, as many at a time as it holds.  A full batch starts the copy of
// L.ahead before it knows the routing, on a stream of its own, so the upload runs under the layer's attention.
// A call that copies only the experts it routes to leaves a share of them to the CPU, which computes them while
// the others are copied: the share that had the two take about as long (Options::cpu_share).
void Engine::moe_rows(Layer& L, int il, int n) {
    if (prompt_lent_ || work_lent_)
        throw std::logic_error("a prompt's experts with their buffers lent to the expert cache");
    const int rr = n * n_used_;
    const size_t cap = (size_t) rows_ * n_used_;
    const bool ahead = opt_.stage_ahead > 0 && (n >= opt_.stage_ahead || shared_ahead_);
    const bool spare = !ahead && opt_.cpu_share != 0 && !all_cached();   // the CPU may take some (below)
    // The layer's tables.  tok and place: a row's token, and its place among that token's experts.  bounds: each
    // expert's first row.  rel: the same for the experts of a staging round, counted from the round's first row.
    // w: the weight of each place.
    int32_t* tok = h_tab_;
    int32_t* place = tok + cap;
    int32_t* bounds = place + cap;
    int32_t* rel = bounds + n_expert_ + 1;
    float* w = (float*) (rel + 2 * n_expert_ + 2);
    int32_t* cb = (int32_t*) (w + cap);   // per class: the cached experts' first rows, in the order of the slots
    // STRATA_POLY_STAGED_TILES: per size class and round, the staged experts' first rows (tb), and the same counted
    // from the round's first row (tb + n_tile * tile_row)
    int32_t* tb = cb + kClasses * ((size_t) n_expert_ + 1);
    router_scores(L, n);
    // Routed on the card, a batch hands the host its picks: the host took longer to route a batch of 1,024 tokens
    // than the card took for its experts.
    const bool on_card = route_card(il, n, d_ids_, d_wts_);
    if (on_card) {
        copy2(d_ids_, h_ids_, rr, d_wts_, w, rr, st_);
        if (spare) copy2(xn_, h_x_, n * n_embd_, nullptr, nullptr, 0, st_);
    } else if (spare) {
        copy2(rl_, h_rl_, n * n_expert_, xn_, h_x_, n * n_embd_, st_);
    } else {
        ck(cudaMemcpyAsync(h_rl_, rl_, (size_t) n * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
           "router");
    }
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");

    moe_shared(L, il, n);

    // read() takes a layer's calls one after the other, so the buffer may hold this layer's experts from the
    // call before (staged_).  The ones copied ahead do not depend on the routing and are this call's too.  A
    // later round is this call's only if it routes to the same experts, and waiting to see means nothing is
    // copied ahead meanwhile.  That pays when the card gets to the routing in under half the time the copy
    // ahead takes: most of that copy then holds the card up.
    //
    // A small call on its own copies only the experts it routes to.  With another launch's experts the batched
    // kernels would split its sums another way, so it cannot take what a call before it left.  In a read of
    // several calls it copies ahead as the others do, and the copy is theirs too.
    if (ahead) plan_ahead(L, il);
    const int n_ahead = ahead ? (int) L.ahead.size() : 0;
    // Under a freeze the staging decisions are the measured times', and those are the machine's, not the
    // request's: a frozen answer takes the staging as it finds it, which is what the first token saw.
    const bool mine = ahead && staged_il_ == il &&
                      (staged_ == L.ahead || (freeze_experts_ ? true : 2 * route_ms_ < ahead_ms_));
    const bool early = n_ahead > 0 && !mine;
    if (early) {
        const Clock::time_point t = Clock::now();
        // the buffer is free once the layer before has read it
        ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) ev_free_, 0), "expert staging");
        stage(L, L.ahead.data(), n_ahead, up_);
        ck(cudaEventRecord((cudaEvent_t) ev_ahead_, cs(up_)), "expert staging");
        if (!freeze_experts_) {   // the measured times are the machine's, not the request's: keep them out of a freeze
            ahead_ms_ = ms_since(t);
            // out already: how long before is not known until a call waits for it
            route_ms_ = cudaEventQuery((cudaEvent_t) ev_) == cudaSuccess ? 0 : ahead_ms_;
        }
    }

    const Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    if (ahead && !early && !freeze_experts_) route_ms_ = ms_since(t0);
    stats_.route_wait_ms += ms_since(t0);

    const size_t cb_row = (size_t) n_expert_ + 1;
    const int32_t* d_tok = tab_;
    const int32_t* d_place = d_tok + cap;
    const int32_t* d_bounds = d_place + cap;
    const int32_t* d_rel = d_bounds + n_expert_ + 1;
    const float* d_w = (const float*) (d_rel + 2 * n_expert_ + 2);
    const int32_t* d_cb = (const int32_t*) (d_w + cap);
    const int32_t* d_tb = d_cb + kClasses * cb_row;

    // A token's routing is a softmax over every expert and a search for the largest few.  On one thread it was
    // 0.9 s of the 2.4 s a 7,709-token prompt took with 40 layers of 256 experts, so the host's threads take a
    // stretch of tokens each.
    const int parts = on_card ? 0 : std::min(workers_->threads(), (n + kStageRows - 1) / kStageRows);
    if (on_card) std::copy(h_ids_, h_ids_ + rr, ids_.begin());
    if (parts > 0) workers_->run(parts, [&](int p) {
        float* tmp = route_tmp_.data() + (size_t) p * 2 * n_expert_;
        const int end = (int) ((int64_t) n * (p + 1) / parts);
        for (int t = (int) ((int64_t) n * p / parts); t < end; ++t)
            route(il, h_rl_ + (size_t) t * n_expert_, tok_ ? tok_[t] : -1, ids_.data() + (size_t) t * n_used_,
                  w + (size_t) t * n_used_, tmp);
    });
    count_.assign((size_t) n_expert_, 0);
    for (int i = 0; i < rr; ++i) {
        ++count_[(size_t) ids_[(size_t) i]];
        ++use_[(size_t) il * n_expert_ + (size_t) ids_[(size_t) i]];
        if (pool_) prompt_use_[(size_t) il * n_expert_ + (size_t) ids_[(size_t) i]] += 1.f;
    }
    if (pool_ && il == first_moe_layer()) prompt_rows_ += n;
    // The CPU computes some of the routed experts outside VRAM while the others are copied: the ones the fewest
    // tokens chose, whose copy costs as much as any other's and which cost the CPU about as much as their tokens.
    int n_spare = 0, n_cpu = 0;
    const int size = std::clamp((int) std::bit_width((unsigned) n) - 6, 0, 4);   // which of share_
    on_cpu_.assign((size_t) n_expert_, 0);
    if (spare) {
        spare_.clear();
        // under the RAM budget, the ones in kept_ or pool_ only: the card gets the others through the staging buffer
        const uint8_t* kept = kept_.empty() ? nullptr : kept_.data() + (size_t) il * n_expert_;
        for (int e = 0; e < n_expert_; ++e)
            if (count_[(size_t) e] > 0 && !L.slot[(size_t) e] && (!pool_ || kept[e] || pool_has(il, e)))
                spare_.push_back(e);
        std::stable_sort(spare_.begin(), spare_.end(),
                         [&](int32_t a, int32_t b) { return count_[(size_t) a] < count_[(size_t) b]; });
        n_spare = (int) spare_.size();
        const double s = opt_.cpu_share > 0 ? std::min(opt_.cpu_share, 1.f) : share_[size];
        n_cpu = (int) std::lround(s * n_spare);
        for (int j = 0; j < n_cpu; ++j) on_cpu_[(size_t) spare_[(size_t) j]] = 1;
    }
    // The cached experts that have rows, class after class and a class's in the order of the slots (chan_ has
    // every cached expert in that order); the experts already on their way; then the others that have rows.
    chan_.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (L.slot[(size_t) e]) chan_.push_back(e);
    std::sort(chan_.begin(), chan_.end(),
              [&](int32_t a, int32_t b) { return L.slot[(size_t) a] < L.slot[(size_t) b]; });
    const int n_chan = (int) chan_.size();
    group_.clear();
    for (int c = 0; c < kClasses; ++c)
        for (const int32_t e : chan_)
            if (count_[(size_t) e] > 0 && tile_class(count_[(size_t) e]) == c) group_.push_back(e);
    const int n_cached = (int) group_.size();
    group_.insert(group_.end(), L.ahead.begin(), L.ahead.begin() + n_ahead);
    sent_.assign((size_t) n_expert_, 0);
    for (int j = 0; j < n_ahead; ++j) sent_[(size_t) L.ahead[(size_t) j]] = 1;
    for (int e = 0; e < n_expert_; ++e)
        if (count_[(size_t) e] > 0 && !L.slot[(size_t) e] && !sent_[(size_t) e] && !on_cpu_[(size_t) e])
            group_.push_back(e);
    const int n_group = (int) group_.size();
    auto rows_of = [&](int g) { return count_[(size_t) group_[(size_t) g]]; };

    const size_t gu2 = L.gu2(), step = L.gu_step;   // a slot's down is gu2 in; staged experts are step apart
    // a round is what the staging buffer holds at a time; round_ has the first group of each, and the end
    const int per_round = (int) (stage_bytes_ / (step + L.d_bytes));
    round_.assign(1, n_cached);
    if (n_ahead > 0) round_.push_back(n_cached + n_ahead);
    while (round_.back() < n_group) round_.push_back(std::min(round_.back() + per_round, n_group));
    const int n_round = (int) round_.size() - 1;
    const int n_tile = (int) staged_tiles_.size() + 1;   // the size classes of a round's experts
    const size_t tile_row = (size_t) 2 * n_expert_ + 2;
    const auto tile_of = [&](int rows) {
        int t = 0;
        while (t < n_tile - 1 && rows > staged_tiles_[(size_t) t]) ++t;
        return t;
    };
    int at = 0;
    for (int g = 0; g < n_cached; ++g) {
        bounds[g] = at;
        first_[(size_t) group_[(size_t) g]] = at;
        at += rows_of(g);
    }
    bounds[n_cached] = at;
    round_row_.assign((size_t) n_round + 1, 0);
    if (n_tile == 1) {
        for (int g = n_cached; g < n_group; ++g) {
            bounds[g] = at;
            first_[(size_t) group_[(size_t) g]] = at;
            at += rows_of(g);
        }
        bounds[n_group] = at;
        for (int r = 0; r <= n_round; ++r) round_row_[(size_t) r] = bounds[round_[(size_t) r]];
    } else {
        // A round's rows lie class after class, and a class has bounds in which the other classes' experts have none
        // (as the cached experts' cb below), so that a launch's tile is as wide as its class's largest expert needs.
        // A row's product does not depend on the tile it is in, so the results are one launch's (within rounding
        // where MMQ uses stream-k).
        for (int r = 0; r < n_round; ++r) {
            const int g0 = round_[(size_t) r], m = round_[(size_t) r + 1] - g0;
            round_row_[(size_t) r] = at;
            for (int k = 0; k < n_tile; ++k) {
                int32_t* ta = tb + (size_t) k * tile_row + (size_t) (g0 - n_cached + r);
                int32_t* tr = ta + (size_t) n_tile * tile_row;
                for (int j = 0; j < m; ++j) {
                    ta[j] = at;
                    tr[j] = at - round_row_[(size_t) r];
                    if (tile_of(rows_of(g0 + j)) == k) {
                        first_[(size_t) group_[(size_t) (g0 + j)]] = at;
                        at += rows_of(g0 + j);
                    }
                }
                ta[m] = at;
                tr[m] = at - round_row_[(size_t) r];
            }
        }
        round_row_[(size_t) n_round] = at;
        bounds[n_group] = at;
    }
    const int rg = at;   // the card's rows
    host_.clear();
    const ExpertScales es = expert_scales(L, d_ex_);
    for (int i = 0; i < rr; ++i) {
        const size_t e = (size_t) ids_[(size_t) i];
        if (!L.h_s_down.empty()) w[i] *= L.h_s_down[e];   // NVFP4's down scale goes on the weight (ExpertScales)
        if (on_cpu_[e]) {   // by token, as cpu_experts() takes them, and nothing of it in the card's sum
            const float gs = es.gate ? L.h_s_gate[e] : 1.f, us = es.up ? L.h_s_up[e] : 1.f;
            if (const uint8_t* p = pool_has(il, (int) e))
                host_.push_back({p, p + L.gu_bytes, p + L.gu2(), w[i], i / n_used_, gs, us});
            else if (const uint8_t* m = mirror(il, (int) e))
                host_.push_back({m, m + L.gu_bytes, m + L.gu2(), w[i], i / n_used_, gs, us});
            else
                host_.push_back({L.exp_gate + e * L.gu_stride, L.exp_up + e * L.u_stride, L.exp_down + e * L.d_bytes,
                                 w[i], i / n_used_, gs, us});
            w[i] = 0.f;
            continue;
        }
        const int row = first_[e]++;
        tok[row] = i / n_used_;
        place[row] = i;
        h_ex_[row] = (int32_t) e;
    }
    for (int r = 0; n_tile == 1 && r < n_round; ++r)
        for (int g = round_[(size_t) r]; g <= round_[(size_t) r + 1]; ++g)
            rel[g - n_cached + r] = bounds[g] - bounds[round_[(size_t) r]];
    // A class's launch runs over every cached expert of a run of slots, and the others must have no rows in it:
    // cb has, for each class, where an expert's rows start when only that class's experts have any.
    for (int c = 0, at = 0; c < kClasses; ++c) {
        int32_t* b = cb + (size_t) c * cb_row;
        for (int j = 0; j < n_chan; ++j) {
            b[j] = at;
            const int rows = count_[(size_t) chan_[(size_t) j]];
            if (rows > 0 && tile_class(rows) == c) at += rows;
        }
        b[n_chan] = at;
    }
    const size_t n_tab =
        3 * cap + 3 * (size_t) n_expert_ + 3 + kClasses * cb_row + 2 * n_tile * tile_row * (n_tile > 1);
    const int n_ex = (es.gate || es.up) && rg > 0 ? rg : 0;
    // A read of several calls copies each layer's experts outside the cache ahead (up_), and a copy of the tables
    // would wait behind them: there a kernel reads them from pinned memory.  A call alone keeps the copy.
    if (kernel_tables_ && opt_.stage_ahead > 0 && shared_ahead_) {
        copy2(h_tab_, tab_, (int) n_tab, h_ex_, d_ex_, n_ex, st_);
    } else {
        ck(cudaMemcpyAsync(tab_, h_tab_, n_tab * sizeof(int32_t), cudaMemcpyHostToDevice, cs(st_)), "expert rows");
        if (n_ex > 0)
            ck(cudaMemcpyAsync(d_ex_, h_ex_, (size_t) n_ex * sizeof(int32_t), cudaMemcpyHostToDevice, cs(st_)),
               "expert rows");
    }

    const int rc = bounds[n_cached];   // the rows of the cached experts
    stats_.batch_cached += rc;
    stats_.batch_staged += rg - rc;

    const float limit = clamp_exp_[(size_t) il];
    const int gt = (int) L.gu_type, ut = (int) L.u_type, dt = (int) L.d_type;
    const bool guard = half_sum_guard_ && Mmq::half_sums(dt);   // the down rows keep their sums in halves
    const int64_t ff = n_ff_exp_, ff2 = 2 * ff;
    // swiglu's rows quantized for down as they are made, where down takes them as int8 (the same numbers, and
    // neither the floats between the two kernels nor a launch: a 2,048-token batch of LFM2.5 writes and reads 50 MB)
    const bool fused = !gelu_ && situ_beta_ <= 0 && !guard && Mmq::int8(dt, ff);
    // A mixed layer multiplies gate and up apart, n_ff rows each, up's into the second half of a row: its rows
    // quantized for up's type too, since quantize() writes them for the matrix type they meet.
    const int64_t gu_rows = L.mixed() ? ff : ff2;
    // Gate and up read each token's row once for every expert it chose: where the product can read rows by index,
    // the call's n rows are quantized once and each expert's rows read by token, where they were quantized once an
    // expert (rg rows), 8 times the work for Qwen3.6.
    const bool by_token = Mmq::reads_by_index(gt, n_embd_) && (!L.mixed() || Mmq::reads_by_index(ut, n_embd_));
    const int32_t* tok_of = by_token ? d_tok : nullptr;
    const int64_t q_rows = by_token ? n : rg;
    const auto up_mul = [&](const uint8_t* w, size_t stride, int m, const int32_t* bounds, int64_t most) {
        if (L.mixed())
            mmq_->mul(w + L.gu_bytes, ut, ff, n_embd_, stride, m, mqu_, bounds, iota_, q_rows, most, gu_ + ff, ff2, st_,
                      tok_of);
    };
    // expert_tiles_: the cached experts' slots as offsets from the lowest one, for a launch a class (Mmq::mul_at())
    const uint8_t* slot0 = nullptr;
    bool by_at = d_slot_at_ && n_cached > 0 && !L.mixed() && Mmq::int8(gt, n_embd_) && Mmq::int8(dt, ff);
    if (by_at) {
        for (int g = 0; g < n_cached; ++g) {
            const uint8_t* p = L.slot[(size_t) group_[(size_t) g]];
            if (!slot0 || p < slot0) slot0 = p;
        }
        for (int g = 0; g < n_cached; ++g) {
            h_slot_at_[g] = L.slot[(size_t) group_[(size_t) g]] - slot0;
            by_at = by_at && h_slot_at_[g] % 16 == 0;
        }
        // by a kernel that reads them from pinned memory, as the tables: a copy would wait behind the experts the
        // copy engine is taking to the card, and the cached experts' launches behind it
        if (by_at) copy2(h_slot_at_, d_slot_at_, 2 * n_cached, nullptr, nullptr, 0, st_);
    }
    // the cached experts lie class after class in group_: a launch for each class with rows, over all of its experts
    // (Metal's Mmq has no mul_at(): by_at is never set there)
    const auto mul_at = [&](const uint8_t* w, int g0, int t, int64_t w_rows, int64_t w_cols, int m, const void* xq,
                            const int32_t* ids, int64_t total, int most, float* dst, int64_t ld, const int32_t* from) {
#if !defined(STRATA_GPU_METAL)
        mmq_->mul_at(w, d_slot_at_ + g0, t, w_rows, w_cols, m, xq, d_bounds + g0, ids, total, most, dst, ld, st_, from);
#else
        (void) w, (void) g0, (void) t, (void) w_rows, (void) w_cols, (void) m, (void) xq, (void) ids, (void) total,
            (void) most, (void) dst, (void) ld, (void) from;
#endif
    };
    const auto each_class_at = [&](auto&& launch) {
        for (int g0 = 0; g0 < n_cached;) {
            const int c = tile_class(rows_of(g0));
            int g1 = g0, most = 0;
            for (; g1 < n_cached && tile_class(rows_of(g1)) == c; ++g1) most = std::max(most, rows_of(g1));
            launch(g0, g1 - g0, most);
            g0 = g1;
        }
    };
    int holds = -1;   // the round in the buffer when this call is done
    // The card's part.  With the CPU's beside it, the rows of its experts' places are left at nothing.
    const auto card = [&] {
        if (n_cpu > 0) ck(cudaMemsetAsync(dx_, 0, (size_t) rr * n_embd_ * sizeof(float), cs(st_)), "expert rows");
        if (rg > 0) Mmq::quantize(xn_, tok_of ? nullptr : d_tok, mq_, gt, n_embd_, n_embd_, q_rows, st_);
        if (rg > 0 && L.mixed()) Mmq::quantize(xn_, tok_of ? nullptr : d_tok, mqu_, ut, n_embd_, n_embd_, q_rows, st_);

        // The cached experts whose slots lie one behind the other are the matrices of one launch, as the experts of
        // a staging round are; a launch for each class that has rows among them.  The cache is filled a layer's
        // experts side by side, so that is the whole layer until experts trade slots (follow()).  A launch for each
        // expert was a sixth of a long prompt's time in a model with 256 experts a layer.
        auto slot_of = [&](int j) { return L.slot[(size_t) chan_[(size_t) j]]; };
        runs_.clear();
        for (int j = 0; j < n_chan;) {
            int m = 1, most[kClasses] = {}, first_c[kClasses], last_c[kClasses];
            for (int c = 0; c < kClasses; ++c) { first_c[c] = j + n_chan; last_c[c] = -1; }
            while (j + m < n_chan && slot_of(j + m) == slot_of(j + m - 1) + L.slot_bytes) ++m;
            for (int i = j; i < j + m; ++i) {
                const int rows = count_[(size_t) chan_[(size_t) i]];
                if (rows > 0) {
                    const int c = tile_class(rows);
                    most[c] = std::max(most[c], rows);
                    first_c[c] = std::min(first_c[c], i);
                    last_c[c] = std::max(last_c[c], i);
                }
            }
            for (int c = 0; c < kClasses; ++c) {
                if (most[c] <= 0) continue;
#if defined(STRATA_GPU_METAL)
                if (opt_.metal_optimize) {
                    runs_.push_back({c, first_c[c], last_c[c] - first_c[c] + 1, most[c]});
                    continue;
                }
#endif
                runs_.push_back({c, j, m, most[c]});
            }
            j += m;
        }
        if (by_at) {
            each_class_at([&](int g0, int m, int most) {
                mul_at(slot0, g0, gt, gu_rows, n_embd_, m, mq_, iota_, q_rows, most, gu_, ff2, tok_of);
            });
        } else {
            for (const Run& r : runs_) {
                mmq_->mul(slot_of(r.j), gt, gu_rows, n_embd_, L.slot_bytes, r.m, mq_, d_cb + (size_t) r.c * cb_row + r.j,
                          iota_, q_rows, r.most, gu_, ff2, st_, tok_of);
                up_mul(slot_of(r.j), L.slot_bytes, r.m, d_cb + (size_t) r.c * cb_row + r.j, r.most);
            }
        }
        if (rc > 0) {
            if (fused) {
                swiglu_clamp_i8_rows(gu_, mq2_, n_ff_exp_, rc, limit, st_, &es);
            } else {
                if (situ_beta_ > 0) situ_rows(gu_, hx_, n_ff_exp_, rc, situ_beta_, situ_lb_, st_, nullptr, &es);
                else if (gelu_) geglu_rows(gu_, hx_, n_ff_exp_, rc, st_, &es);
                else swiglu_clamp_rows(gu_, hx_, n_ff_exp_, rc, limit, st_, &es);
                if (guard) guard_half_sums(hx_, ff, n_ff_exp_, rc, hf_, st_);
                Mmq::quantize(hx_, nullptr, mq2_, dt, ff, ff, rc, st_);
            }
        }
        if (by_at) {
            each_class_at([&](int g0, int m, int most) {
                mul_at(slot0 + gu2, g0, dt, n_embd_, ff, m, mq2_, d_place, rc, most, dx_, n_embd_, nullptr);
            });
        } else {
            for (const Run& r : runs_)
                mmq_->mul(slot_of(r.j) + gu2, dt, n_embd_, ff, L.slot_bytes, r.m, mq2_,
                          d_cb + (size_t) r.c * cb_row + r.j, d_place, rc, r.most, dx_, n_embd_, st_);
        }
        if (guard && rc > 0) scale_rows(dx_, n_embd_, n_embd_, rc, d_place, hf_, false, st_);

        // The round the buffer holds goes first and is not copied.  If it holds none of this call's, the experts
        // that go ahead are copied last, so the next call finds them.  The order does not change a round's launches.
        int kept = -1;
        for (int r = 0; r < n_round && mine && kept < 0; ++r) {
            const auto first = group_.begin() + round_[(size_t) r], last = group_.begin() + round_[(size_t) r + 1];
            if ((size_t) (last - first) == staged_.size() && std::equal(first, last, staged_.begin())) kept = r;
        }
        const int from = mine && kept < 0 ? 1 : 0;
        holds = early ? 0 : kept;
        for (int turn = 0; turn < n_round; ++turn) {
            const int r = kept < 0 ? (turn + from) % n_round : turn == 0 ? kept : turn - (turn <= kept ? 1 : 0);
            const int g0 = round_[(size_t) r], m = round_[(size_t) r + 1] - g0;
            const size_t down_at = pad((size_t) m * step);
            const int r0 = round_row_[(size_t) r], rows = round_row_[(size_t) r + 1] - r0;
            if (r == kept) {
                // the buffer has it
            } else if (r == 0 && early) {
                ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_ahead_, 0), "expert staging");
            } else if (rows > 0) {
                stage(L, group_.data() + g0, m, st_, n_cpu > 0);
                holds = r;
            }
            if (rows == 0) continue;   // nothing routed to the experts that go ahead
            int most = 0;
            for (int g = g0; g < g0 + m; ++g) most = std::max(most, rows_of(g));
            // with size classes, a launch for each class that has rows in the round, as wide a tile as its experts need
            const auto each_class = [&](auto&& launch) {
                if (n_tile == 1) return launch(0, most);
                for (int k = 0; k < n_tile; ++k) {
                    int top = 0;
                    for (int g = g0; g < g0 + m; ++g)
                        if (tile_of(rows_of(g)) == k) top = std::max(top, rows_of(g));
                    if (top > 0) launch(k, top);
                }
            };
            each_class([&](int k, int top) {
                const int32_t* b = n_tile == 1 ? d_bounds + g0 : d_tb + (size_t) k * tile_row + (size_t) (g0 - n_cached + r);
                mmq_->mul(stage_, gt, gu_rows, n_embd_, step, m, mq_, b, iota_, q_rows, top, gu_, ff2, st_, tok_of);
                up_mul(stage_, step, m, b, top);
            });
            ExpertScales er = es;
            if (er.ids) er.ids += r0;
            if (fused) {
                swiglu_clamp_i8_rows(gu_ + (size_t) r0 * ff2, mq2_, n_ff_exp_, rows, limit, st_, &er);
            } else {
                if (situ_beta_ > 0)
                    situ_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, situ_beta_, situ_lb_, st_, nullptr,
                              &er);
                else if (gelu_) geglu_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, st_, &er);
                else swiglu_clamp_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, limit, st_, &er);
                if (guard) guard_half_sums(hx_ + (size_t) r0 * ff, ff, n_ff_exp_, rows, hf_ + r0, st_);
                Mmq::quantize(hx_ + (size_t) r0 * ff, nullptr, mq2_, dt, ff, ff, rows, st_);
            }
            each_class([&](int k, int top) {
                const int32_t* b = (n_tile == 1 ? d_rel + (g0 - n_cached + r)
                                                : d_tb + (size_t) (n_tile + k) * tile_row + (size_t) (g0 - n_cached + r));
                mmq_->mul(stage_ + down_at, dt, n_embd_, ff, L.d_bytes, m, mq2_, b, d_place + r0, rows, top, dx_, n_embd_,
                          st_);
            });
            if (guard) scale_rows(dx_, n_embd_, n_embd_, rows, d_place + r0, hf_ + r0, false, st_);
        }
        ck(cudaEventRecord((cudaEvent_t) ev_free_, cs(st_)), "expert staging");
    };
    if (n_cpu == 0) {
        card();
    } else {   // the card's part on a thread of its own, and the CPU's on the workers
        if (!stagers_) stagers_ = std::make_unique<Workers>(4);   // one thread copied slower than the bus
        Clock::time_point done;
        Behind behind;
        const Clock::time_point start = Clock::now();
        behind.go([&] {
            cudaSetDevice(device_);   // a thread starts on card 0
            card();
            ck(cudaStreamSynchronize(cs(st_)), "expert staging");
            done = Clock::now();
        });
        cpu_experts(*workers_, host_.data(), (int) host_.size(), n, gt, ut, dt, n_embd_, n_ff_exp_, h_x_, limit, gelu_,
                    h_cpu_, nullptr, nullptr, nullptr, situ_beta_, situ_lb_);
        const double cpu = ms_since(start);
        behind.wait();
        const double gpu = std::chrono::duration<double, std::milli>(done - start).count();
        stats_.cpu_ms += cpu;
        stats_.cpu_experts += (int64_t) host_.size();
        // the share that would have had both done at once, if each one's time grows with its share
        if (opt_.cpu_share < 0 && n_cpu < n_spare && !freeze_experts_) {
            const double s = (double) n_cpu / n_spare, c = cpu / s, g = gpu / (1 - s);
            share_[size] = std::clamp(0.5 * share_[size] + 0.5 * g / (c + g), 0.05, 0.95);
        }
    }
    moe_sum(dx_, d_w, fo_, n_embd_, n_used_, n, st_);
    if (n_cpu > 0) moe_add(n);
    if (ahead && holds >= 0) {
        staged_.assign(group_.begin() + round_[(size_t) holds], group_.begin() + round_[(size_t) holds + 1]);
        staged_il_ = il;
    }
}

// The rows a recorded attention launch is sized for, for n tokens from pos(): the power of two from 1024 that
// holds their rows and the ones before them, or the context if that is less.  room_key() numbers the sizes.
int Engine::room(int n) const { return (int) std::min<int64_t>(int64_t(1024) << room_key(n), opt_.max_context); }

int Engine::room_key(int n) const {
    int key = 0;
    for (int64_t r = 1024; r < pos_ + n; r *= 2) ++key;
    return key;
}

// Records what `work` hands to the stream, as a graph that is launched with one call from then on.
template <class F>
void* Engine::record(F&& work) {
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    recording_ = recorded_ = true;
    try {
        work();
    } catch (...) {
        recording_ = false;
        if (cudaStreamEndCapture(cs(st_), &g) == cudaSuccess && g) cudaGraphDestroy(g);
        cudaGetLastError();   // a capture the failure broke does not end cleanly: the work's exception says why
        throw;
    }
    recording_ = false;
    ck(cudaStreamEndCapture(cs(st_), &g), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    cudaError_t made = cudaGraphInstantiateWithFlags(&exec, g, 0);
    if (made == cudaErrorMemoryAllocation && !step_used_.empty()) {
        // Inkling 2's step recordings (step_replay(): only where tail_by_rows()) give their place to this one: run()'s
        // and step()'s other recordings.  Every family else has none, and fails here as before.
        cudaGetLastError();
        trim_step_graphs(0, !in_step_);
        exec = nullptr;
        made = cudaGraphInstantiateWithFlags(&exec, g, 0);
    }
    cudaGraphDestroy(g);
    ck(made, "cudaGraphInstantiate");
    graphs_.push_back(exec);
    return exec;
}

// step()'s recording `in[key]` launched, made first if it is not there; false when `work` ran unrecorded instead.  A
// recording that fails for memory has already run `work` on the host once (its launches captured, nothing run on the
// card), so `work` runs again unrecorded: a step's host side is the same each time it is recorded or run (layer_head()
// and layer_tail() keep no host state across a call: moe_left_ is never set in step(), whose whole_rec_ is false, and
// next_layer() brings pass_ back to 0).  STRATA_POLY_STEP_RECORD_FAIL=N (tests) fails every Nth recording that way.
bool Engine::step_replay(std::map<std::vector<int>, void*>& in, const std::vector<int>& key,
                         const std::function<void()>& work) {
    const auto it = in.find(key);
    void* g = it != in.end() ? it->second : nullptr;
    if (!g && step_no_ >= step_rec_retry_) {
        g = record_step(work);
        if (g) in[key] = g;
    }
    if (!g) {
        work();
        ++step_unrecorded_;
        return false;
    }
    ck(cudaGraphLaunch((cudaGraphExec_t) g, cs(st_)), "layer graph");
    step_used_[g] = step_no_;
    return true;
}

// record() for step_replay(), with the step recordings held to a count (step_cap_).  The count, not the VRAM they
// seem to take: cudaGraphExecDestroy() gives the driver's memory back to no cudaMemGetInfo(), and the recordings
// made after a drop take the memory it kept, so a fixed number held keeps the VRAM flat where a sum of what
// cudaMemGetInfo() showed would not.  A recording the card has no memory for drops every step recording but this
// step's and tries again; failing still, it is not made (null), and no step records for the next 256 steps.
void* Engine::record_step(const std::function<void()>& work) {
    if (step_cap_ == 0) {
        // the cap: a recording is 220-250 KiB (an Inkling 2 layer's head or tail, RTX 2070 and 2080 Ti); 4 a MiB of a
        // 64th of the card, 256 to 1,024 recordings (512, 128 MiB, on an 8 GB card), or STRATA_POLY_STEP_GRAPHS
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        step_cap_ = std::clamp<size_t>((total_b >> 26) * 4, 256, 1024);
        if (const char* v = std::getenv("STRATA_POLY_STEP_GRAPHS"); v && std::atoi(v) > 0) step_cap_ = (size_t) std::atoi(v);
    }
    static const int fail_every = [] {
        const char* v = std::getenv("STRATA_POLY_STEP_RECORD_FAIL");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    recording_ = recorded_ = true;
    try {
        work();
    } catch (...) {
        recording_ = false;
        if (cudaStreamEndCapture(cs(st_), &g) == cudaSuccess && g) cudaGraphDestroy(g);
        cudaGetLastError();
        throw;
    }
    recording_ = false;
    ck(cudaStreamEndCapture(cs(st_), &g), "cudaStreamEndCapture");
    const bool forced = fail_every > 0 && ++step_tries_ % (uint64_t) fail_every == 0;
    size_t f0 = 0, f1 = 0, total = 0;
    ck(cudaMemGetInfo(&f0, &total), "cudaMemGetInfo");
    cudaGraphExec_t exec = nullptr;
    cudaError_t made = forced ? cudaErrorMemoryAllocation : cudaGraphInstantiateWithFlags(&exec, g, 0);
    if (made == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        trim_step_graphs(0, false);
        ck(cudaMemGetInfo(&f0, &total), "cudaMemGetInfo");
        exec = nullptr;
        made = forced ? cudaErrorMemoryAllocation : cudaGraphInstantiateWithFlags(&exec, g, 0);
    }
    cudaGraphDestroy(g);
    if (made == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        step_rec_retry_ = step_no_ + (forced ? 1 : 256);
        if (!forced)
            std::fprintf(stderr, "strata-poly: no VRAM for a step's recording: the steps run unrecorded for a while\n");
        return nullptr;
    }
    ck(made, "cudaGraphInstantiate");
    ck(cudaMemGetInfo(&f1, &total), "cudaMemGetInfo");
    if (step_dropped_ == 0) step_bytes_ += f0 > f1 ? f0 - f1 : 0;   // (for the log: right only until the first drop)
    graphs_.push_back(exec);
    step_used_[exec] = step_no_;
    ++step_made_;
    if (step_used_.size() > step_cap_) trim_step_graphs(step_cap_ / 4 * 3, false);
    if (step_dropped_ == 0 && step_used_.size() >= 2 * std::max<size_t>(step_logged_, 128)) {   // the size, for the cap
        step_logged_ = step_used_.size();
        std::fprintf(stderr, "strata-poly: step recordings: %zu held, %.1f MiB (%.0f KiB each), cap %zu\n",
                     step_used_.size(), step_bytes_ / 1048576.0, step_bytes_ / 1024.0 / (double) step_used_.size(),
                     step_cap_);
    }
    return exec;
}

// The step recordings launched longest ago go until `keep` are held: never this step's own (`all`: every one, from
// outside step(), where none is under way).  Those are done on the card: step() waits for the step before it first.
void Engine::trim_step_graphs(size_t keep, bool all) {
    if (step_used_.size() <= keep) return;
    std::vector<std::pair<uint64_t, void*>> old;
    for (const auto& [g, used] : step_used_)
        if (all || used != step_no_) old.push_back({used, g});
    std::sort(old.begin(), old.end());
    std::set<void*> gone;
    for (const auto& [used, g] : old) {
        if (step_used_.size() - gone.size() <= keep) break;
        gone.insert(g);
    }
    if (gone.empty()) return;
    const auto drop = [&](std::map<std::vector<int>, void*>& in) {
        for (auto it = in.begin(); it != in.end();) it = gone.count(it->second) ? in.erase(it) : std::next(it);
    };
    for (Layer& L : layers_) {
        drop(L.rows_heads);
        drop(L.rows_tails);
    }
    drop(step_passes_);
    graphs_.erase(std::remove_if(graphs_.begin(), graphs_.end(), [&](void* g) { return gone.count(g) > 0; }),
                  graphs_.end());
    for (void* g : gone) {
        ck(cudaGraphExecDestroy((cudaGraphExec_t) g), "a step's recording");
        step_used_.erase(g);
    }
    step_dropped_ += gone.size();
    std::fprintf(stderr, "strata-poly: step recordings: dropped %zu, %zu held (cap %zu); %zu made, %zu dropped, %zu runs "
                         "unrecorded in all\n",
                 gone.size(), step_used_.size(), step_cap_, step_made_, step_dropped_, step_unrecorded_);
}

// n tokens from pos() through every layer.  The sequence is not longer for it: the caller says by how much.
void Engine::run(const int32_t* tokens, int n) {
    if (n < 1 || n > rows_) throw std::logic_error("eval takes 1 to max_rows() tokens");
    lock_idle_.store(false, std::memory_order_relaxed);
    if (n >= kStageRows) take_prompt();
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");
    cover(pos_ + n);

    // h_r_ and h_cpu_ may still be the source of the previous call's copies
    ck(cudaStreamSynchronize(cs(st_)), "eval");
    // Greedy answers have to be a function of the request alone: which experts take the GPU path and which the
    // CPU one decides a layer's sums to the last bit (the two paths add in different orders), so the cache's
    // state at the first layer may not depend on whether an upload happened to have finished.  Every expert on
    // its way arrives here, and the followers' arrivals are claimed, before the first token reads the cache.
    // The cost is the upload a window still waits for, a millisecond or so of a token: under 5% of one.
    // STRATA_FREEZE_EXPERTS holds the split still for the whole request instead: the request's first token
    // claims the arrivals and asks for the trades, and the tokens after it see the cache as it was then.
    // follow() above takes the arrivals it finds, so the freeze only has to keep the later ones from asking.
    // lock_loop()'s copies are not waited for: until one is the mirror's, its expert is read from the mapping, and a
    // card_far_ call that routes to it goes to the host.
    lock_settle();
    if (n_swap_ > 0) {
        settle_swaps();
        follow(n < kStageRows ? n : 0);
    }
    tok_ = tokens;
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    if (pass_)   // a later card: the card before it left the rows
        std::memcpy(h_r_, pass_ + (size_t) (pos_ - pass_at_) * r_row, (size_t) n * r_row * sizeof(float));
    else
        embed_rows(tokens, pos_, n, h_r_);
    if (n < kStageRows) copy2(h_r_, r_, n * (int) r_row, nullptr, nullptr, 0, st_);
    else ck(cudaMemcpyAsync(r_, h_r_, (size_t) n * r_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");
    carry_in(pos_ - pass_at_, n);

    // A token alone, or a window of guessed ones, hands the card the same launches every time, some thirty a
    // layer up to the router, and they cost this thread more than the card.  They are recorded once, a graph a
    // layer and a number of tokens, and replayed with one call.  Only for the layers that can be (Layer::fixed).
    // What a launch needs of the position it reads at at_; an attention over the rows so far is recorded again
    // when the rows outgrow what it was sized for (head_key()).  A prompt's few tokens are not a window: they
    // leave no snapshots of the recurrent state, so their launches are others.
    // (never with the scratch in a prompt batch's layout: the recordings hold the few tokens' buffers)
    bool replay = opt_.graphs && n <= opt_.window && (windowed_ || block_pass_ || n == 1) && !opt_.dump &&
                  !prompt_scratch_;
    for (int t = 0; t < n && replay; ++t) replay = !image_at(pos_ + t);
    const auto replayed = [&](std::vector<void*>& made, size_t key, auto&& work) {
        if (made.size() <= key) made.resize(key + 1, nullptr);
        if (!made[key]) made[key] = record(work);
        ck(cudaGraphLaunch((cudaGraphExec_t) made[key], cs(st_)), "layer graph");
    };
    if (replay) {   // a pair a token: a window's tokens may go one by one, each from its own
        for (int t = 0; t < n; ++t) {
            h_at_[2 * t] = pos_ + t;
            h_at_[2 * t + 1] = turn(pos_ + t);
        }
        copy2(h_at_, at_, n * 2 * (int) (sizeof(long long) / 4), nullptr, nullptr, 0, st_);
    }
    // A layer's launches around its routed experts.  up(): to its router's copy, with the shared expert behind it.
    // tail(): what the layer does with the experts' sum.  `rec` numbers the recording of m tokens' launches: a
    // call's tokens together are [0, window) by their number, and a half of a window comes after those, by the
    // window's tokens and which half it is.  Which tokens leave their state is the window's, so its halves' too.
    const size_t wn = (size_t) opt_.window, recs = 3 * wn + (size_t) opt_.batch + 1;   // step()'s rows after
    const auto up = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        prefetch_layer(il, m);
        const size_t head = (size_t) head_key(il, m) * recs + rec;
        if (replay && L.fixed) replayed(L.heads, head, [&] { layer_head(L, il, m); });
        else layer_head(L, il, m);
        if (!L.moe || batched(L, m)) return;
        ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
        if (replay && shared_recorded()) replayed(L.shared, rec, [&] { moe_shared(L, il, m); });
        else moe_shared(L, il, m);
    };
    const auto tail = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        dbg("ffn", il, fo_, n_embd_);
        if (replay && L.fixed) replayed(L.tail, (size_t) tail_key(il, m) * recs + rec, [&] { layer_tail(L, il, m); });
        else layer_tail(L, il, m);
    };

    // A window's tokens, with experts on the CPU, go through the layers together or in two halves, one behind
    // the other (halves_, which window() sets).  Together, the card's part of a layer and the CPU's take turns:
    // the card waits out the CPU's experts and the CPU the card's layer.  In halves each has work while the
    // other is busy: this thread does nothing but route a half and compute its experts, half after half, and
    // while it does, one of the experts' threads hands the card the other half: its experts' sum, the rest of
    // its layer and the next layer up to the router.  That half's scores are there when its turn comes again.
    //
    // The card's part of a layer takes about as long for two tokens as for one: it reads the layer's dense
    // weights once for all of them, and its launches are the same.  So a window is two halves and not a part a
    // token, which had the card read every layer once for each token.  Halves have the card read it twice where
    // the tokens together have it read once, and that is what the choice between the two turns on.
    //
    // A half's launches are the ones its tokens have in a call of their own, at their rows of the buffers
    // (shift_rows()), so each token comes out as it would alone.
    if (n > 1 && windowed_ && !card_routes() && halves_) {
        const int first[3] = {0, (n + 1) / 2, n};   // half k is the tokens [first[k], first[k + 1])
        const unsigned bits = snap_bits_;           // the window's tokens that leave the recurrent state
        int at = 0;
        // The buffers at half k's rows, and its share of the window's states.  Returns the half's tokens.
        const auto half = [&](int k) {
            shift_rows(first[k] - at);
            at = first[k];
            const int m = first[k + 1] - at;
            snap_bits_ = bits >> at & ((1u << m) - 1u);
            snap_at_ = snap_ + size_t(std::popcount(bits & ((1u << at) - 1u))) * rec_n_;
            ev_ = ev_row_[k];
            return m;
        };
        // Half k on from layer il, whose experts it has (-1: from its start), to the next router's copy.
        const auto go_on = [&](int k, int il, bool add) {
            const int m = half(k);
            const size_t rec = wn + 2 * (size_t) (n - 1) + (size_t) k;
            if (il >= 0) {
                if (add) moe_add(m);
                tail(il, m, rec);
            }
            for (int j = il < 0 ? lo_ : il + 1; j < hi_; ++j) {
                up(j, m, rec);
                if (layers_[(size_t) j].moe) break;
            }
        };
        const auto back = [&] {
            shift_rows(-at);
            at = 0;
            ev_ = ev_row_[0];
            snap_bits_ = bits;
            snap_at_ = snap_;
        };
        struct {
            int k = -1, il = -1;
            bool add = false;
        } late;   // the half whose experts are computed and that has not gone on
        std::exception_ptr failed;
        try {
            for (int k = 0; k < 2; ++k) go_on(k, -1, false);
            for (int il = lo_; il < hi_; ++il) {
                Layer& L = layers_[(size_t) il];
                if (!L.moe) continue;
                for (int k = 0; k < 2; ++k) {
                    // by another thread while this one computes: it must leave the rows as it found them
                    const std::function<void()> other = [&] {
                        try {
                            if (split_) cudaSetDevice(device_);
                            go_on(late.k, late.il, late.add);
                            half(k);
                        } catch (...) {
                            failed = std::current_exception();
                        }
                    };
                    const int m = half(k);
                    const bool add = moe(L, il, m, late.k >= 0 ? &other : nullptr);
                    if (failed) std::rethrow_exception(failed);
                    late = {k, il, add};
                }
            }
            if (late.k >= 0) go_on(late.k, late.il, late.add);
        } catch (...) {
            back();
            throw;
        }
        back();
        return;
    }
    // The routed experts on the card alone, when every one of them is in VRAM: recorded too, once a layer has
    // shown it routes there (Layer::card).  With the doorbell (bells()) a token's misses go to this thread's
    // workers from inside the launches, which bell_serve() answers once they are all handed to the card.
    bell_ = bell_on(n);
    bell_counts_ |= bell_;
    struct BellOff {
        bool& on;
        ~BellOff() { on = false; }
    } bell_off{bell_};
    const bool card = n < kStageRows && card_moe(n);
    const auto on_card = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        if (!card) return false;
        if (replay && L.card == 1) replayed(L.card_moe, rec, [&] { moe_card(L, il, m); });
        else if (!moe_card(L, il, m)) return false;
        stats_.gpu_experts += m * n_used_;
        return true;
    };
    // Once every layer has shown that, a call's layers are one recording: on an RTX 5090 the card waited 2 to 4
    // microseconds from one layer's recording to the next, four of them a layer.
    bool whole = card && replay;
    for (int il = lo_; il < hi_ && whole; ++il) {
        const Layer& L = layers_[(size_t) il];
        whole = L.fixed && (!L.moe || L.card == 1);
    }
    // A block of guesses is recorded whole or not at all: a layer's own recordings are a window's, of other kernels.
    if (block_pass_ && !whole) replay = false;
    if (whole) {
        std::vector<int> key{block_pass_ ? -n : n, bell_ ? 1 : 0};
        for (int il = lo_; il < hi_; ++il) key.push_back(head_key(il, n));
        // Two recordings where the family asks for it: the card starts on the first few layers while this thread
        // still hands it the rest (a launch takes this thread about 0.1 us a kernel).
        const int first = whole_first_layers();
        int il = lo_;
        const auto layers = [&](int count) {
            return [&, count] {
                struct Whole {
                    bool& on;
                    bool pdl;
                    ~Whole() { on = false, whole_pdl_scope() = pdl; }
                } whole_rec{whole_rec_, whole_pdl_scope()};
                whole_rec_ = true;
                whole_pdl_scope() = whole_pdl();
                for (int done = 0; il < hi_ && done < count; il = next_layer(il), ++done) {
                    Layer& L = layers_[(size_t) il];
                    layer_head(L, il, n);
                    if (!L.moe) continue;
                    moe_shared(L, il, n);
                    moe_card(L, il, n);
                    layer_tail(L, il, n);
                }
            };
        };
        const auto runs_done = [&] {   // a run of a layer each (a looped core's layers run several times a token)
            for (int at = lo_; at < hi_; at = next_layer(at)) stats_.gpu_experts += layers_[(size_t) at].moe ? n * n_used_ : 0;
        };
#if defined(STRATA_GPU_METAL)
        // Metal encodes a replay into the stream's command buffer, and the card starts on it when it is committed: in
        // recordings of `first` layer runs each, committed one after the other, the card runs one while this thread
        // encodes the next.  Two recordings left the card idle while the second, 44 of Inkling 2's 48 runs, was
        // encoded, and a token took 1.4 times as long as without recordings.
        if (first > 0) {
            const auto part = [&](int c) {
                std::vector<int> k = key;
                k.push_back(INT_MIN + c);
                return k;
            };
            if (!passes_.count(part(0)))
                for (int c = 0; il < hi_; ++c) passes_[part(c)] = record(layers(first));
            for (int c = 0;; ++c) {
                const auto it = passes_.find(part(c));
                if (it == passes_.end()) break;
                ck(cudaGraphLaunch((cudaGraphExec_t) it->second, cs(st_)), "layer graph");
                ck(gpu::metal::submit(static_cast<gpu::Stream>(st_)), "layer graph");
            }
            runs_done();
            return;
        }
#endif
        std::vector<int> rest_key = key;
        rest_key.push_back(INT_MIN);
        void*& pass = passes_[key];
        void** rest = first > 0 ? &passes_[rest_key] : nullptr;
        if (!pass || (rest && !*rest)) {
            pass = record(layers(rest ? first : INT_MAX));
            if (rest) *rest = record(layers(INT_MAX));
        }
        ck(cudaGraphLaunch((cudaGraphExec_t) pass, cs(st_)), "layer graph");
        if (rest) ck(cudaGraphLaunch((cudaGraphExec_t) *rest, cs(st_)), "layer graph");
        runs_done();
        if (bell_) bell_serve();
        return;
    }
    for (int il = lo_; il < hi_; il = next_layer(il)) {
        Layer& L = layers_[(size_t) il];
        up(il, n, (size_t) n - 1);
        if (!L.moe) continue;
        if (batched(L, n)) {
            moe_rows(L, il, n);
        } else if (!on_card(il, n, (size_t) n - 1)) {
            if (moe(L, il, n)) moe_add(n);
        } else if (bell_) {
            // Layer by layer (the first tokens, before the whole pass is recorded): the ring is answered here, since
            // what this thread does next (a recording, a moe() that waits for the router) may wait for the card.
            bell_serve();
        }
        tail(il, n, (size_t) n - 1);
    }
}

// A row a sequence.  A layer's head is the sequence's own: its row of the buffers (shift_rows()) with the sequence
// bound, recorded and replayed by sequence and row (the sequence's Layer::heads).  The router's copy, the experts
// and the layer's tail take the rows together, as a call of n tokens does: an expert two sequences route to is
// read once for both.  The draft block is not run: a sequence that guesses again later finds its block's rows of
// these tokens missing, which makes its guesses worse and its tokens no different.  On several cards each takes the
// rows from the one before it, a row a sequence in the same order (pass_ from its start), and the last one gives the
// logits.
void Engine::step(const int* seqs, const int32_t* tokens, int n) { step(seqs, tokens, n, nullptr); }

bool Engine::steps_guesses() const {
    return opt_.batch_mtp && draft_ && step_h_ && !block_guess_ && !next_ && !pass_ && batch_heads() &&
           (rec_n_ == 0 || (rows_snaps() && step_snap_));
}

int Engine::step_rows_alone() const { return std::min(rows_, kStageRows - 1); }

void Engine::step(const int* seqs, const int32_t* tokens, int n, const int32_t* guesses) {
    // step_picks() asks this step only: taken first, so a step that throws (KvFull below) leaves no ask for the next
    const bool picks = std::exchange(step_picks_, false) && d_picks_;
    ++step_no_;   // (step_replay()'s clock)
    struct InStep {
        bool& on;
        ~InStep() { on = false; }
    } in_step{in_step_ = true};
    if (n < 1 || n > std::min(rows_, sequences())) throw std::logic_error("step takes 1 to sequences() tokens");
    if (guesses && !steps_guesses()) throw std::logic_error("step: guesses need --batch-mtp, a draft block and one card");
    if (!step_seqs_.empty()) throw std::logic_error("step: the rows of the step before are not kept yet (keep_step())");
    OnCard here(device_, split_);
    end_read();
    const int home = bound_;
    const auto use = [&](int s) { hold(s); };
    // the rows: each sequence's token, then its guess when it has one
    std::vector<int> rseq;
    std::vector<int32_t> rtok;
    std::vector<int64_t> at;
    std::vector<float*> rsnap;
    for (int i = 0; i < n; ++i) {
        if (seqs[i] < 0 || seqs[i] >= sequences() || std::count(seqs, seqs + i, seqs[i]))
            throw std::logic_error("step: no such sequence, or one twice");
        use(seqs[i]);
        const int m = guesses && guesses[i] >= 0 ? 2 : 1;
        if (pos_ + m > opt_.max_context)
            throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");
        cover(pos_ + m);
        for (int j = 0; j < m; ++j) {
            rseq.push_back(seqs[i]);
            rtok.push_back(j ? guesses[i] : tokens[i]);
            at.push_back(pos_ + j);
            rsnap.push_back(m == 2 && j == 0 && step_snap_ ? step_snap_ + (size_t) seqs[i] * rec_n_ : nullptr);
        }
    }
    const int R = (int) rseq.size();
    if (R > rows_) throw std::logic_error("step: more rows than a call takes");
    picks_ = picks;

    ck(cudaStreamSynchronize(cs(st_)), "step");
    if (n_swap_ > 0) follow(R);
    tok_ = rtok.data();
    windowed_ = false;
    snap_bits_ = 0;
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    const float scale = embed_scale();
    if (pass_) std::memcpy(h_r_, pass_, (size_t) R * r_row * sizeof(float));   // a later card
    for (int i = 0; i < R; ++i) {
        use(rseq[(size_t) i]);
        float* r = h_r_ + (size_t) i * r_row;
        if (!pass_) {
            if (!input(rtok[(size_t) i], at[(size_t) i], r) && scale != 1.f)
                for (size_t k = 0; k < e_row; ++k) r[k] *= scale;
            for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
        }
        h_at_[2 * i] = at[(size_t) i];
        h_at_[2 * i + 1] = turn(at[(size_t) i]);
    }
    copy2(h_r_, r_, R * (int) r_row, nullptr, nullptr, 0, st_);
    carry_in(0, R);
    copy2(h_at_, at_, R * 2 * (int) (sizeof(long long) / 4), nullptr, nullptr, 0, st_);

    const size_t wn = (size_t) opt_.window, recs = 3 * wn + (size_t) opt_.batch + 1, rec = (size_t) R - 1;
    const bool replay = opt_.graphs && !opt_.dump && !prompt_scratch_;
    const auto replayed = [&](std::vector<void*>& made, size_t key, auto&& work) {
        if (made.size() <= key) made.resize(key + 1, nullptr);
        if (!made[key]) made[key] = record(work);
        ck(cudaGraphLaunch((cudaGraphExec_t) made[key], cs(st_)), "layer graph");
    };
    // A step of one row rings the doorbell as run() rings it for a token (bell_on(): card_moe() routes such a row on
    // the card for it): its misses go to this thread's workers from inside the launches, which bell_serve() answers
    // once they are handed to the card.  Without the ring, moe_card() would read the mirror's table that card_moe()
    // did not fill for it.
    bell_ = bell_on(R);
    bell_counts_ |= bell_;
    struct BellOff {
        bool& on;
        ~BellOff() { on = false; }
    } bell_off{bell_};
    const bool card = R < kStageRows && card_moe(R);
    // a token alone is routed on the card (the doorbell, bell_on()), so the rows the host takes are too (below)
    // (card_moe(1)'s test without its side effects: the doorbell, the card's slot table, no --expert-gpu; a layer that
    // card_moe() found it cannot route, L.card == 0, is routed on the host alone too: below)
    bool picks_like_alone = h_wts_ && bell_on(1) && d_slots_ && n_used_ <= card_rows() && !opt_.dump && R > 1 &&
                            !guesses && opt_.cache_bias <= 0.f;
#if !defined(STRATA_GPU_METAL)
    picks_like_alone = picks_like_alone && !ecard_;
#endif
    // an architecture that takes the rows in one head: their sequences and positions while it does
    std::vector<Row> rows;
    if (batch_heads())
        for (int i = 0; i < R; ++i) rows.push_back({rseq[(size_t) i], at[(size_t) i], rsnap[(size_t) i]});
    struct Clear {
        const Row*& p;
        ~Clear() { p = nullptr; }
    } clear{rows_of_};
    std::vector<int> key;
    // The output of rows [r0, r0 + m), whose rows of the buffers are at xn_ (shift_rows()), into their logits
    const auto logits_rows = [&](int r0, int m) {
        float* lg = wlogits_ + (size_t) r0 * n_vocab_;
        output_rows(0, m, xn_);
        lin(output_, xn_, lg, m);
        finish_logits(lg, m);
        if (picks) {
            likeliest_rows(lg, n_vocab_, m, d_picks_ + r0, st_);
            ck(cudaMemcpyAsync(h_picks_ + r0, d_picks_ + r0, (size_t) m * sizeof(Likeliest), cudaMemcpyDeviceToHost,
                               cs(st_)),
               "logits");
        } else {
            ck(cudaMemcpyAsync(h_wlogits_ + (size_t) r0 * n_vocab_, lg, (size_t) m * n_vocab_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cs(st_)),
               "logits");
        }
        // with guesses: each row's output for its sequence's draft block, in its rows after the first, as window()
        // leaves them (draft_rows())
        if (guesses)
            for (int i = r0, j = 0; i < r0 + m; ++i, ++j) {
                if (i > r0 && rseq[(size_t) i] != rseq[(size_t) i - 1]) j = 0;
                ck(cudaMemcpyAsync(seq_state(rseq[(size_t) i], hn_) + (size_t) (1 + j) * e_row,
                                   xn_ + (size_t) (i - r0) * e_row, e_row * sizeof(float), cudaMemcpyDeviceToDevice,
                                   cs(st_)),
                   "step");
            }
    };
    bool out_done[2] = {false, false};   // the halves whose logits are on their way
    // Once every layer has shown it routes on the card, the step's layers and its logits are one recording, as
    // run()'s are: the card waited 2 to 4 microseconds from one recording to the next, four of them a layer, and
    // after the last for the host's launch of the logits.  The same launches as the layer by layer replays.
    bool whole = replay && card && batch_heads() && !next_;
    for (int il = lo_; il < hi_ && whole; ++il) {
        const Layer& L = layers_[(size_t) il];
        whole = L.fixed && (!L.moe || (L.card == 1 && !batched(L, R)));
    }
    if (whole) {
        const int64_t keep = pos_;
        std::vector<int> wkey{R, picks ? 1 : 0, bell_ ? 1 : 0};
        for (int i = 0; i < R; ++i) {
            pos_ = at[(size_t) i];
            wkey.push_back(rseq[(size_t) i]);
            wkey.push_back(rsnap[(size_t) i] ? 1 : 0);
            wkey.push_back(guesses ? 1 : 0);
            for (int il = lo_; il < hi_; ++il) wkey.push_back(head_key(il, 1));
        }
        pos_ = keep;
        const auto whole_work = [&] {
                // the layers' runs in next_layer()'s order (a looped core's layers run more than once a token)
                for (int il = lo_; il < hi_; il = next_layer(il)) {
                    Layer& L = layers_[(size_t) il];
                    rows_of_ = rows.data();
                    n_rows_of_ = R;
                    layer_head(L, il, R);
                    if (!tail_by_rows() || !L.moe) rows_of_ = nullptr;
                    if (!L.moe) continue;
                    moe_shared(L, il, R);
                    moe_card(L, il, R);
                    layer_tail(L, il, R);
                    rows_of_ = nullptr;
                }
                logits_rows(0, R);
            };
        if (tail_by_rows()) {
            step_replay(step_passes_, wkey, whole_work);
        } else {
            void*& pass = step_passes_[wkey];
            if (!pass) pass = record(whole_work);
            ck(cudaGraphLaunch((cudaGraphExec_t) pass, cs(st_)), "layer graph");
        }
        for (int il = lo_; il < hi_; il = next_layer(il))
            stats_.gpu_experts += layers_[(size_t) il].moe ? R * n_used_ : 0;
        out_done[0] = out_done[1] = true;
        if (bell_) bell_serve();
    }
    // With experts on the CPU, the rows go through the layers in two halves, a sequence's rows in one, as a window's
    // tokens do (run()): this thread routes a half and computes its CPU experts, and meanwhile the thread that hands
    // the card its cached experts hands it first the other half's sum, the rest of its layer and the next layer up to
    // its router.  The card's layer and the CPU's experts overlap instead of taking turns.  A half's recordings are
    // its own (its rows' place in the buffers is in them).  STRATA_POLY_STEP_HALVES=0 keeps the rows together.
    static const bool halves_on = [] {
        const char* v = std::getenv("STRATA_POLY_STEP_HALVES");
        return !(v && v[0] == '0');
    }();
    int mid = R / 2;
    while (mid < R && mid > 0 && rseq[(size_t) mid] == rseq[(size_t) mid - 1]) ++mid;
    // (not where the tails take the rows: a half's tail is recorded by its place in the buffers, not by sequence)
    const bool halved = halves_on && !card && batch_heads() && !tail_by_rows() && !all_cached() && R >= 4 && mid > 0 &&
                        mid < R && !batched(layers_[(size_t) first_moe_layer()], R);
    if (halved) {
        const int first[3] = {0, mid, R};
        int at0 = 0;
        const auto half = [&](int k) {
            shift_rows(first[k] - at0);
            at0 = first[k];
            ev_ = ev_row_[k];
            return first[k + 1] - first[k];
        };
        const auto rec_of = [&](int k) { return recs + 64 * (size_t) first[k] + (size_t) (first[k + 1] - first[k]); };
        const auto head_half = [&](Layer& L, int il, int k) {
            const int r0 = first[k], m = first[k + 1] - first[k];
            const int64_t keep = pos_;
            std::vector<int> hkey{m, 1000 + r0};
            for (int i = r0; i < r0 + m; ++i) {
                pos_ = at[(size_t) i];
                hkey.push_back(rseq[(size_t) i]);
                hkey.push_back(head_key(il, 1));
                if (guesses) hkey.push_back(rsnap[(size_t) i] ? 1 : 0);
            }
            pos_ = keep;
            rows_of_ = rows.data() + r0;
            n_rows_of_ = m;
            if (replay && L.fixed) {
                void*& made = L.rows_heads[hkey];
                if (!made) made = record([&] { layer_head(L, il, m); });
                ck(cudaGraphLaunch((cudaGraphExec_t) made, cs(st_)), "layer graph");
            } else {
                layer_head(L, il, m);
            }
            rows_of_ = nullptr;
        };
        // half k on from layer il, whose experts it has (-1: from the start), to the next router's copy, or through
        // its logits: the first half's are made while the CPU computes the other's last experts
        const auto go_on = [&](int k, int il, bool add) {
            const int m = half(k);
            const size_t rk = rec_of(k);
            if (il >= 0) {
                Layer& L = layers_[(size_t) il];
                if (add) moe_add(m);
                if (replay && L.fixed) replayed(L.tail, rk, [&] { layer_tail(L, il, m); });
                else layer_tail(L, il, m);
            }
            int j = il < 0 ? lo_ : il + 1;
            for (; j < hi_; ++j) {
                Layer& J = layers_[(size_t) j];
                head_half(J, j, k);
                if (!J.moe) continue;
                ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
                if (replay && shared_recorded()) replayed(J.shared, rk, [&] { moe_shared(J, j, m); });
                else moe_shared(J, j, m);
                break;
            }
            if (j == hi_ && !next_) {
                logits_rows(first[k], m);
                out_done[k] = true;
            }
        };
        const auto back = [&] {
            shift_rows(-at0);
            at0 = 0;
            ev_ = ev_row_[0];
            rows_of_ = nullptr;
        };
        struct {
            int k = -1, il = -1;
            bool add = false;
        } late;
        std::exception_ptr failed;
        try {
            for (int k = 0; k < 2; ++k) go_on(k, -1, false);
            for (int il = lo_; il < hi_; ++il) {
                Layer& L = layers_[(size_t) il];
                if (!L.moe) continue;
                for (int k = 0; k < 2; ++k) {
                    const std::function<void()> other = [&] {
                        try {
                            if (split_) cudaSetDevice(device_);
                            go_on(late.k, late.il, late.add);
                            half(k);
                        } catch (...) {
                            failed = std::current_exception();
                        }
                    };
                    const int m = half(k);
                    const bool add = moe(L, il, m, late.k >= 0 ? &other : nullptr);
                    if (failed) std::rethrow_exception(failed);
                    late = {k, il, add};
                }
            }
            if (late.k >= 0) go_on(late.k, late.il, late.add);
        } catch (...) {
            back();
            throw;
        }
        back();
    }
    for (int il = lo_; il < hi_ && !halved && !whole; il = next_layer(il)) {
        Layer& L = layers_[(size_t) il];
        if (batch_heads()) {
            const int64_t keep = pos_;
            key.assign(1, R);
            for (int i = 0; i < R; ++i) {
                pos_ = at[(size_t) i];
                key.push_back(rseq[(size_t) i]);
                key.push_back(head_key(il, 1));
                if (guesses) key.push_back(rsnap[(size_t) i] ? 1 : 0);   // a recording that leaves the state
            }
            pos_ = keep;
            rows_of_ = rows.data();
            n_rows_of_ = R;
            if (replay && L.fixed && tail_by_rows()) {
                step_replay(L.rows_heads, key, [&] { layer_head(L, il, R); });
            } else if (replay && L.fixed) {
                void*& made = L.rows_heads[key];
                if (!made) made = record([&] { layer_head(L, il, R); });
                ck(cudaGraphLaunch((cudaGraphExec_t) made, cs(st_)), "layer graph");
            } else {
                layer_head(L, il, R);
            }
            rows_of_ = nullptr;
        } else {
            for (int i = 0; i < R; ++i) {
                use(rseq[(size_t) i]);
                shift_rows(i);
                pos_ = at[(size_t) i];
                const size_t head = (size_t) head_key(il, 1) * recs + 3 * wn + (size_t) i;
                if (replay && L.fixed) replayed(L.heads, head, [&] { layer_head(L, il, 1); });
                else layer_head(L, il, 1);
                shift_rows(-i);
                pos_ = at[(size_t) i];
            }
        }
        if (!L.moe) continue;
        if (batched(L, R)) {
            moe_rows(L, il, R);
        } else {
            // Rows the host takes, where a token alone is routed on the card (the doorbell's route_card()): they take
            // the card's picks too (moe(), step_ids_).  The host's route() computes the gates' sigmoid with its own
            // exp, which can differ from the card's in the last bit, and a row would not get the weights, or now
            // and then the experts, it gets alone.
            // (a layer card_moe() has not tried yet, L.card == -1, counts as the card's only where its types have the
            // card's pick kernels, as moe_card() would find: else a token alone would be routed on the host)
            const bool layer_card = L.card == 1 || (L.card == -1 && kernels::native_mmvq_ids_supported((int) L.gu_type) &&
                                                    kernels::native_mmvq_ids_supported((int) L.u_type) &&
                                                    kernels::native_mmvq_ids_supported((int) L.d_type));
            const bool card_picks = !card && picks_like_alone && layer_card && route_card(il, R, d_ids_, d_wts_);
            if (card_picks) copy2(d_ids_, h_ids_, R * n_used_, d_wts_, h_wts_, R * n_used_, st_);
            ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
            if (replay && shared_recorded()) replayed(L.shared, rec, [&] { moe_shared(L, il, R); });
            else moe_shared(L, il, R);
            bool on_card = true;
            if (card && L.card == 1 && replay) replayed(L.card_moe, rec, [&] { moe_card(L, il, R); });
            else on_card = card && moe_card(L, il, R);
            if (on_card) {
                stats_.gpu_experts += R * n_used_;
                if (bell_) bell_serve();   // as run() layer by layer: what this thread does next may wait for the card
            } else {
                struct Picks {
                    const int32_t*& ids;
                    ~Picks() { ids = nullptr; }
                } picks_off{step_ids_};
                step_ids_ = card_picks ? h_ids_ : nullptr;
                if (moe(L, il, R)) moe_add(R);
            }
        }
        if (tail_by_rows()) {   // the rows' own states: recorded by their sequences
            std::vector<int> tkey{R, tail_key(il, R)};
            for (int i = 0; i < R; ++i) tkey.push_back(rseq[(size_t) i]);
            rows_of_ = rows.data();
            n_rows_of_ = R;
            if (replay && L.fixed) {
                step_replay(L.rows_tails, tkey, [&] { layer_tail(L, il, R); });
            } else {
                layer_tail(L, il, R);
            }
            rows_of_ = nullptr;
        } else if (replay && L.fixed) {
            replayed(L.tail, rec, [&] { layer_tail(L, il, R); });
        } else {
            layer_tail(L, il, R);
        }
    }

    if (next_) {
        float* h = pass_rows(R);
        ck(cudaMemcpyAsync(h, r_, (size_t) R * r_row * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "the next card");
        carry_out(h, R);
        ck(cudaStreamSynchronize(cs(st_)), "the next card");
        next_->pass_ = h;
        next_->step(seqs, tokens, n);
    } else {
        if (!(out_done[0] && out_done[1])) logits_rows(0, R);
        ck(cudaStreamSynchronize(cs(st_)), "step");
    }
    if (guesses) {   // keep_step() settles the positions
        step_seqs_.assign(seqs, seqs + n);
        step_rows_.assign((size_t) n, 1);
        step_tok_.assign((size_t) 2 * n, 0);
        for (int i = 0; i < n; ++i) {
            step_rows_[(size_t) i] = guesses[i] >= 0 ? 2 : 1;
            step_tok_[(size_t) 2 * i] = tokens[i];
            step_tok_[(size_t) 2 * i + 1] = guesses[i];
        }
        use(home);
        return;
    }
    for (int i = 0; i < n; ++i) {
        use(seqs[i]);
        pos_ = at[(size_t) i] + 1;
        ahead_pos_ = -1;
    }
    use(home);
    stats_.tokens += n;
}

int32_t Engine::picked(int i) const {
    if (next_) return next_->picked(i);   // a layer split: the last card's pick, or its logits
    return picks_ ? h_picks_[i].token : most_likely(logits_of(i), n_vocab_);
}

void Engine::keep_step(const int* kept) {
    if (step_seqs_.empty()) throw std::logic_error("keep_step: no step with guesses to keep");
    OnCard here(device_, split_);
    const int home = bound_;
    const size_t e = (size_t) n_embd_;
    // The draft block's rows of the kept tokens, every sequence's in one pass: token j at the sequence's position
    // + j, with the model's output for the token before it (hn_'s row j).  draft() wrote the first token's already
    // where it guessed from it.
    std::vector<Row> rows;
    std::vector<int32_t> toks;
    for (size_t i = 0; i < step_seqs_.size(); ++i) {
        const int k = kept[i], m = step_rows_[i];
        if (k < 1 || k > m) throw std::logic_error("keep_step keeps 1 to the sequence's rows");
        if (!draft_) continue;
        hold(step_seqs_[i]);
        const int32_t* t = step_tok_.data() + 2 * i;
        for (int j = ahead_pos_ == pos_ && t[0] == ahead_tok_ ? 1 : 0; j < k; ++j) {
            ck(cudaMemcpyAsync(step_h_ + rows.size() * e, hn_ + (size_t) j * e, e * sizeof(float),
                               cudaMemcpyDeviceToDevice, cs(st_)),
               "keep_step");
            rows.push_back({step_seqs_[i], pos_ + j});
            toks.push_back(t[j]);
        }
    }
    if (!rows.empty()) draft_rows_of(rows.data(), toks.data(), (int) rows.size(), nullptr);
    for (size_t i = 0; i < step_seqs_.size(); ++i) {
        const int k = kept[i], m = step_rows_[i];
        hold(step_seqs_[i]);
        // a guess that was wrong: the state after the sequence's first row, which that row left
        if (k < m && rec_n_ > 0)
            ck(cudaMemcpyAsync(state_, step_snap_ + (size_t) step_seqs_[i] * rec_n_, rec_n_ * sizeof(float),
                               cudaMemcpyDeviceToDevice, cs(st_)),
               "keep_step");
        // the output the next token pairs with
        if (draft_) copy2(hn_ + (size_t) k * e, hn_, n_embd_, nullptr, nullptr, 0, st_);
        pos_ += k;
        ahead_pos_ = -1;
        stats_.tokens += k;
    }
    step_seqs_.clear();
    hold(home);
}

// The draft block over rows of different sequences, as draft_step() and draft_rows() run it for one: row i takes
// tokens[i] at rows[i].pos, with step_h_'s row i (the model's output for the token before it), and keeps its rows
// in its own sequence's caches (the architecture's step() rows, rows_of_).  With `out` the rows go through the
// rest of the block too and out[i] gets row i's guess.  The card is left working.
void Engine::draft_rows_of(const Row* rows, const int32_t* tokens, int n, Likeliest* out) {
    Layer& L = layers_[(size_t) n_layer_];
    const size_t e = (size_t) n_embd_;
    for (int i = 0; i < n; ++i) {
        h_at_[2 * i] = rows[i].pos;
        h_at_[2 * i + 1] = turn(rows[i].pos);
    }
    copy2(h_at_, at_, n * 2 * (int) (sizeof(long long) / 4), nullptr, nullptr, 0, st_);
    // draft_input()'s, with each row's own sequence for its embedding (a picture's rows are a sequence's)
    if (!ev_hr_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_hr_ = ev;
    } else {
        ck(cudaEventSynchronize((cudaEvent_t) ev_hr_), "embed");
    }
    for (int i = 0; i < n; ++i) {
        hold(rows[i].seq);
        input(tokens[i], rows[i].pos, h_r_ + (size_t) i * e);
    }
    copy2(h_r_, cur_, n * n_embd_, nullptr, nullptr, 0, st_);
    ck(cudaEventRecord((cudaEvent_t) ev_hr_, cs(st_)), "embed");
    rms_norm(cur_, fp(d_enorm_), r_, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    rms_norm(step_h_, fp(d_hnorm_), r_ + e, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    float* x = r_ + 2 * (size_t) n * e;
    lin(d_proj_, r_, x, n);
    rms_norm(x, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    struct Clear {
        const Row*& p;
        ~Clear() { p = nullptr; }
    } clear{rows_of_};
    rows_of_ = rows;
    n_rows_of_ = n;
    block_attend(rows[0].pos, n);
    rows_of_ = nullptr;
    if (!out) return;
    axpy(ao_, 1.f, x, n * n_embd_, true, st_);
    rms_norm(x, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    moe_router(L, n);
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
    moe_shared(L, n_layer_, n);
    tok_ = nullptr;   // a guess's row has no token of the sequence
    if (moe(L, n_layer_, n)) moe_add(n);
    axpy(fo_, 1.f, x, n * n_embd_, true, st_);
    rms_norm(x, fp(d_head_), step_h_, n_embd_, n, rms_eps_, st_);
    Dev head = output_;   // the matrix's first rows are the common tokens'
    head.n_out = draft_rows();
    lin(head, step_h_, wlogits_, n);
    finish_logits(wlogits_, n);
    for (int r = 0; r < n; ++r) likeliest(wlogits_ + (size_t) r * head.n_out, head.n_out, out + r, st_);
}

void Engine::draft_many(const int* seqs, const int32_t* tokens, int n, int32_t* guesses) {
    if (!steps_guesses() || !step_h_ || n < 1 || n > opt_.batch + 1)
        throw std::logic_error("draft_many: guesses need --batch-mtp and a draft block");
    if (!step_seqs_.empty()) throw std::logic_error("draft_many: the rows of the step before are not kept yet");
    OnCard here(device_, split_);
    end_read();
    const int home = bound_;
    const size_t e = (size_t) n_embd_;
    // a sequence with no room for its token and a guess after it gets no guess
    std::vector<Row> rows;
    std::vector<int32_t> toks;
    std::vector<int> of;
    for (int i = 0; i < n; ++i) {
        guesses[i] = -1;
        hold(seqs[i]);
        if (pos_ + 2 > opt_.max_context) continue;
        cover(pos_ + 1);
        ck(cudaMemcpyAsync(step_h_ + rows.size() * e, hn_, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
           "draft");
        rows.push_back({seqs[i], pos_});
        toks.push_back(tokens[i]);
        of.push_back(i);
    }
    if (!rows.empty()) {
        draft_rows_of(rows.data(), toks.data(), (int) rows.size(), h_guesses_);
        ck(cudaStreamSynchronize(cs(st_)), "draft");
        for (size_t r = 0; r < rows.size(); ++r) {
            hold(rows[r].seq);
            ahead_pos_ = rows[r].pos;
            ahead_tok_ = toks[r];
            guesses[of[r]] = h_guesses_[r].token;
        }
    }
    hold(home);
}

void Engine::shift_rows(ptrdiff_t by) {
    const ptrdiff_t e = n_embd_;
    r_ += by * streams() * e;
    xn_ += by * e;
    fo_ += by * e;
    rl_ += by * n_expert_;
    xq_ = (uint8_t*) xq_ + by * (ptrdiff_t) kernels::native_q8_1_bytes(n_embd_);
    h_rl_ += by * n_expert_;
    h_x_ += by * e;
    h_cpu_ += by * e;
    at_ += by * 2;
    pos_ += by;
    if (tok_) tok_ += by;
}

// Pinned memory for n rows of the residual on their way to the next card.
float* Engine::pass_rows(int n) {
    if (h_pass_rows_ < n) {
        hfree(h_pass_);
        h_pass_ = nullptr;
        h_pass_rows_ = std::max(n, rows_);
        h_pass_ = (float*) hmalloc((size_t) h_pass_rows_ * (streams() + carry_rows()) * n_embd_ * sizeof(float),
                                   "pinned memory");
    }
    return h_pass_;
}

// The n rows of the residual a call from `pos` leaves in r_, to the next card, which reads them for its layers.
void Engine::hand_on(int64_t pos, int n, float* h) {
    ck(cudaMemcpyAsync(h, r_, (size_t) n * streams() * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "the next card");
    carry_out(h, n);
    ck(cudaStreamSynchronize(cs(st_)), "the next card");
    next_->pass_ = h;
    next_->pass_at_ = pos;
}

// Inkling 2's residual streams and the loop's bank, which its calls keep beside r_ (carry_rows()): after the n rows of
// r_ at h, block after block, n rows each.
void Engine::carry_out(float* h, int n) {
    const size_t e = (size_t) n_embd_, at = (size_t) n * streams() * e;
    for (int i = 0; i < carry_rows(); ++i)
        ck(cudaMemcpyAsync(h + at + (size_t) i * n * e, carried(i), (size_t) n * e * sizeof(float),
                           cudaMemcpyDeviceToHost, cs(st_)),
           "the next card");
    next_->pass_n_ = n;
}

// A later card's carried rows for its call of n tokens, from row `from` of the card before's hand-off; the hand-off's
// memory is that card's again when this returns.
void Engine::carry_in(int64_t from, int n) {
    if (!pass_ || carry_rows() == 0) return;
    const size_t e = (size_t) n_embd_;
    const float* c = pass_ + (size_t) pass_n_ * streams() * e;
    for (int i = 0; i < carry_rows(); ++i)
        ck(cudaMemcpyAsync(carried(i), c + ((size_t) i * pass_n_ + (size_t) from) * e, (size_t) n * e * sizeof(float),
                           cudaMemcpyHostToDevice, cs(st_)),
           "the card before");
    ck(cudaStreamSynchronize(cs(st_)), "the card before");
}

void Engine::eval(const int32_t* tokens, int n, bool want_logits) {
    OnCard here(device_, split_);
    run(tokens, n);
    if (next_) {
        hand_on(pos_, n, pass_rows(n));
        next_->eval(tokens, n, want_logits);
        pos_ += n;
        stats_.tokens += n;
        return;
    }
    // The model's output: the last token's for the logits, and every token's for the draft block.
    const bool picks = std::exchange(eval_picks_, false) && want_logits && n == 1 && d_picks_;
    picks_ = picks;
    if (want_logits || draft_) {
        const bool replay = opt_.graphs && n == 1 && want_logits && !opt_.dump && !prompt_scratch_;
        const int m = draft_ ? n : 1;
        float* h = draft_ ? hn_ + n_embd_ : xn_;
        const auto tail = [&] {
            output_rows(n - m, m, h);
            dbg("out_norm", n_layer_, h, n_embd_);
            if (!want_logits) return;
            lin(output_, h + (size_t) (m - 1) * n_embd_, logits_, 1);
            finish_logits(logits_, 1);
            if (picks) {   // the lowest id of the largest score, as most_likely() picks it
                likeliest(logits_, n_vocab_, d_picks_, st_);
                ck(cudaMemcpyAsync(h_picks_, d_picks_, sizeof(Likeliest), cudaMemcpyDeviceToHost, cs(st_)), "logits");
                return;
            }
            ck(cudaMemcpyAsync(h_logits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cs(st_)),
               "logits");
        };
        if (replay) {
            void*& graph = picks ? tail_picks_ : tail_;
            if (!graph) graph = record(tail);
            ck(cudaGraphLaunch((cudaGraphExec_t) graph, cs(st_)), "logits graph");
        } else {
            tail();
        }
        if (want_logits) ck(cudaStreamSynchronize(cs(st_)), "eval");
    }
    // behind the logits: the caller has them while the card does this
    if (draft_) draft_rows(tokens, ahead_pos_ == pos_ && tokens[0] == ahead_tok_ ? 1 : 0, n);
    pos_ += n;
    stats_.tokens += n;
}

// A layer for n tokens, as run() hands it to the card when nothing is replayed or printed.
void Engine::layer(int il, int n) {
    Layer& L = layers_[(size_t) il];
    layer_head(L, il, n);
    if (!L.moe) return;
    if (batched(L, n)) {
        moe_rows(L, il, n);
    } else {
        ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
        moe_shared(L, il, n);
        if (moe(L, il, n)) moe_add(n);
    }
    layer_tail(L, il, n);
}

// A pack of read()'s calls through layer il, its rows at `rows`: each call's head at its rows of the buffers
// (shift_rows()) with its sequence bound, as step() takes a row a sequence, and then the routed experts and the tail
// of all the rows at once.  Sixteen co-reads of 100 tokens were 17 calls a layer of Qwen3.6, each its own router
// wait and its own copies of the experts it routes to outside the cache: 1.2 s, where a prompt of 1,600 tokens takes
// 0.3.  Leaves the last call's sequence bound and pos_ where that call starts.
void Engine::layer_pack(int il, const Pack& p, float* rows) {
    Layer& L = layers_[(size_t) il];
    for (int c = p.first; c < p.first + p.n; ++c) {
        const Call& call = calls_[(size_t) c];
        const ptrdiff_t by = call.at - p.at;
        if (call.seq != bound_) hold(call.seq);
        r_ = rows;
        tok_ = nullptr;
        pos_ = call.pos - by;
        shift_rows(by);
        tok_ = call.tok;
        layer_head(L, il, call.m);
        tok_ = nullptr;
        shift_rows(-by);
    }
    const Call& last = calls_[(size_t) (p.first + p.n - 1)];
    r_ = rows;
    pos_ = last.pos;
    if (L.moe) {
        tok_ = pack_tok_.data() + p.at;
        if (batched(L, p.m)) {
            moe_rows(L, il, p.m);
        } else {
            ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
            moe_shared(L, il, p.m);
            if (moe(L, il, p.m)) moe_add(p.m);
        }
        layer_tail(L, il, p.m);
    }
    tok_ = last.tok;
}

// A prompt's tokens in the calls eval() would get, max_rows() at a time.  With experts outside the cache, each
// call copies every layer's through the staging buffer, which is most of what reading a prompt takes.  So the
// calls go through a layer one after the other before any goes on to the next, and the second one finds the
// layer's experts where the first one's copy left them (moe_rows()).  A call's tokens meet what they met in a
// call of their own, in the same order, so the result is eval()'s.  But a call under stage_ahead tokens takes
// those experts too, which on its own it would not, and its sums are split as a larger call's.  A read of fewer
// than stage_ahead tokens is the calls themselves: the copy of every layer's experts took a second, where its
// calls copy the experts they route to.
//
// All a call needs kept from one layer to the next is its rows of the residual.  r_ holds two calls' rows.  With
// more calls, a call's rows go to host memory when the card is done with them and come back before the call's
// turn at the next layer, in the half of r_ the call two before it has left: both copies are on a stream of
// their own, beside the card's work on the call between.
//
// A checkpoint inside the read (a mark) is taken a layer at a time: when a layer has run the tokens before it,
// the layer's state is what save_state() would find there, and it is copied before the next call changes it.
bool Engine::read(const int32_t* tokens, int n, const Mark* marks, int n_marks,
                  const std::function<bool(int, int)>* turn, const CoRead* co, int n_co) {
    const int lead = bound_;
    int all = n;   // the rows of the read and its co-reads
    for (int j = 0; j < n_co; ++j) {
        if (co[j].seq < 0 || co[j].seq >= sequences() || co[j].seq == lead || co[j].n < 1 ||
            co[j].n > opt_.max_context)
            throw std::logic_error("read: a co-read is of another sequence, 1 to max_context() tokens");
        for (int i = 0; i < j; ++i)
            if (co[i].seq == co[j].seq) throw std::logic_error("read: two co-reads of one sequence");
        all += co[j].n;
    }
    if (n_co > 0 && !co_reads()) throw std::logic_error("read: co-reads are one card's, where a read is several calls");
    // (a read of no tokens of its own takes co-reads: a request whose prompt the conversation holds but its last token)
    if (n < (n_co > 0 ? 0 : 1) || all > read_rows_)
        throw std::logic_error("read takes 1 to max_read() tokens, co-reads among them");
    lock_idle_.store(false, std::memory_order_relaxed);
    if (all >= kStageRows) take_prompt();
    for (const Image& im : images_)
        if (im.at < pos_ + n && im.at + im.n > pos_)
            throw std::logic_error("read takes no rows of a picture: eval() does, a picture a call");
    for (int k = 0; k < n_marks; ++k)
        if (marks[k].at <= (k ? marks[k - 1].at : 0) || marks[k].at >= n)
            throw std::logic_error("read: the marks rise, above 0 and below the tokens read");

    // the calls: up to max_rows() tokens each, and none across a mark; then each co-read's, from its row 0
    calls_.clear();
    for (int at = 0, k = 0; at < n;) {
        const int m = std::min(rows_, (k < n_marks ? marks[k].at : n) - at);
        const bool ends = k < n_marks && at + m == marks[k].at;
        calls_.push_back({at, m, ends ? k : -1, lead, pos_ + at, tokens + at});
        at += m;
        k += ends;
    }
    for (int j = 0, row = n; j < n_co; row += co[j].n, ++j)
        for (int at = 0; at < co[j].n;) {
            const int m = std::min(rows_, co[j].n - at);
            calls_.push_back({row + at, m, -1, co[j].seq, at, co[j].tokens + at});
            at += m;
        }
    const int n_calls = (int) calls_.size();
    // Without co-reads a pack is a call, so the read's sums are split as before.  A call with a mark ends its pack:
    // the layer's state is copied after the pack.
    packs_.clear();
    pack_tok_.clear();
    for (const Call& call : calls_) {
        Pack* p = packs_.empty() ? nullptr : &packs_.back();
        if (n_co > 0 && p && p->m + call.m <= rows_ && calls_[(size_t) (p->first + p->n - 1)].mark < 0) {
            ++p->n;
            p->m += call.m;
        } else {
            packs_.push_back({(int) (&call - calls_.data()), 1, call.at, call.m});
        }
        if (n_co > 0) pack_tok_.insert(pack_tok_.end(), call.tok, call.tok + call.m);
    }
    const int n_packs = (int) packs_.size();
    OnCard here(device_, split_);
    // the co-reads' sequences at the end of them, and the read's own bound again, however the read ends
    struct Rebind {
        Engine& e;
        int lead;
        const CoRead* co;
        int n_co;
        bool done = false;
        void now(bool read) {
            if (std::exchange(done, true)) return;
            for (int j = 0; j < n_co && read; ++j) {
                e.hold(co[j].seq);
                e.pos_ = co[j].n;
            }
            if (n_co > 0) e.hold(lead);
        }
        ~Rebind() {
            try {
                now(false);
            } catch (...) {
            }
        }
    } rebind{*this, lead, co, n_co};
    for (int j = 0; j < n_co; ++j) {   // each co-read's rows, from 0
        hold(co[j].seq);
        if (images_.size()) throw std::logic_error("read: a co-read's sequence holds a picture");
        if (pos_ != 0) throw std::logic_error("read: a co-read's sequence is not at its start (reset it)");
        cover(co[j].n);
    }
    hold(lead);
    cover(pos_ + n);
    // On several cards, each reads through its own layers and hands the rows on, and `turn` hears of a card's
    // turns as its share of the model's layers: [lo_, to).
    const std::function<bool(int, int)>* asked = turn;
    std::function<bool(int, int)> share;
    const auto share_to = [&](int to) {
        if (!asked || (lo_ == 0 && to == n_layer_)) return asked;   // all of them
        share = [&, to](int done, int of) { return (*asked)(lo_ * of + done * (to - lo_), n_layer_ * of); };
        return (const std::function<bool(int, int)>*) &share;
    };
    // An architecture whose calls keep rows of their own between layer runs takes them through the runs a group at a
    // time (read_groups()), where the staging buffer can grow for the read.
    const bool groups = carry_rows() > 0 && n_co == 0 && !next_ && !draft_ && stage_wide_ > stage_narrow_;
    // a call's carried rows reach a later card through eval(), which the card before takes such calls through below
    if (pass_ && carry_rows() > 0) throw std::logic_error("read: a later card's carried rows come through eval()");
    // co-reads go through the layers in packs whatever the experts: one by one their calls would wait as long
    if (n_co == 0 &&
        (n_calls == 1 || !mmq_ || all_cached() || all < opt_.stage_ahead || !(reads_by_layer() || groups))) {
        turn = share_to(n_layer_);   // eval() takes the calls through the cards after this one
        // On several cards the next one takes a call on a thread of its own, while this one runs the call after
        // it.  Each card runs the calls as eval() does, in the same order.
        Behind behind;
        // On one card a checkpoint's state goes to pinned memory in the stream, behind its call, and from there to
        // the checkpoint on aside_ while the card reads the next call.  As save_state() it held the card 8 ms a
        // checkpoint of Qwen3.6 (66 MB), most of it the copy into the checkpoint's fresh pageable memory.
        const bool behind_states = !next_ && aside_ && ev_saved_ && h_state_bytes_ >= card_state_bytes();
        uint8_t* stage = h_state_;
        uint8_t* stage_to = nullptr;
        const std::function<void()> state_copy = [&] {
            ck(cudaEventSynchronize((cudaEvent_t) ev_saved_), "read");
            std::memcpy(stage_to, stage, card_state_bytes());
        };
        struct Settle {   // the copy is done before the checkpoint is anyone's, however the read ends
            Aside* a;
            bool on = false;
            void now() {
                if (std::exchange(on, false)) a->finish();
            }
            ~Settle() {
                try {
                    now();
                } catch (...) {
                }
            }
        } settle{aside_.get()};
        for (int c = 0; c < n_calls; ++c) {
            const Call& call = calls_[(size_t) c];
            if (next_) {
                run(tokens + call.at, call.m);
                behind.wait();
                hand_on(pos_, call.m, pass_rows(call.m));
                behind.go([this, t = tokens + call.at, m = call.m] { next_->eval(t, m, false); });
                pos_ += call.m;
                stats_.tokens += call.m;
            } else {
                eval(tokens + call.at, call.m, false);
            }
            if (call.mark >= 0 && marks[call.mark].state && behind_states) {
                settle.now();   // the copy before is done: h_state_ is free
                stage_to = marks[call.mark].state;
                for (int il = 0; il < (int) layers_.size(); ++il)
                    if (runs(il)) save_layer(il, pos_, stage);
                save_last(stage);
                ck(cudaEventRecord((cudaEvent_t) ev_saved_, cs(st_)), "read");
                aside_->start(state_copy);
                settle.on = true;
            } else if (call.mark >= 0 && marks[call.mark].state) {
                behind.wait();
                save_state(marks[call.mark].state);
            }
            if (turn && c + 1 < n_calls && !(*turn)(c + 1, n_calls)) {
                behind.wait();
                settle.now();
                return false;
            }
        }
        behind.wait();
        settle.now();
        return true;
    }
    widen_stage();
    turn = share_to(hi_);
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");

    ck(cudaStreamSynchronize(cs(st_)), "read");
    lock_settle();
    if (n_swap_ > 0) follow(0);
    if (groups) return read_groups(marks, n_marks, turn, n);
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row, r_bytes = r_row * sizeof(float);
    const size_t part = (size_t) std::max(streams(), draft_ ? 3 : 1) * rows_ * e_row;   // a call's rows of r_
    float* const r0 = r_;
    const int64_t pos0 = pos_;
    const bool host = n_packs > 2;
    if (host && h_res_rows_ < all) {   // for this read's tokens and some more, so that a longer one seldom asks again
        hfree(h_res_);
        h_res_ = nullptr;
        h_res_rows_ = std::min(read_rows_, (all + 4095) / 4096 * 4096);
        h_res_ = (float*) hmalloc((size_t) h_res_rows_ * r_bytes, "pinned memory");
    }
    const auto home = [&](int p) { return h_res_ + (size_t) packs_[(size_t) p].at * r_row; };
    const auto half = [&](int p) { return r0 + (size_t) (host ? p % 2 : p) * part; };
    // pack p's rows back into its half of r_, behind whatever the stream has there of the half's last pack
    const auto fetch = [&](int p) {
        ck(cudaMemcpyAsync(half(p), home(p), (size_t) packs_[(size_t) p].m * r_bytes, cudaMemcpyHostToDevice,
                           cs(rs_)),
           "read");
        ck(cudaEventRecord((cudaEvent_t) ev_in_[p % 2], cs(rs_)), "read");
    };
    for (int p = 0; p < n_packs; ++p) {
        const Pack& pk = packs_[(size_t) p];
        float* const rows = host ? home(p) : h_r_;
        for (int c = pk.first; c < pk.first + pk.n; ++c) {
            const Call& call = calls_[(size_t) c];
            float* const at = rows + (size_t) (call.at - pk.at) * r_row;
            if (pass_)
                std::memcpy(at, pass_ + (size_t) (pos0 + call.at - pass_at_) * r_row, (size_t) call.m * r_bytes);
            if (!pass_) {
                if (call.seq != bound_) hold(call.seq);   // its pictures, none for a co-read
                embed_rows(call.tok, call.pos, call.m, at);
            }
        }
        if (host) continue;
        ck(cudaMemcpyAsync(half(p), h_r_, (size_t) pk.m * r_bytes, cudaMemcpyHostToDevice, cs(st_)), "embed");
        ck(cudaStreamSynchronize(cs(st_)), "embed");   // h_r_ is filled again
    }
    if (bound_ != lead) hold(lead);
    if (host) {
        fetch(0);
        fetch(1);
    }
    // a turn for each layer, and one for the model's output, which the draft block keeps a row of
    const int turns = hi_ - lo_ + (draft_ ? 1 : 0);
    // A checkpoint's layers are copied in the stream, behind the call before it and ahead of the next one's turn
    // at the layer, so the card goes on.  The first goes to pinned memory, at the card's speed, and to its state
    // after the read; the others straight to their states, which are pageable: 2 MB of Qwen3.6's took the card
    // 1.3 ms a layer.
    std::vector<uint8_t*> staged((size_t) n_marks, nullptr);
    for (int m = 0; m < n_marks; ++m)
        if (marks[m].state && h_state_bytes_ >= card_state_bytes()) {
            staged[(size_t) m] = h_state_;
            break;
        }
    shared_ahead_ = all >= opt_.stage_ahead;   // (co-reads of fewer tokens: one pack, its own experts)
    try {
        for (int k = 0; k < turns; ++k) {
            const int il = lo_ + k;
            // Between two turns the caller may give the read up.  The layers so far have the read's tokens in
            // their state and the others have not: the sequence cannot go on from here, nor from where it was.
            if (turn && k > 0 && !(*turn)(k, turns)) {
                if (host) ck(cudaStreamSynchronize(cs(rs_)), "read");
                ck(cudaStreamSynchronize(cs(st_)), "read");
                shared_ahead_ = false;
                rebind.now(false);
                r_ = r0;
                pos_ = pos0;
                return false;
            }
            for (int p = 0; p < n_packs; ++p) {
                const Pack& pk = packs_[(size_t) p];
                if (host) ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_in_[p % 2], 0), "read");
                if (il < n_layer_ && pk.n > 1) {
                    layer_pack(il, pk, half(p));
                } else {
                    for (int c = pk.first; c < pk.first + pk.n; ++c) {
                        const Call& call = calls_[(size_t) c];
                        if (call.seq != bound_) hold(call.seq);   // its caches, state and draft rows
                        r_ = half(p) + (size_t) (call.at - pk.at) * r_row;
                        pos_ = call.pos;
                        tok_ = call.tok;
                        if (il < n_layer_) {
                            layer(il, call.m);
                        } else {
                            output_rows(0, call.m, hn_ + n_embd_);
                            draft_rows(call.tok, ahead_pos_ == pos_ && call.tok[0] == ahead_tok_ ? 1 : 0, call.m);
                            ++read_draft_calls_;
                        }
                    }
                }
                const Call& call = calls_[(size_t) (pk.first + pk.n - 1)];   // (a mark ends its pack)
                if (uint8_t* state = call.mark >= 0 ? marks[call.mark].state : nullptr) {
                    uint8_t* const to = staged[(size_t) call.mark] ? staged[(size_t) call.mark] : state;
                    const int64_t pos = pos_ + call.m;
                    if (il < n_layer_) save_layer(il, pos, to);
                    if (k == turns - 1) {   // what no turn is for: a layer behind the model's, the last output
                        for (int j = n_layer_; j < (int) layers_.size(); ++j) save_layer(j, pos, to);
                        save_last(to);
                    }
                }
                // The card's work on the next pack starts now.  This pack's rows leave beside it, unless this was
                // the last turn, and the half is the next pack's that uses it: the one two on, or the one with
                // this half at the next turn.  After the last turn the next card reads them.
                const bool last = k == turns - 1, out = last && next_;
                if (!host || (last && p + 2 >= n_packs && !out)) continue;
                ck(cudaEventRecord((cudaEvent_t) ev_out_[p % 2], cs(st_)), "read");
                ck(cudaStreamWaitEvent(cs(rs_), (cudaEvent_t) ev_out_[p % 2], 0), "read");
                if (!last || out)
                    ck(cudaMemcpyAsync(home(p), half(p), (size_t) pk.m * r_bytes, cudaMemcpyDeviceToHost,
                                       cs(rs_)),
                       "read");
                if (!last || p + 2 < n_packs) fetch(p + 2 < n_packs ? p + 2 : p % 2);
            }
        }
        if (host) ck(cudaStreamSynchronize(cs(rs_)), "read");
        ck(cudaStreamSynchronize(cs(st_)), "read");
        for (int m = 0; m < n_marks; ++m)
            if (staged[(size_t) m]) state_out(staged[(size_t) m], marks[m].state);
        if (next_ && !host) {   // the rows are in r_, a pack's in each half
            float* h = pass_rows(n);
            for (int p = 0; p < n_packs; ++p)
                ck(cudaMemcpyAsync(h + (size_t) packs_[(size_t) p].at * r_row, half(p), packs_[(size_t) p].m * r_bytes,
                                   cudaMemcpyDeviceToHost, cs(st_)),
                   "the next card");
            ck(cudaStreamSynchronize(cs(st_)), "the next card");
        }
    } catch (...) {
        if (host) cudaStreamSynchronize(cs(rs_));
        shared_ahead_ = false;
        try {
            rebind.now(false);
        } catch (...) {
        }
        r_ = r0;
        pos_ = pos0;
        throw;
    }
    shared_ahead_ = false;
    rebind.now(true);   // each co-read's sequence at its end
    for (int j = 0; j < n_co; ++j) stats_.tokens += co[j].n;
    r_ = r0;
    pos_ = pos0;
    if (next_) {
        next_->pass_ = host ? h_res_ : h_pass_;
        next_->pass_at_ = pos0;
        std::vector<Mark> on(marks, marks + n_marks);   // the next card's part of each state is after this one's
        for (Mark& m : on)
            if (m.state) m.state += card_state_bytes();
        if (!next_->read(tokens, n, on.data(), n_marks, asked)) return false;
    }
    pos_ = pos0 + n;
    stats_.tokens += n;
    return true;
}

// read()'s calls where a call keeps rows of its own from one layer run to the next (carry_rows()): a group of calls
// at a time through every layer run, in the order next_layer() gives, the group's rows of r_ and its carried rows
// on the card.  A run's experts outside VRAM are copied once a group, where a call at a time through all the runs
// copied them once a call.  Inkling 2 Mini UD-Q3_K_XL on the RTX 4070 Laptop, 1,144 of 3,072 experts in VRAM: a
// 30,954-token read copied 15.2 GB a call that way, 472 GB in all, and the card waited 27 of its 71 s for the bus.
// A group is as many calls as the card has VRAM for, the expert cache's among it (66 MB a call of 1,024 tokens
// there); with room for none, a call at a time in the engine's own buffers.  Each call meets the layer runs in the
// order it would alone, and each run's state takes the group's calls in their order.  A mark's state is copied run
// by run behind the mark's call (save_run()), as read() copies a layer's.
bool Engine::read_groups(const Mark* marks, int n_marks, const std::function<bool(int, int)>* turn, int n) {
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    const size_t per = (size_t) rows_ * (r_row + (size_t) carry_rows() * e_row);   // a call's floats
    const int n_calls = (int) calls_.size();
    float* const r0 = r_;
    const int64_t pos0 = pos_;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    if (pool_on_) free_b = vpool_bytes();
    const size_t quarter = pool_on_ ? 0 : (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
    const size_t room = (free_b > quarter ? free_b - quarter : 0) +
                        (exp_ && arena_.mapped > kArenaPiece ? arena_.mapped - kArenaPiece : 0);
    int g = (int) std::min<size_t>((size_t) n_calls, room / (per * sizeof(float)));
    float* base = nullptr;
    const size_t most = (size_t) ((read_rows_ + rows_ - 1) / rows_) * per * sizeof(float);
    for (; g > 1 && !(base = carry_take((size_t) g * per * sizeof(float), most)); g = (g + 1) / 2) {
    }
    g = std::max(g, 1);
    int visits = 0;
    for (int il = lo_; il < hi_; il = next_layer(il)) ++visits;
    const int groups = (n_calls + g - 1) / g, steps = groups * visits;
    std::vector<uint8_t*> staged((size_t) n_marks, nullptr);   // the first mark's state goes through h_state_
    for (int m = 0; m < n_marks; ++m)
        if (marks[m].state && h_state_bytes_ >= card_state_bytes()) {
            staged[(size_t) m] = h_state_;
            break;
        }
    shared_ahead_ = n >= opt_.stage_ahead;
    const auto restore = [&] {
        r_ = r0;
        carry_at(nullptr);
        pos_ = pos0;
        shared_ahead_ = false;
    };
    try {
        for (int k = 0; k < groups; ++k) {
            const int c0 = k * g, c1 = std::min(n_calls, c0 + g);
            const auto rows_of = [&](int c) { return base ? base + (size_t) (c - c0) * per : r0; };
            for (int c = c0; c < c1; ++c) {
                const Call& call = calls_[(size_t) c];
                embed_rows(call.tok, call.pos, call.m, h_r_);
                ck(cudaMemcpyAsync(rows_of(c), h_r_, (size_t) call.m * r_row * sizeof(float), cudaMemcpyHostToDevice,
                                   cs(st_)),
                   "embed");
                ck(cudaStreamSynchronize(cs(st_)), "embed");   // h_r_ is filled again
            }
            int il = lo_;
            for (int v = 0; v < visits; ++v, il = next_layer(il)) {
                const int step = k * visits + v;
                if (turn && step > 0 && !(*turn)(step, steps)) {
                    ck(cudaStreamSynchronize(cs(st_)), "read");
                    restore();
                    return false;
                }
                for (int c = c0; c < c1; ++c) {
                    const Call& call = calls_[(size_t) c];
                    r_ = rows_of(c);
                    carry_at(base ? r_ + (size_t) rows_ * r_row : nullptr);
                    pos_ = call.pos;
                    tok_ = call.tok;
                    layer(il, call.m);
                    if (uint8_t* state = call.mark >= 0 ? marks[call.mark].state : nullptr)
                        save_run(il, call.pos + call.m, staged[(size_t) call.mark] ? staged[(size_t) call.mark] : state);
                }
            }
            for (int c = c0; c < c1; ++c) {   // what no run is for: a layer behind the model's
                const Call& call = calls_[(size_t) c];
                uint8_t* state = call.mark >= 0 ? marks[call.mark].state : nullptr;
                if (!state) continue;
                uint8_t* to = staged[(size_t) call.mark] ? staged[(size_t) call.mark] : state;
                for (int j = n_layer_; j < (int) layers_.size(); ++j) save_layer(j, call.pos + call.m, to);
                save_last(to);
            }
        }
        ck(cudaStreamSynchronize(cs(st_)), "read");
        for (int m = 0; m < n_marks; ++m)
            if (staged[(size_t) m]) state_out(staged[(size_t) m], marks[m].state);
    } catch (...) {
        restore();
        throw;
    }
    restore();
    pos_ = pos0 + n;
    stats_.tokens += n;
    return true;
}

// read_groups()'s rows: `bytes` of VRAM at carry_map_ (addresses for `most`), out of the expert cache where the card
// has not that much free, until end_read() gives it back (carry_give()).  Null where it cannot be mapped.
float* Engine::carry_take(size_t bytes, size_t most) {
    if (!carry_map_.base) vmm_reserve(carry_map_, most);
    if (bytes > carry_map_.reserved) return nullptr;
    if (carry_map_.mapped >= bytes) return (float*) carry_map_.base;
    if (up_) ck(cudaStreamSynchronize(cs(up_)), "a read's rows");
    ck(cudaStreamSynchronize(cs(st_)), "a read's rows");
    const size_t want = up_to(bytes, pool_on_ ? pg_ : grain_) - carry_map_.mapped;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    if (pool_on_) free_b = vpool_bytes();
    const size_t quarter = pool_on_ ? 0 : (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
    if (exp_ && free_b < want + quarter) {
        const size_t had = arena_.mapped;
        shrink_cache(want + quarter - free_b);
        held_is_budget();
        carry_cut_ += had - arena_.mapped;
    }
    carry_lent_ = true;   // carry_give() gives back what was cut even if the map below fails
    try {
        vmm_map(carry_map_, bytes, SIZE_MAX, "a read's rows");
    } catch (const std::runtime_error&) {
        carry_give();
        return nullptr;
    }
    return (float*) carry_map_.base;
}

void Engine::carry_give() {
    if (!carry_lent_) return;
    if (up_) ck(cudaStreamSynchronize(cs(up_)), "a read's rows");
    ck(cudaStreamSynchronize(cs(st_)), "a read's rows");
    vmm_unmap(carry_map_, 0);
    carry_lent_ = false;
    if (exp_ && carry_cut_ > 0) grow_cache(carry_cut_, true);
    carry_cut_ = 0;
}

// The tokens of a window of t that leave the recurrent state as it is after them: each but the last when there
// is a place for each, else as many as there are places, the first token among them and the others spread over
// the window.
unsigned Engine::snap_tokens(int t) const {
    const int most = t - 1;
    if (most <= snap_places_) return (1u << most) - 1u;
    unsigned bits = 0;
    for (int i = 0; i < snap_places_; ++i) bits |= 1u << (i * most / snap_places_);
    return bits;
}

// The recurrent layers are the only ones a wrong guess would spoil: every other layer's row of a position is
// written again when the position gets its real token.  So they leave the state as it was after tokens of the
// window (snap_, snap_tokens()), and keep() puts back the one after the last token that was right.
void Engine::window(const int32_t* tokens, int t) {
    if (t < 1 || t > opt_.window) throw std::logic_error("a window takes 1 to max_window() tokens");
    OnCard here(device_, split_);
    end_read();
    // the two ways are a choice only with experts on the CPU; a call that records launches says nothing of them
    const bool timed = t > 1 && !card_routes();
    const bool recorded = std::exchange(recorded_, false);
    // Under a freeze the way is the measured times', not the request's: a frozen answer's sums are the ones its
    // first token's way gives, whichever way that was.
    //
    // With the next layer's experts uploaded ahead (vram_ahead()) the tokens go together: halves overlap the CPU's
    // experts with the card's layer, and there is little of the CPU's left to overlap, while each half reads the
    // layer's weights again and guesses ahead for fewer tokens.  At 6 GB the tokens together wrote 181 tokens/s
    // where the timed choice wrote 172 (12 GB: 275 and 252), four runs each.
    //
    // The halves go through the layers one after the other in order, each at its own rows of r_ (run()), so not
    // where the layers keep rows of their own or do not run in order (reads_by_layer()): Inkling 2's looped core
    // ran once a token in halves, and its answers lost words.
    if (timed && !freeze_experts_) halves_ = ahead_slots_ == 0 && reads_by_layer() && pick_way(t);
    const Clock::time_point t0 = Clock::now();
    windowed_ = true;
    snap_bits_ = snap_tokens(t);
    snap_at_ = snap_;
    try {
        run(tokens, t);
    } catch (...) {
        windowed_ = false;
        snap_bits_ = 0;
        recorded_ = recorded_ || recorded;
        throw;
    }
    windowed_ = false;
    snap_bits_ = 0;
    std::copy(tokens, tokens + t, win_tok_);
    if (next_) {   // a card's way is timed by its own layers
        hand_on(pos_, t, pass_rows(t));
        if (timed && !recorded_ && !freeze_experts_) took_way(t, ms_since(t0));
        recorded_ = recorded_ || recorded;
        win_ = t;
        next_->window(tokens, t);
        return;
    }
    float* h = draft_ ? hn_ + n_embd_ : xn_;
    output_rows(0, t, h);
    lin(output_, h, wlogits_, t);
    finish_logits(wlogits_, t);
    ck(cudaMemcpyAsync(h_wlogits_, wlogits_, (size_t) t * n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "logits");
    ck(cudaStreamSynchronize(cs(st_)), "window");
    if (timed && !recorded_ && !freeze_experts_) took_way(t, ms_since(t0));
    recorded_ = recorded_ || recorded;
    win_ = t;
}

// How a window of t tokens goes through the layers: in halves, or its tokens together.  Each way is timed a few
// times, then the one that took less is taken.  The other still gets a window now and then: what each takes
// moves with the experts the cache holds and with what else the machine is doing.
bool Engine::pick_way(int t) {
    const Way& whole = way_[0][t];
    const Way& split = way_[1][t];
    if (split.n < kWayFirst || whole.n < kWayFirst) return split.n < whole.n;   // in turn, the whole way first
    const bool halves = split.ms < whole.ms;
    if (++way_calls_[t] < std::min(kWayAgain << way_backoff_[t], kWayAgainMost)) return halves;
    way_calls_[t] = 0;
    if ((kWayAgain << way_backoff_[t]) < kWayAgainMost) ++way_backoff_[t];
    return !halves;
}

void Engine::took_way(int t, double ms) {
    Way& w = way_[halves_ ? 1 : 0][t];
    const Way& other = way_[halves_ ? 0 : 1][t];
    const bool behind = other.n >= kWayFirst && w.ms > other.ms;
    w.ms = w.n < kWayFirst ? (w.ms * w.n + ms) / (w.n + 1) : w.ms + (behind ? kWayBehind : kWayWeight) * (ms - w.ms);
    ++w.n;
    if (behind && w.ms < other.ms) way_calls_[t] = way_backoff_[t] = 0;   // the other way is behind now
}

void Engine::keep(int n) {
    if (n < 1 || n > win_) throw std::logic_error("keep takes 1 to the window's tokens");
    OnCard here(device_, split_);
    // The next card reads the window's rows again for the tokens it puts through its layers again: a right token's
    // row is the same, whatever came after it.
    if (n < win_) {
        // The state after the last right token, when that token left it.  Else the one of the last token before
        // it that did, and the right tokens after that one go through the layers again: they come out as they
        // did, and the state with them.
        const unsigned left = snap_tokens(win_) & ((1u << n) - 1u);   // the right tokens that left their state
        const int from = (int) std::bit_width(left);                  // the tokens the state at hand is after
        ck(cudaMemcpyAsync(state_, snap_ + size_t(std::popcount(left) - 1) * rec_n_, rec_n_ * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "keep");
        if (from < n) {
            pos_ += from;
            try {
                run(win_tok_ + from, n - from);
            } catch (...) {
                pos_ -= from;
                throw;
            }
            pos_ -= from;
        }
    }
    if (draft_) draft_rows(win_tok_, ahead_pos_ == pos_ && win_tok_[0] == ahead_tok_ ? 1 : 0, n);
    if (next_) next_->keep(n);
    pos_ += n;
    stats_.tokens += n;
    win_ = 0;
}

// The draft block's input for n tokens: each token's embedding and the model's output for the token before it
// (h, a row a token), each normed, the two side by side through the block's projection.  Returns the block's
// residual, and leaves its normed attention input in xn_ and xq_.  r_ is free between the model's calls, and the
// block works in it: n rows of the two halves, then the residual.
//
// h_r_ is written once the card has read the last copy out of it (ev_hr_): read()'s draft turn calls this for one
// call after another without waiting for the card in between.
float* Engine::draft_input(const int32_t* tokens, int64_t row, const float* h, int n) {
    const size_t e = (size_t) n_embd_, rows = (size_t) n;
    // read()'s draft turn calls this for one call after another without waiting for the card: the copy of the call
    // before may still be reading h_r_, so the host waits for it before writing the next call's rows
    if (!ev_hr_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_hr_ = ev;
    } else {
        ck(cudaEventSynchronize((cudaEvent_t) ev_hr_), "embed");
    }
    // a call's rows on the CPU threads, as embed_rows() makes them: on this thread a prompt's call of 2,048 held the
    // card up between its draft turns
    const auto one = [&](int t) { input(tokens[t], row + t, h_r_ + (size_t) t * e); };
    if (n >= 64 && workers_) {
        for (int t = 0; t < n; ++t)
            if ((tokens[t] < 0 || tokens[t] >= n_vocab_) && !image_at(row + t))
                throw std::runtime_error("a token id is outside the vocabulary");
        workers_->run(n, one);
    } else {
        for (int t = 0; t < n; ++t) one(t);
    }
    if (n < kStageRows) copy2(h_r_, cur_, n * n_embd_, nullptr, nullptr, 0, st_);
    else ck(cudaMemcpyAsync(cur_, h_r_, rows * e * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");
    ck(cudaEventRecord((cudaEvent_t) ev_hr_, cs(st_)), "embed");
    rms_norm(cur_, fp(d_enorm_), r_, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    rms_norm(h, fp(d_hnorm_), r_ + e, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    float* x = r_ + 2 * rows * e;
    lin(d_proj_, r_, x, n);
    rms_norm(x, fp(layers_[(size_t) n_layer_].attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    return x;
}

// The rows the draft block's attention keeps for tokens [first, n) of a call that has run: token i is at pos() + i,
// and hn_'s row i is the model's output for the token before it.  A row is all a later guess reads of a token, so
// the rest of the block is not computed for it.  Then the call's last output is the one the next token pairs with.
void Engine::draft_rows(const int32_t* tokens, int first, int n) {
    const size_t e = (size_t) n_embd_;
    if (block_layer_ && first < n) {
        draft_input(tokens + first, pos_ + first, hn_ + (size_t) first * e, n - first);
        block_rows(pos_ + first, n - first);
    }
    copy2(hn_ + (size_t) n * e, hn_, n_embd_, nullptr, nullptr, 0, st_);
    ahead_pos_ = -1;
}

// One token through the draft block, at pos() + i.  The first token's row is then what draft_rows() would leave
// for it, and eval() and keep() skip it; the rows of the guesses after it are written again when the positions get
// their real tokens.
void Engine::draft_step(int32_t token, int i) {
    Layer& L = layers_[(size_t) n_layer_];
    float* own = hn_ + n_embd_;   // the block's output, which stands in for the model's from the second guess on
    float* x = draft_input(&token, pos_ + i, i == 0 ? hn_ : own, 1);
    block_attend(pos_ + i, 1);
    axpy(ao_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(L.ffn_norm), xn_, n_embd_, 1, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_, 1, st_);
    moe_router(L, 1);
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
    moe_shared(L, n_layer_, 1);
    tok_ = nullptr;   // a guess's row has no token of the sequence
    if (moe(L, n_layer_, 1)) moe_add(1);
    axpy(fo_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(d_head_), own, n_embd_, 1, rms_eps_, st_);
    Dev head = output_;   // the matrix's first rows are the common tokens'
    head.n_out = draft_rows();
    lin(head, own, logits_, 1);
    finish_logits(logits_, 1);
}

// The model's own guesses at what follows `token`, a block at once (block_mask()): the token at pos() and masks
// after it to the window's end, through every layer together and seeing each other, and each mask's own row of
// the output picks the token at its place.  The block changes no recurrent state; the rows it leaves in the
// attention caches are the window's places, which the window writes again.
void Engine::guess_block(int32_t token) {
    const int n = (int) std::min<int64_t>(opt_.window, opt_.max_context - pos_);
    block_n_ = 0;
    if (n < 2) return;
    int32_t tokens[8];
    tokens[0] = token;
    std::fill(tokens + 1, tokens + n, (int32_t) block_mask());
    block_pass_ = true;
    try {
        run(tokens, n);
    } catch (...) {
        block_pass_ = false;
        throw;
    }
    block_pass_ = false;
    Dev head = output_;   // the matrix's first rows are the common tokens'
    head.n_out = draft_rows();
    output_rows(1, n - 1, xn_);
    lin(head, xn_, wlogits_, n - 1);
    finish_logits(wlogits_, n - 1);
    for (int r = 0; r < n - 1; ++r) likeliest(wlogits_ + (size_t) r * head.n_out, head.n_out, h_block_ + r, st_);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    block_n_ = n - 1;
}

int32_t Engine::draft(int32_t token, int i, float* prob) {
    if (next_) return next_->draft(token, i, prob);   // the last card has the draft block
    OnCard here(device_, split_);
    if (!draft_ && !block_guess_) throw std::logic_error("draft: this model has no block that guesses ahead");
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (i < 0 || pos_ + i >= opt_.max_context) throw std::logic_error("draft: the position is outside the context");
    if (block_guess_) {
        if (i == 0) guess_block(token);
        *prob = i < block_n_ ? h_block_[i].prob : 0.f;
        return i < block_n_ ? h_block_[i].token : 0;
    }
    cover(pos_ + i + 1);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    draft_step(token, i);
    // The guess and its probability are found on the card.  On the host a vocabulary's softmax takes up to 2 ms
    // where the block is unsure, four times what the block takes.
    likeliest(logits_, draft_rows(), h_guess_, st_);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    if (i == 0) {
        ahead_pos_ = pos_;
        ahead_tok_ = token;
    }
    *prob = h_guess_->prob;
    return h_guess_->token;
}

int most_likely(const float* logits, int n) {
    const float* at = std::find(logits, logits + n, largest(logits, n));
    return at == logits + n ? 0 : (int) (at - logits);
}

}  // namespace strata::poly
