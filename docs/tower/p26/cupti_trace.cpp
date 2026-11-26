// p26/cupti_trace.cpp: a CUPTI activity tracer for any CUDA program, loaded with CUDA_INJECTION64_PATH (no Nsight).
// Every kernel, copy, memset and runtime call with CUPTI's timestamps (ns) to $CUPTI_OUT as CSV:
//   K,start,end,stream,correlation,name    M,start,end,stream,correlation,kind,bytes    S,start,end,stream,correlation,value
//   R,start,end,thread,correlation,call
// Build: g++ -shared -fPIC -O2 -I/usr/local/cuda/include cupti_trace.cpp -o cupti_trace.so -L/usr/local/cuda/lib64 -lcupti
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

static FILE* out = nullptr;
static std::mutex mu;

static void CUPTIAPI want(uint8_t** buf, size_t* size, size_t* max_records) {
    *size = 16 << 20;
    *buf = (uint8_t*) aligned_alloc(8, *size);
    *max_records = 0;
}

static void CUPTIAPI done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    std::lock_guard<std::mutex> g(mu);
    CUpti_Activity* r = nullptr;
    while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS) {
        switch (r->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_KERNEL: {
            auto* k = (CUpti_ActivityKernel10*) r;
            fprintf(out, "K,%llu,%llu,%u,%u,%.120s\n", (unsigned long long) k->start, (unsigned long long) k->end,
                    k->streamId, k->correlationId, k->name ? k->name : "?");
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto* m = (CUpti_ActivityMemcpy6*) r;
            fprintf(out, "M,%llu,%llu,%u,%u,%u,%llu\n", (unsigned long long) m->start, (unsigned long long) m->end,
                    m->streamId, m->correlationId, (unsigned) m->copyKind, (unsigned long long) m->bytes);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto* m = (CUpti_ActivityMemset4*) r;
            fprintf(out, "S,%llu,%llu,%u,%u,%u\n", (unsigned long long) m->start, (unsigned long long) m->end,
                    m->streamId, m->correlationId, m->value);
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto* a = (CUpti_ActivityAPI*) r;
            const char* name = nullptr;
            cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
            fprintf(out, "R,%llu,%llu,%u,%u,%s\n", (unsigned long long) a->start, (unsigned long long) a->end,
                    a->threadId, a->correlationId, name ? name : "?");
            break;
        }
        default:
            break;
        }
    }
    free(buf);
}

static void finish() {
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> g(mu);
    if (out) fflush(out);
}

extern "C" int InitializeInjection(void) {
    const char* p = getenv("CUPTI_OUT");
    out = fopen(p ? p : "/tmp/cupti.csv", "w");
    if (!out) return 0;
    cuptiActivityRegisterCallbacks(want, done);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    atexit(finish);
    return 1;
}
