// direct-I/O read speed of 64 MiB pieces from a park file: alone, and while another file is written
#include "park_file.hpp"
#include <chrono>
#include <cstdio>
using namespace strata::poly;
using C = std::chrono::steady_clock;
int main(int, char** argv) {
    const std::string dir = argv[1];
    const size_t piece = 64 << 20; const int n = 14;   // ~900 MB
    int err = 0;
    auto f = park::File::open(dir, err), g = park::File::open(dir, err);
    { park::Writer w(256 << 20, nullptr);
      auto t0 = C::now();
      for (int i = 0; i < n; ++i) { auto b = w.take(piece); std::memset(b.get(), i, piece); w.put(f, std::move(b), piece, (int64_t) i * piece); }
      f->wait();
      std::printf("write %.0f MB/s\n", n * piece / 1e6 / std::chrono::duration<double>(C::now() - t0).count()); }
    auto b = park::aligned(piece);
    for (int rep = 0; rep < 2; ++rep) {
        auto t0 = C::now();
        for (int i = 0; i < n; ++i) park::read_at(f->fd(), b.get(), piece, (int64_t) i * piece);
        std::printf("read alone %.0f MB/s\n", n * piece / 1e6 / std::chrono::duration<double>(C::now() - t0).count());
    }
    park::Writer w(256 << 20, nullptr);
    for (int i = 0; i < 4; ++i) { auto c = w.take(piece); std::memset(c.get(), 1, piece); w.put(g, std::move(c), piece, (int64_t) i * piece); }
    auto t0 = C::now();
    for (int i = 0; i < n; ++i) park::read_at(f->fd(), b.get(), piece, (int64_t) i * piece);
    std::printf("read with 256 MiB of writes queued %.0f MB/s\n", n * piece / 1e6 / std::chrono::duration<double>(C::now() - t0).count());
    g->wait();
}
