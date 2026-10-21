// Test-only C facade over the exported C++ ABI. Never linked into the wheel.
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
using V = const std::vector<int64_t>&;
using Forward = void (*)(void*, void*, V, V, void*, V, V, void*, V, V,
                        void*, V, V, float, bool, void*, uint64_t, int32_t);
using Query = size_t (*)(int,int,int,int,int,int,int,bool,int,bool);
static thread_local std::string error;
extern "C" const char* query_raw(void* fn, const int* dims, int splits,
                                bool tune, uint64_t* result) {
    try {
        *result = reinterpret_cast<Query>(fn)(dims[0],dims[1],dims[2],dims[3],
                  dims[4],dims[5],dims[6],false,splits,tune);
        return nullptr;
    } catch (const std::exception& e) { error=e.what(); return error.c_str(); }
}
extern "C" const char* call_raw(void* fn, void* stream, void* q, void* k,
    void* v, void* o, const int64_t* shapes, const int64_t* strides,
    float scale, bool causal, void* workspace, uint64_t capacity, int32_t dtype) {
    try {
        std::vector<int64_t> sh[4], st[4];
        for (int i=0;i<4;++i) {
            sh[i].assign(shapes+4*i,shapes+4*i+4);
            st[i].assign(strides+4*i,strides+4*i+4);
        }
        reinterpret_cast<Forward>(fn)(stream,q,sh[0],st[0],k,sh[1],st[1],
            v,sh[2],st[2],o,sh[3],st[3],scale,causal,workspace,capacity,dtype);
        return nullptr;
    } catch (const std::exception& e) { error=e.what(); return error.c_str(); }
}
