#include <optix_device.h>
#include <cuda_runtime.h>
#include "LaunchParams.h"

#include "gdt/random/random.h"

#include "reservoir.h"

#include <stdio.h>

using namespace osc;

namespace osc {
  typedef gdt::LCG<16> Random;

  extern "C" __constant__ LaunchParams optixLaunchParams;

  struct PRD {
    bool visibility;
  };

  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  extern "C" __global__ void __closesthit__empty()
  {

  }
  extern "C" __global__ void __anyhit__empty()
  {

  }

  extern "C" __global__ void __miss__shadow()
  {
    PRD* prd = getPRD<PRD>();
    prd->visibility = true;
  }

  extern "C" __global__ void __raygen__sample()
  {
    
  }
}