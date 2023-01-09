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

  extern "C" __global__ void __raygen__lighting()
  {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;

    auto &restir = optixLaunchParams.restir;
    const uint32_t reservoir_index = restir.config.num_eveluated_samples * fbIndex;

    bool mask = optixLaunchParams.frame.mask[fbIndex];
    if (mask) {
      vec3f color = vec3f(0.0f);
      for (uint32_t i = 0; i < restir.config.num_eveluated_samples; i++) {
        const Reservoir &reservoir = restir.reservoirs[reservoir_index + i];
        if (!restir.config.visibility_reuse) {
          PRD prd;
          prd.visibility = false;

          const vec3f light_vec = reservoir.sample.position - optixLaunchParams.frame.posBuffer[fbIndex];
          const float light_dist = length(light_vec);
          const vec3f light_dir = light_vec / light_dist;

          uint32_t u0, u1;
          packPointer( &prd, u0, u1 );
          optixTrace(optixLaunchParams.traversable,
                    optixLaunchParams.frame.posBuffer[fbIndex],
                    light_dir,
                    0.001f,    // tmin
                    light_dist * (1.f - 1e-3f),  // tmax
                    0.0f,      // rayTime
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                    0,         // SBT offset
                    1,         // SBT stride
                    0,         // missSBTIndex 
                    u0, u1 );
          if (prd.visibility) {
            color += reservoir.sample.shade * reservoir.w;
            // if (reservoir.sample.shade.x != 0.0f || reservoir.sample.shade.y != 0.0f || reservoir.sample.shade.z != 0.0f) {
            //   printf("color: %f %f %f\n", reservoir.sample.shade.x, reservoir.sample.shade.y, reservoir.sample.shade.z);
            // }
          }
        }
        else {
          color += reservoir.sample.shade * reservoir.w;
          // vec3f temp = color / (float)restir.config.num_eveluated_samples;
          // if (temp.x > 1.0f && temp.y > 1.0f && temp.z > 1.0f) {
          //   printf("reservoir.w: %f\n", reservoir.w);
          //   printf("reservoir.sample.shade: %f %f %f\n", reservoir.sample.shade.x, reservoir.sample.shade.y, reservoir.sample.shade.z);
          // }
        }
      }

      color /= (float)restir.config.num_eveluated_samples;
      // color = vec3f(ix%255/255.f, iy%255/255.f, 0.0f);
      // if (color.x != 0.0f || color.y != 0.0f || color.z != 0.0f) {
      //   printf("color: %f %f %f\n", color.x, color.y, color.z);
      // }
      if (isnan(color.x) || isnan(color.y) || isnan(color.z)) {
        printf("color is nan\n");
      }
      const int r = int(255.99f*min(color.x,1.f));
      const int g = int(255.99f*min(color.y,1.f));
      const int b = int(255.99f*min(color.z,1.f));
      const uint32_t rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);
      optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }
    else {
      // printf("mask is false\n");
    }
  }
}