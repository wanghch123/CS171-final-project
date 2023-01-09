#include <optix_device.h>
#include <cuda_runtime.h>
#include "LaunchParams.h"

#include "gdt/random/random.h"

#include "reservoir.h"

#include <stdio.h>

#define PI 3.141592653579f
#define INV_PI 0.31830988618379067154

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

  static __device__ vec3f Shade(vec3f lightDir, vec3f surfNorm, vec3f lightNorm, vec3f surfColor, vec3f lightColor, float distance)
  {
    vec3f radiance = vec3f(0.0f, 0.0f, 0.0f);
    if (dot(lightDir, surfNorm) > 0.f && dot(-lightDir, lightNorm) > 0.f) {
      radiance = surfColor * (float)INV_PI * lightColor * dot(lightDir, surfNorm) * dot(-lightDir, lightNorm) / max(distance * distance, 0.001f);
    }
    return radiance;
  }

  static __device__ float Luminance(const gdt::vec3f &c)
  {
    return 0.299f * c.x + 0.597f * c.y + 0.114f * c.z;
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

  extern "C" __global__ void __raygen__spatial() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;

    const int accumID  = optixLaunchParams.frame.accumID;
    const auto &camera = optixLaunchParams.camera;

    Random random;
    random.init(ix+accumID*optixLaunchParams.frame.size.x, iy+accumID*optixLaunchParams.frame.size.y);

    auto &restir = optixLaunchParams.restir;
    const uint32_t reservoir_index = restir.config.num_eveluated_samples * fbIndex;
    
    const vec3f pos = optixLaunchParams.frame.posBuffer[fbIndex];
    const vec3f normal = optixLaunchParams.frame.normalBuffer[fbIndex];
    const vec3f diffuse = optixLaunchParams.frame.diffuseBuffer[fbIndex];

    const float cam_dist = length(pos - camera.position);

    if (!optixLaunchParams.frame.mask[fbIndex]) {
      return;
    }

    for (uint16_t i = 0; i < restir.config.num_eveluated_samples; ++i) {
      Reservoir reservoir = restir.prev_reservoirs[reservoir_index + i];
      uint32_t num_samples = reservoir.num_samples;
      // if (reservoir.num_samples != 0) {
      //   printf("Here!\n");
      // }
      int final_sample_idx = -1;
      uint32_t valid_spatial_samples = 0;
      uint32_t spatial_samples_fbids[8];

      for (uint8_t j = 0; j < restir.config.num_spatial_samples; j++) {
        const int dx = (2 * random() - 1) * (float)restir.config.spatial_radius;
        const int dy = (2 * random() - 1) * (float)restir.config.spatial_radius;
        // printf("dx: %d, dy: %d\n", dx, dy);
        const int x = max(min(ix + dx, (int)optixLaunchParams.frame.size.x - 1), 0);
        const int y = max(min(iy + dy, (int)optixLaunchParams.frame.size.y - 1), 0);

        const uint32_t neighbor_fbIndex = x + y * optixLaunchParams.frame.size.x;
        const uint32_t neighbor_reservoir_index = restir.config.num_eveluated_samples * neighbor_fbIndex;

        if (!optixLaunchParams.frame.mask[neighbor_fbIndex]) {
          continue;
        }

        const vec3f neighbor_pos = optixLaunchParams.frame.posBuffer[neighbor_fbIndex];
        const vec3f neighbor_normal = optixLaunchParams.frame.normalBuffer[neighbor_fbIndex];
        // const vec3f neighbor_diffuse = optixLaunchParams.frame.diffuseBuffer[neighbor_fbIndex];

        const float neighbor_cam_dist = length(neighbor_pos - camera.position);
        const float d_cam_dist_ratio = fabs(neighbor_cam_dist - cam_dist) / cam_dist;
        const float d_normal = dot(normal, neighbor_normal);

        if ((d_cam_dist_ratio > 0.1f || d_normal < 0.9063077870366499f) && !restir.config.unbiased) {
          continue;
        }

        Reservoir neighbor_reservoir = restir.prev_reservoirs[neighbor_reservoir_index + i];

        const vec3f light_vec = neighbor_reservoir.sample.position - pos;
        const float light_dist = length(light_vec);
        const vec3f light_dir = light_vec / light_dist;

        neighbor_reservoir.sample.shade = Shade(normalize(light_dir), normal, neighbor_reservoir.sample.normal, diffuse, neighbor_reservoir.sample.color, light_dist);
        neighbor_reservoir.sample.p_hat = Luminance(neighbor_reservoir.sample.shade);
        const float weight = neighbor_reservoir.sample.p_hat * neighbor_reservoir.w * neighbor_reservoir.num_samples;
        // if (isnan(weight)) {
        //   printf("get it\n");
        // }
        if (reservoir.Update(neighbor_reservoir.sample, weight, random, neighbor_reservoir.num_samples)) {
          // printf("Here!\n");
          final_sample_idx = valid_spatial_samples;
        }

        spatial_samples_fbids[valid_spatial_samples] = neighbor_fbIndex;
        valid_spatial_samples++;
      }
      int Z = num_samples;
      if (restir.config.unbiased) {
        const vec3f final_light_pos = reservoir.sample.position;
        const vec3f final_light_normal = reservoir.sample.normal;
        const vec3f final_light_color = reservoir.sample.color;
        for (uint32_t j = 0; j < valid_spatial_samples; j++) {
          const vec3f neighbor_pos = optixLaunchParams.frame.posBuffer[spatial_samples_fbids[j]];
          const vec3f neighbor_normal = optixLaunchParams.frame.normalBuffer[spatial_samples_fbids[j]];
          const vec3f neighbor_diffuse = optixLaunchParams.frame.diffuseBuffer[spatial_samples_fbids[j]];

          const vec3f light_vec = final_light_pos - neighbor_pos;
          const float light_dist = length(light_vec);
          const vec3f light_dir = light_vec / light_dist;

          bool shadowed = false;
          if (restir.config.visibility_reuse) {
            PRD prd;
            prd.visibility = false;

            uint32_t u0, u1;
            packPointer( &prd, u0, u1 );
            optixTrace(optixLaunchParams.traversable,
                        neighbor_pos,
                        light_dir,
                        0.001f,    // tmin
                        light_dist * (1.f - 1e-3f),  // tmax
                        0.0f,      // rayTime
                        OptixVisibilityMask( 255 ),
                        OPTIX_RAY_FLAG_NONE,
                        0,  // SBT offset
                        1,      // SBT stride
                        0,  // missSBTIndex 
                        u0, u1 );
            
            if (!prd.visibility) { shadowed = true; }
          }
          if (shadowed || dot(light_dir, neighbor_normal) <= 0.f || dot(-light_dir, final_light_normal) <= 0.f) { continue; }

          Z += restir.prev_reservoirs[spatial_samples_fbids[j] * restir.config.num_eveluated_samples + i].num_samples;
        }
        // reservoir.num_samples = Z;
      }

      bool shadowed = false;
      if (restir.config.visibility_reuse) {
        PRD prd;
        prd.visibility = false;

        const vec3f light_vec = reservoir.sample.position - pos;
        const float light_dist = length(light_vec);
        const vec3f light_dir = light_vec / light_dist;

        uint32_t u0, u1;
        packPointer( &prd, u0, u1 );
        optixTrace(optixLaunchParams.traversable,
                    pos,
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
        if (!prd.visibility) {
          shadowed = true;
        }
      }

      if (shadowed) {
        reservoir.w = 0.f;
        reservoir.w_sum = 0.f;
      }
      else if (restir.config.unbiased) {
        reservoir.w = reservoir.w_sum / fmax(reservoir.sample.p_hat * Z, 0.001f);
        // if (isnan(reservoir.w_sum)) {
        //   printf("reservoir.w_sum is nan\n");
        // }
        // if (isnan(reservoir.w)) {
        //   printf("reservoir.w is nan\n");
        // }
      }
      else {
        // printf("Here!\n");
        // if (reservoir.w_sum > 0.f) {
        //   printf("Here!\n");
        // }
        reservoir.CalcW();
      }
      // if (reservoir.w > 0.f) {
      //   printf("Here!\n");
      // }
      
      restir.reservoirs[reservoir_index + i] = reservoir;
    }
  }
}