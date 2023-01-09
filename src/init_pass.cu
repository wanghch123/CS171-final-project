// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>
#include "LaunchParams.h"

#include "gdt/random/random.h"

#include "reservoir.h"

#include <stdio.h>

using namespace osc;

#define NUM_LIGHT_SAMPLES 16
#define NUM_PIXEL_SAMPLES 1

#define PI 3.141592653579f
#define INV_PI 0.31830988618379067154

namespace osc {
  
  typedef gdt::LCG<16> Random;

  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  struct PRD {
    Random random;
    vec3f  pixelColor;
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
  
  static __device__ void SampleLight(Random random, vec3f &lightPos, vec3f &lightNorm, vec3f &lightPower, float &sample_pdf)
  {
    vec3f *lightPositions = (vec3f *)optixLaunchParams.lights.positions;
    float *lightSizes = (float *)optixLaunchParams.lights.sizes;
    vec3f *lightColors = (vec3f *)optixLaunchParams.lights.colors;
    int numLights = optixLaunchParams.lights.numLights;
    float *lightProb = (float *)optixLaunchParams.lights.probs;
    float *lightCDF = (float *)optixLaunchParams.lights.cdf;

    float x = random();
    // printf("%f \n", x);
    for (int i = 0; i < numLights; i++)
    {
      if (x < lightCDF[i])
      {
        vec3f pos = lightPositions[i];
        float size = lightSizes[i];
        vec3f color = lightColors[i];
        float prob = lightProb[i];

        float t = random();
        float u = random();
        float v = random();

        if (t < 0.1667f) {
          lightNorm = vec3f(0.0f, 0.0f, -1.0f);
          lightPos = vec3f(pos.x + 2.0f * size * (u - 0.5f), pos.y + 2.0f * size * (v - 0.5f), pos.z - size) + 1e-2f * lightNorm;
        }
        else if (t < 0.3333f) {
          lightNorm = vec3f(0.0f, 0.0f, 1.0f);
          lightPos = vec3f(pos.x + 2.0f * size * (u - 0.5f), pos.y + 2.0f * size * (v - 0.5f), pos.z + size) + 1e-2f * lightNorm;
        }
        else if (t < 0.5f) {
          lightNorm = vec3f(0.0f, -1.0f, 0.0f);
          lightPos = vec3f(pos.x + 2.0f * size * (u - 0.5f), pos.y - size, pos.z + 2.0f * size * (v - 0.5f)) + 1e-2f * lightNorm;
        }
        else if (t < 0.6667f) {
          lightNorm = vec3f(0.0f, 1.0f, 0.0f);
          lightPos = vec3f(pos.x + 2.0f * size * (u - 0.5f), pos.y + size, pos.z + 2.0f * size * (v - 0.5f)) + 1e-2f * lightNorm;
        }
        else if (t < 0.8333f) {
          lightNorm = vec3f(-1.0f, 0.0f, 0.0f);
          lightPos = vec3f(pos.x - size, pos.y + 2.0f * size * (u - 0.5f), pos.z + 2.0f * size * (v - 0.5f)) + 1e-2f * lightNorm;
        }
        else {
          lightNorm = vec3f(1.0f, 0.0f, 0.0f);
          lightPos = vec3f(pos.x + size, pos.y + 2.0f * size * (u - 0.5f), pos.z + 2.0f * size * (v - 0.5f)) + 1e-2f * lightNorm;
        }

        lightPower = color;
        float area = 4.0f * size * size * 6.0f;
        sample_pdf = prob * (1.0f / area);
        return;
      }
    }

    // lightPos = lightPositions[0] + vec3f(0.0f, -lightSizes[0], 0.0f);
    // lightNorm = vec3f(0.0f, -1.0f, 0.0f);
    // lightPower = lightColors[0];
    // return;
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

  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;

    const auto &restir = optixLaunchParams.restir;

    const uint32_t reservoir_index = fbIndex * restir.config.num_eveluated_samples;
    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    vec3f Ng     = normalize(cross(B-A,C-A));
    vec3f Ns = (sbtData.normal)
      ? ((1.f-u-v) * sbtData.normal[index.x]
         +       u * sbtData.normal[index.y]
         +       v * sbtData.normal[index.z])
      : Ng;

    const vec3f rayDir = optixGetWorldRayDirection();

    if (dot(rayDir,Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);
    
    if (dot(Ng,Ns) < 0.f)
      Ns -= 2.f*dot(Ng,Ns)*Ng;
    Ns = normalize(Ns);

    PRD &prd = *getPRD<PRD>();

    const vec3f surfPos
      = (1.f-u-v) * sbtData.vertex[index.x]
      +         u * sbtData.vertex[index.y]
      +         v * sbtData.vertex[index.z];

    optixLaunchParams.frame.posBuffer[fbIndex] = surfPos;
    optixLaunchParams.frame.normalBuffer[fbIndex] = Ns;
    optixLaunchParams.frame.diffuseBuffer[fbIndex] = sbtData.color;
    optixLaunchParams.frame.idBuffer[fbIndex] = sbtData.id;

    if (sbtData.isEmissive) {
      optixLaunchParams.frame.mask[fbIndex] = false;
      // printf("Here!!!\n");
      const int r = int(255.99f*min(sbtData.color.x,1.f));
      const int g = int(255.99f*min(sbtData.color.y,1.f));
      const int b = int(255.99f*min(sbtData.color.z,1.f));
      const uint32_t rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);
      optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }
    else {
      optixLaunchParams.frame.mask[fbIndex] = true;
      int prev_reservoir_index = -1;
      int prev_fbIndex = -1;
      if (restir.config.temporal_reuse && restir.prev_reservoirs) {
        const auto &prev_camera = optixLaunchParams.prev_camera;
        const vec3f d = normalize(surfPos - prev_camera.position);
        const float t = dot(prev_camera.direction, prev_camera.direction) / dot(d, prev_camera.direction);
        const vec3f op = t * d - prev_camera.direction;
        float prev_u = dot(op, prev_camera.horizontal) / dot(prev_camera.horizontal, prev_camera.horizontal) + 0.5f;
        float prev_v = dot(op, prev_camera.vertical) / dot(prev_camera.vertical, prev_camera.vertical) + 0.5f;
        int prev_x = int(prev_u * optixLaunchParams.frame.size.x);
        int prev_y = int(prev_v * optixLaunchParams.frame.size.y);
        // printf("prev_x: %d, x: %d, prev_y: %d, y: %d\n", prev_x, ix, prev_y, iy);
        if (prev_x >= 0 && prev_x < optixLaunchParams.frame.size.x && prev_y >= 0 && prev_y < optixLaunchParams.frame.size.y) {
          prev_fbIndex = prev_x + prev_y * optixLaunchParams.frame.size.x;
          if (optixLaunchParams.frame.prev_idBuffer[prev_fbIndex] == sbtData.id) {
            prev_reservoir_index = prev_fbIndex * restir.config.num_eveluated_samples;
          }
        }
      }
      // printf("x: %d, y: %d\n", ix, iy);
      for (uint8_t i = 0; i < restir.config.num_eveluated_samples; ++i) {
        Reservoir reservoir = Reservoir::New();
        for (uint16_t j = 0; j < restir.config.num_initial_samples; ++j) {
          ReservoirSample sample;
          float sample_pdf;
          SampleLight(prd.random, sample.position, sample.normal, sample.color, sample_pdf);
          sample.shade = Shade(normalize(sample.position - surfPos), Ns, sample.normal, sbtData.color, sample.color, length(sample.position - surfPos));
          sample.p_hat = Luminance(sample.shade);
          // if (sample.p_hat > 0.f) {
          //   printf("p_hat: %f\n", sample.p_hat);
          // }
          if (reservoir.Update(sample, sample.p_hat / max(sample_pdf, 0.001f), prd.random)) {
            // printf("reservoir updated\n");
          }
        }

        if (restir.config.visibility_reuse) {
          bool visibility = false;
          const vec3f light_vec = reservoir.sample.position - optixLaunchParams.frame.posBuffer[fbIndex];
          const float light_dist = length(light_vec);
          const vec3f light_dir = light_vec / light_dist;
          uint32_t u0, u1;
          packPointer( &visibility, u0, u1 );
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
                    1,         // SBT offset
                    2,         // SBT stride
                    1,         // missSBTIndex 
                    u0, u1 );
          if (!visibility) {
            reservoir.w = 0.f;
            reservoir.w_sum = 0.f;
          }
        }

        if (prev_reservoir_index >= 0) {
          if (optixLaunchParams.frame.prev_mask[prev_fbIndex]) {
            const uint32_t prev_reservoir_index = prev_fbIndex * restir.config.num_eveluated_samples;
            const Reservoir &prev_reservoir = restir.prev_reservoirs[prev_reservoir_index + i];
            const uint32_t bounded_num_samples = min(prev_reservoir.num_samples, 20 * reservoir.num_samples);
            const float weight = prev_reservoir.sample.p_hat * prev_reservoir.w * bounded_num_samples;
            reservoir.Update(prev_reservoir.sample, weight, prd.random, bounded_num_samples);
          }
        }

        // if (reservoir.num_samples != 0) {
        //   printf("reservoir.num_samples: %d\n", reservoir.num_samples);
        // }

        reservoir.CalcW();
        restir.reservoirs[reservoir_index + i] = reservoir;
      }
      
    }
  }
  
  extern "C" __global__ void __closesthit__shadow()
  {
    /* not going to be used ... */
  }

  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }

  extern "C" __global__ void __anyhit__shadow()
  { /*! not going to be used */ }
  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    PRD &prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.pixelColor = vec3f(0.f);
    // printf("miss\n");
  }

  extern "C" __global__ void __miss__shadow()
  {
    // we didn't hit anything, so the light is visible
    // vec3f &prd = *(vec3f*)getPRD<vec3f>();
    // prd = vec3f(1.f);
    bool &visibility = *(bool*)getPRD<bool>();
    visibility = true;
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int accumID  = optixLaunchParams.frame.accumID;
    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    PRD prd;
    prd.random.init(ix+accumID*optixLaunchParams.frame.size.x,
                 iy+accumID*optixLaunchParams.frame.size.y);
    prd.pixelColor = vec3f(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    int numPixelSamples = NUM_PIXEL_SAMPLES;

    vec3f pixelColor = 0.f;
    for (int sampleID=0;sampleID<numPixelSamples;sampleID++) {
      // normalized screen plane position, in [0,1]^2
      const vec2f screen(vec2f(ix+prd.random(),iy+prd.random())
                         / vec2f(optixLaunchParams.frame.size));
    
      // generate ray direction
      vec3f rayDir = normalize(camera.direction
                               + (screen.x - 0.5f) * camera.horizontal
                               + (screen.y - 0.5f) * camera.vertical);

      optixTrace(optixLaunchParams.traversable,
                 camera.position,
                 rayDir,
                 0.f,    // tmin
                 1e20f,  // tmax
                 0.0f,   // rayTime
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                 RADIANCE_RAY_TYPE,            // SBT offset
                 RAY_TYPE_COUNT,               // SBT stride
                 RADIANCE_RAY_TYPE,            // missSBTIndex 
                 u0, u1 );
      // pixelColor += prd.pixelColor;
    }
    
  //   const int r = 0;
  //   const int g = 0;
  //   const int b = 0;


  // //   // convert to 32-bit rgba value (we explicitly set alpha to 0xff
  // //   // to make stb_image_write happy ...
  //   const uint32_t rgba = 0xff000000
  //     | (r<<0) | (g<<8) | (b<<16);

  // //   // and write to frame buffer ...
  //   const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
  //   optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
  
} // ::osc
