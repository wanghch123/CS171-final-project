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

#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include "reservoir.h"

namespace osc {
  using namespace gdt;
  
  enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

  struct TriangleMeshSBTData {
    vec3f  color;
    vec3f *vertex;
    vec3f *normal;
    vec3i *index;
    bool isEmissive;
    int id;
  };

  struct ReSTIR_conig {
    uint16_t num_initial_samples;
    uint8_t num_eveluated_samples;
    uint8_t num_spatial_samples;
    uint8_t spatial_radius;
    uint8_t num_spatial_reuse_pass;
    bool temporal_reuse;
    bool visibility_reuse;
    bool unbiased;
    bool mis_spatial_reuse;
  };
  
  struct LaunchParams
  {
    struct {
      uint32_t *colorBuffer;
      bool *prev_mask;
      bool *mask;
      vec3f *posBuffer;
      vec3f *normalBuffer;
      vec3f *diffuseBuffer;
      int *idBuffer;
      int *prev_idBuffer;
      vec2i     size;
      int       accumID { 0 };
    } frame;
    
    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } prev_camera;

    struct {
      vec3f *positions;
      float *sizes;
      vec3f *colors;
      float *probs;
      float *cdf;
      int    numLights;
    } lights;

    struct {
      ReSTIR_conig config;
      Reservoir *prev_reservoirs;
      Reservoir *reservoirs;
    } restir;

    OptixTraversableHandle traversable;
  };

} // ::osc
