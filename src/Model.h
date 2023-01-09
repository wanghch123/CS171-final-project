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

#include "gdt/math/AffineSpace.h"
#include <vector>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f              diffuse;

    bool isEmissive = false;
  };
  
  struct Lights {
    std::vector<vec3f> positions;
    std::vector<float> sizes;
    std::vector<vec3f> colors;
  };

  struct Model {
    ~Model()
    { for (auto mesh : meshes) delete mesh; }
    
    std::vector<TriangleMesh *> meshes;
    std::vector<TriangleMesh *> light_meshes;
    Lights lights; 
    //! bounding box of all vertices in the model
    box3f bounds;
  };

  Model *loadOBJ(Model* model, const std::string &objFile, float scale = 1.0f, vec3f translate = vec3f(0.f), vec3f diffuse = vec3f(1.f));
  void addLight(Model *model, const vec3f &pos, const float size, const vec3f &color);
  Model *ContructScene(const std::string &objFile);
}
