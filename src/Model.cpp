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

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  


  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices,
                float scale = 1.0f,
                vec3f translate = vec3f(0.f))
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
    
    int newID = mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index] * scale + translate);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
      while (mesh->texcoord.size() < mesh->vertex.size())
        mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
      mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
      mesh->normal.resize(mesh->vertex.size());
    
    return newID;
  }
  
  Model *loadOBJ(Model* model, const std::string &objFile, float scale, vec3f translate, vec3f diffuse)
  {
    const std::string mtlDir
      = objFile.substr(0,objFile.rfind('/')+1);
    PRINT(mtlDir);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
                         &err,
						 objFile.c_str(),
                         mtlDir.c_str(),
                         /* triangulate */true);
    if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "+objFile+":"+mtlDir+" : "+err);
    }

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];

      // std::set<int> materialIDs;
      // for (auto faceMatID : shape.mesh.material_ids)
      //   materialIDs.insert(faceMatID);
      
      // std::map<tinyobj::index_t,int> knownVertices;
      
      // for (int materialID : materialIDs) {
      //   TriangleMesh *mesh = new TriangleMesh;
        
      //   for (int faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
      //     if (shape.mesh.material_ids[faceID] != materialID) continue;
      //     tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
      //     tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
      //     tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          
      //     vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
      //               addVertex(mesh, attributes, idx1, knownVertices),
      //               addVertex(mesh, attributes, idx2, knownVertices));
      //     mesh->index.push_back(idx);
      //     mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
      //     // mesh->diffuse = gdt::randomColor(materialID);
      //   }

      //   if (mesh->vertex.empty())
      //     delete mesh;
      //   else
      //     model->meshes.push_back(mesh);
      // }

      std::map<tinyobj::index_t,int> knownVertices;
      TriangleMesh *mesh = new TriangleMesh;

      for (int i = 0; i < shape.mesh.indices.size(); i += 3) {
        tinyobj::index_t idx0 = shape.mesh.indices[i + 0];
        tinyobj::index_t idx1 = shape.mesh.indices[i + 1];
        tinyobj::index_t idx2 = shape.mesh.indices[i + 2];

        vec3i idx(addVertex(mesh, attributes, idx0, knownVertices, scale, translate),
                  addVertex(mesh, attributes, idx1, knownVertices, scale, translate),
                  addVertex(mesh, attributes, idx2, knownVertices, scale, translate));

        mesh->index.push_back(idx);
        mesh->diffuse = diffuse;
      }
      model->meshes.push_back(mesh);
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
      for (auto vtx : mesh->vertex)
        model->bounds.extend(vtx);

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
  }

  void addLight(Model *model, const vec3f &pos, const float size, const vec3f &color)
  {
    TriangleMesh *mesh = new TriangleMesh;
    mesh->vertex.push_back(pos+vec3f(-size,-size,-size));
    mesh->vertex.push_back(pos+vec3f(+size,-size,-size));
    mesh->vertex.push_back(pos+vec3f(+size,+size,-size));
    mesh->vertex.push_back(pos+vec3f(-size,+size,-size));
    mesh->vertex.push_back(pos+vec3f(-size,-size,+size));
    mesh->vertex.push_back(pos+vec3f(+size,-size,+size));
    mesh->vertex.push_back(pos+vec3f(+size,+size,+size));
    mesh->vertex.push_back(pos+vec3f(-size,+size,+size));
    mesh->index.push_back(vec3i(0,1,2));
    mesh->index.push_back(vec3i(2,3,0));
    mesh->index.push_back(vec3i(4,5,6));
    mesh->index.push_back(vec3i(6,7,4));
    mesh->index.push_back(vec3i(0,1,5));
    mesh->index.push_back(vec3i(5,4,0));
    mesh->index.push_back(vec3i(1,2,6));
    mesh->index.push_back(vec3i(6,5,1));
    mesh->index.push_back(vec3i(2,3,7));
    mesh->index.push_back(vec3i(7,6,2));
    mesh->index.push_back(vec3i(3,0,4));
    mesh->index.push_back(vec3i(4,7,3));
    mesh->isEmissive = true;
    mesh->diffuse = color;
    model->meshes.push_back(mesh);
    model->light_meshes.push_back(mesh);

    model->lights.positions.push_back(pos);
    model->lights.colors.push_back(color);
    model->lights.sizes.push_back(size);
  }

  Model *ContructScene(const std::string &objFile)
  {
    Model *model = new Model;
    // loadOBJ(model, objFile);
    loadOBJ(model, "../../assets/stanford_dragon.obj", 20.f, vec3f(1.5f,2.0f,-1.0f), vec3f(0.725f, 0.71f, 0.68f));
    loadOBJ(model, "../../assets/stanford_bunny.obj", 20.f, vec3f(-2.0f,-0.5f,1.0f), vec3f(0.725f, 0.71f, 0.68f));
    loadOBJ(model, "../../assets/back.obj", 10.f, vec3f(0.f,0.f,0.f), vec3f(0.725f, 0.71f, 0.68f));
    loadOBJ(model, "../../assets/left.obj", 10.f, vec3f(0.f,0.f,0.f), vec3f(0.63f, 0.065f, 0.05f));
    loadOBJ(model, "../../assets/right.obj", 10.f, vec3f(0.f,0.f,0.f), vec3f(0.14f, 0.45f, 0.091f));
    loadOBJ(model, "../../assets/ceiling.obj", 10.f, vec3f(0.f,0.f,0.f), vec3f(0.725f, 0.71f, 0.68f));
    loadOBJ(model, "../../assets/floor.obj", 10.f, vec3f(0.f,0.f,0.f), vec3f(0.725f, 0.71f, 0.68f));
    loadOBJ(model, "../../assets/front.obj", 10.f, vec3f(0.f,0.f,0.f), vec3f(0.725f, 0.71f, 0.68f));
    loadOBJ(model, "../../assets/short_box.obj", 5.f, vec3f(0.f,0.f,0.f), vec3f(0.725f, 0.71f, 0.68f));
    // loadOBJ(model, "../../assets/tall_box.obj", 1.f);
    addLight(model, vec3f(4.0f, 8.0f, 4.0f), 0.5f, vec3f(17.0f,5.0f,5.0f));
    addLight(model, vec3f(-4.0f, 8.0f, -4.0f), 0.5f, vec3f(5.0f,5.0f,17.0f));
    addLight(model, vec3f(4.0f, 8.0f, -4.0f), 0.5f, vec3f(5.0f,17.0f,5.0f));
    return model;
  }
}
