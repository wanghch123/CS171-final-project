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

#include "SampleRenderer.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  extern "C" char embedded_ptx_code[];
  extern "C" char light_ptx_code[];
  extern "C" char spatial_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
  };

  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord_light
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
  };


  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer(Model *model)
    : model(model)
  {
    initOptix();

    restir_config_.num_initial_samples = 32;
    restir_config_.num_eveluated_samples = 4;
    restir_config_.num_spatial_samples = 5;
    restir_config_.num_spatial_reuse_pass = 2;
    restir_config_.spatial_radius = 30;
    restir_config_.visibility_reuse = true;
    restir_config_.temporal_reuse = true;
    restir_config_.unbiased = true;
    restir_config_.mis_spatial_reuse = false;           // TODO: implement 
    launchParams.restir.config = restir_config_;

    // read light from model to cuda buffer

    float scale = 1.0f;
    
    for (int i = 0; i < model->lights.positions.size(); ++i) {
      model->lights.colors[i] *= scale;
    }

    lightposBuffer.alloc_and_upload(model->lights.positions);
    lightcolorBuffer.alloc_and_upload(model->lights.colors);
    lightsizeBuffer.alloc_and_upload(model->lights.sizes);

    launchParams.lights.positions = (vec3f*)lightposBuffer.d_pointer();
    launchParams.lights.colors = (vec3f*)lightcolorBuffer.d_pointer();
    launchParams.lights.sizes = (float*)lightsizeBuffer.d_pointer();
    launchParams.lights.numLights = model->lights.positions.size();

    std::vector<float> probs;
    std::vector<float> cdf;
    float sum = 0;

    for (int i = 0; i < model->lights.positions.size(); ++i) {
      float area = 4 * model->lights.sizes[i] * model->lights.sizes[i] * 6;
      float luminance = 0.299f * model->lights.colors[i].x + 0.587f * model->lights.colors[i].y + 0.114f * model->lights.colors[i].z;
      float prob = area * luminance;
      sum += prob;
      probs.push_back(prob);
    }

    for (int i = 0; i < probs.size(); ++i) {
      probs[i] /= sum;
    }

    lightprobBuffer.alloc_and_upload(probs);
    launchParams.lights.probs = (float*)lightprobBuffer.d_pointer();

    cdf.push_back(probs[0]);
    for (int i = 1; i < probs.size(); ++i) {
      cdf.push_back(cdf[i - 1] + probs[i]);
    }

    lightcdfBuffer.alloc_and_upload(cdf);
    launchParams.lights.cdf = (float*)lightcdfBuffer.d_pointer();
      
    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();
      
    // std::cout << "#osc: setting up module ..." << std::endl;
    // createModule();

    // std::cout << "#osc: creating raygen programs ..." << std::endl;
    // createRaygenPrograms();
    // std::cout << "#osc: creating miss programs ..." << std::endl;
    // createMissPrograms();
    // std::cout << "#osc: creating hitgroup programs ..." << std::endl;
    // createHitgroupPrograms();

    OptixProgramConfig programConfig;
    programConfig.ptxCode = embedded_ptx_code;
    programConfig.launch_params = "optixLaunchParams";
    programConfig.max_trace_depth = 2;
    programConfig.raygen = "__raygen__renderFrame";
    programConfig.miss = {"__miss__radiance", "__miss__shadow"};
    programConfig.closesthit = {"__closesthit__radiance", "__closesthit__shadow"};
    programConfig.anyhit = {"__anyhit__radiance", "__anyhit__shadow"};
    
    program = std::make_unique<OptixProgram>(programConfig, optixContext);

    OptixProgramConfig spatial_programConfig;
    spatial_programConfig.ptxCode = spatial_ptx_code;
    spatial_programConfig.launch_params = "optixLaunchParams";
    spatial_programConfig.max_trace_depth = 2;
    spatial_programConfig.raygen = "__raygen__spatial";
    spatial_programConfig.miss = {"__miss__shadow"};
    spatial_programConfig.closesthit = {"__closesthit__empty"};
    spatial_programConfig.anyhit = {"__anyhit__empty"};

    spatial_program = std::make_unique<OptixProgram>(spatial_programConfig, optixContext);

    OptixProgramConfig light_programConfig;
    light_programConfig.ptxCode = light_ptx_code;
    light_programConfig.launch_params = "optixLaunchParams";
    light_programConfig.max_trace_depth = 2;
    light_programConfig.raygen = "__raygen__lighting";
    light_programConfig.miss = {"__miss__shadow"};
    light_programConfig.closesthit = {"__closesthit__empty"};
    light_programConfig.anyhit = {"__anyhit__empty"};

    light_program = std::make_unique<OptixProgram>(light_programConfig, optixContext);

    launchParams.traversable = buildAccel();

    const size_t reserviors_buffer_size = sizeof(Reservoir) * restir_config_.num_eveluated_samples * 1200 * 800;
    reservoirs_buffer_[0].alloc(reserviors_buffer_size);
    reservoirs_buffer_[1].alloc(reserviors_buffer_size);
    
    // std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    // createPipeline();

    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();
    buildSBT_lighting();
    buildSBT_spatial();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  OptixTraversableHandle SampleRenderer::buildAccel()
  {
    PING;
    PRINT(model->meshes.size());
    
    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());
    normalBuffer.resize(model->meshes.size());
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<uint32_t> triangleInputFlags(model->meshes.size());

    for (int meshID=0;meshID<model->meshes.size();meshID++) {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];
      vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
      indexBuffer[meshID].alloc_and_upload(mesh.index);

      if (!mesh.normal.empty())
        normalBuffer[meshID].alloc_and_upload(mesh.normal);
  
      triangleInput[meshID] = {};
      triangleInput[meshID].type
        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      // create local variables, because we need a *pointer* to the
      // device pointers
      d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
      d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
      triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
      triangleInput[meshID].triangleArray.numVertices         = (int)mesh.vertex.size();
      triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
      triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(vec3i);
      triangleInput[meshID].triangleArray.numIndexTriplets    = (int)mesh.index.size();
      triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
      triangleInputFlags[meshID] = 0 ;
    
      // in this example we have one SBT entry, and no per-primitive
      // materials:
      triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
      triangleInput[meshID].triangleArray.numSbtRecords               = 1;
      triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)model->meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    
    return asHandle;
  }
  
  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;
      
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }



  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  // void SampleRenderer::createModule()
  // {
  //   moduleCompileOptions.maxRegisterCount  = 50;
  //   moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  //   moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  //   pipelineCompileOptions = {};
  //   pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  //   pipelineCompileOptions.usesMotionBlur     = false;
  //   pipelineCompileOptions.numPayloadValues   = 2;
  //   pipelineCompileOptions.numAttributeValues = 2;
  //   pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
  //   pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
  //   pipelineLinkOptions.maxTraceDepth          = 2;
      
  //   const std::string ptxCode = embedded_ptx_code;
      
  //   char log[2048];
  //   size_t sizeof_log = sizeof( log );
  //   OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
  //                                        &moduleCompileOptions,
  //                                        &pipelineCompileOptions,
  //                                        ptxCode.c_str(),
  //                                        ptxCode.size(),
  //                                        log,&sizeof_log,
  //                                        &module
  //                                        ));
  //   if (sizeof_log > 1) PRINT(log);
  // }
    


  /*! does all setup for the raygen program(s) we are going to use */
  // void SampleRenderer::createRaygenPrograms()
  // {
  //   // we do a single ray gen program in this example:
  //   raygenPGs.resize(1);
      
  //   OptixProgramGroupOptions pgOptions = {};
  //   OptixProgramGroupDesc pgDesc    = {};
  //   pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  //   pgDesc.raygen.module            = module;           
  //   pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  //   // OptixProgramGroup raypg;
  //   char log[2048];
  //   size_t sizeof_log = sizeof( log );
  //   OPTIX_CHECK(optixProgramGroupCreate(optixContext,
  //                                       &pgDesc,
  //                                       1,
  //                                       &pgOptions,
  //                                       log,&sizeof_log,
  //                                       &raygenPGs[0]
  //                                       ));
  //   if (sizeof_log > 1) PRINT(log);
  // }
    
  /*! does all setup for the miss program(s) we are going to use */
  // void SampleRenderer::createMissPrograms()
  // {
  //   // we do a single ray gen program in this example:
  //   missPGs.resize(RAY_TYPE_COUNT);

  //   char log[2048];
  //   size_t sizeof_log = sizeof( log );

  //   OptixProgramGroupOptions pgOptions = {};
  //   OptixProgramGroupDesc pgDesc    = {};
  //   pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
  //   pgDesc.miss.module              = module ;           

  //   // ------------------------------------------------------------------
  //   // radiance rays
  //   // ------------------------------------------------------------------
  //   pgDesc.miss.entryFunctionName = "__miss__radiance";

  //   OPTIX_CHECK(optixProgramGroupCreate(optixContext,
  //                                       &pgDesc,
  //                                       1,
  //                                       &pgOptions,
  //                                       log,&sizeof_log,
  //                                       &missPGs[RADIANCE_RAY_TYPE]
  //                                       ));
  //   if (sizeof_log > 1) PRINT(log);

  //   // ------------------------------------------------------------------
  //   // shadow rays
  //   // ------------------------------------------------------------------
  //   pgDesc.miss.entryFunctionName = "__miss__shadow";

  //   OPTIX_CHECK(optixProgramGroupCreate(optixContext,
  //                                       &pgDesc,
  //                                       1,
  //                                       &pgOptions,
  //                                       log,&sizeof_log,
  //                                       &missPGs[SHADOW_RAY_TYPE]
  //                                       ));
  //   if (sizeof_log > 1) PRINT(log);
  // }
    
  /*! does all setup for the hitgroup program(s) we are going to use */
  // void SampleRenderer::createHitgroupPrograms()
  // {
  //   // for this simple example, we set up a single hit group
  //   hitgroupPGs.resize(RAY_TYPE_COUNT);

  //   char log[2048];
  //   size_t sizeof_log = sizeof( log );
      
  //   OptixProgramGroupOptions pgOptions  = {};
  //   OptixProgramGroupDesc    pgDesc     = {};
  //   pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  //   pgDesc.hitgroup.moduleCH            = module;           
  //   pgDesc.hitgroup.moduleAH            = module;           

  //   // -------------------------------------------------------
  //   // radiance rays
  //   // -------------------------------------------------------
  //   pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  //   pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  //   OPTIX_CHECK(optixProgramGroupCreate(optixContext,
  //                                       &pgDesc,
  //                                       1,
  //                                       &pgOptions,
  //                                       log,&sizeof_log,
  //                                       &hitgroupPGs[RADIANCE_RAY_TYPE]
  //                                       ));
  //   if (sizeof_log > 1) PRINT(log);

  //   // -------------------------------------------------------
  //   // shadow rays: technically we don't need this hit group,
  //   // since we just use the miss shader to check if we were not
  //   // in shadow
  //   // -------------------------------------------------------
  //   pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
  //   pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  //   OPTIX_CHECK(optixProgramGroupCreate(optixContext,
  //                                       &pgDesc,
  //                                       1,
  //                                       &pgOptions,
  //                                       log,&sizeof_log,
  //                                       &hitgroupPGs[SHADOW_RAY_TYPE]
  //                                       ));
  //   if (sizeof_log > 1) PRINT(log);
  // }
    

  /*! assembles the full pipeline of all programs */
  // void SampleRenderer::createPipeline()
  // {
  //   std::vector<OptixProgramGroup> programGroups;
  //   for (auto pg : raygenPGs)
  //     programGroups.push_back(pg);
  //   for (auto pg : hitgroupPGs)
  //     programGroups.push_back(pg);
  //   for (auto pg : missPGs)
  //     programGroups.push_back(pg);
      
  //   char log[2048];
  //   size_t sizeof_log = sizeof( log );
  //   PING;
  //   PRINT(programGroups.size());
  //   OPTIX_CHECK(optixPipelineCreate(optixContext,
  //                                   &pipelineCompileOptions,
  //                                   &pipelineLinkOptions,
  //                                   programGroups.data(),
  //                                   (int)programGroups.size(),
  //                                   log,&sizeof_log,
  //                                   &pipeline
  //                                   ));
  //   if (sizeof_log > 1) PRINT(log);

  //   OPTIX_CHECK(optixPipelineSetStackSize
  //               (/* [in] The pipeline to configure the stack size for */
  //                pipeline, 
  //                /* [in] The direct stack size requirement for direct
  //                   callables invoked from IS or AH. */
  //                2*1024,
  //                /* [in] The direct stack size requirement for direct
  //                   callables invoked from RG, MS, or CH.  */                 
  //                2*1024,
  //                /* [in] The continuation stack requirement. */
  //                2*1024,
  //                /* [in] The maximum depth of a traversable graph
  //                   passed to trace. */
  //                1));
  //   if (sizeof_log > 1) PRINT(log);
  // }



  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;

    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(program->RaygenProgram(),&rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);

    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<program->MissPrograms().size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(program->MissPrograms()[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      for (int rayID=0;rayID<RAY_TYPE_COUNT;rayID++) {
        auto mesh = model->meshes[meshID];
      
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(program->HitgroupPrograms()[rayID],&rec));
        if (mesh->isEmissive) {
          rec.data.isEmissive = true;
          rec.data.color = mesh->diffuse;
        } else {
          rec.data.isEmissive = false;
          rec.data.color = mesh->diffuse;
        }
        rec.data.index    = (vec3i*)indexBuffer[meshID].d_pointer();
        rec.data.vertex   = (vec3f*)vertexBuffer[meshID].d_pointer();
        rec.data.normal   = (vec3f*)normalBuffer[meshID].d_pointer();
        rec.data.id       = meshID;
        hitgroupRecords.push_back(rec);
      }
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }

  void SampleRenderer::buildSBT_lighting()
  {
    std::vector<RaygenRecord> raygenRecords;

    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(light_program->RaygenProgram(),&rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);

    raygenRecordsBuffer_lighting.alloc_and_upload(raygenRecords);
    sbt_lighting.raygenRecord = raygenRecordsBuffer_lighting.d_pointer();

    std::vector<MissRecord> missRecords;
    for (int i=0;i<light_program->MissPrograms().size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(light_program->MissPrograms()[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer_lighting.alloc_and_upload(missRecords);
    sbt_lighting.missRecordBase          = missRecordsBuffer_lighting.d_pointer();
    sbt_lighting.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_lighting.missRecordCount         = (int)missRecords.size();

    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord_light> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      for (int i = 0; i < light_program->HitgroupPrograms().size(); i++) {
        HitgroupRecord_light rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(light_program->HitgroupPrograms()[i],&rec));
        rec.data = nullptr;
        hitgroupRecords.push_back(rec);
      }
    }
    hitgroupRecordsBuffer_lighting.alloc_and_upload(hitgroupRecords);
    sbt_lighting.hitgroupRecordBase          = hitgroupRecordsBuffer_lighting.d_pointer();
    sbt_lighting.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord_light);
    sbt_lighting.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }

  void SampleRenderer::buildSBT_spatial()
  {
    std::vector<RaygenRecord> raygenRecords;

    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(spatial_program->RaygenProgram(),&rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);

    raygenRecordsBuffer_spatial.alloc_and_upload(raygenRecords);
    sbt_spatial.raygenRecord = raygenRecordsBuffer_spatial.d_pointer();

    std::vector<MissRecord> missRecords;
    for (int i=0;i<spatial_program->MissPrograms().size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(spatial_program->MissPrograms()[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer_spatial.alloc_and_upload(missRecords);
    sbt_spatial.missRecordBase          = missRecordsBuffer_spatial.d_pointer();
    sbt_spatial.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_spatial.missRecordCount         = (int)missRecords.size();

    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord_light> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      for (int i = 0; i < spatial_program->HitgroupPrograms().size(); i++) {
        HitgroupRecord_light rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(spatial_program->HitgroupPrograms()[i],&rec));
        rec.data = nullptr;
        hitgroupRecords.push_back(rec);
      }
    }
    hitgroupRecordsBuffer_spatial.alloc_and_upload(hitgroupRecords);
    sbt_spatial.hitgroupRecordBase          = hitgroupRecordsBuffer_spatial.d_pointer();
    sbt_spatial.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord_light);
    sbt_spatial.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }

  /*! render one frame */
  void SampleRenderer::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) return;

    launchParams.restir.reservoirs = (Reservoir *)reservoirs_buffer_[reservoirs_buffer_curr_index_].d_pointer();
    if (reservoirs_buffer_prev_valid_) {
      launchParams.restir.prev_reservoirs = (Reservoir *)reservoirs_buffer_[reservoirs_buffer_curr_index_ ^ 1].d_pointer();
    } else {
      launchParams.restir.prev_reservoirs = nullptr;
      reservoirs_buffer_prev_valid_ = true;
    }

    launchParams.frame.mask = (bool*)maskBuffer[maskBuffer_curr_index_].d_pointer();
    launchParams.frame.idBuffer = (int*)idBuffer[idBuffer_curr_index_].d_pointer();
    if (maskBuffer_prev_valid_) {
      launchParams.frame.prev_mask = (bool*)maskBuffer[maskBuffer_curr_index_ ^ 1].d_pointer();
      launchParams.frame.prev_idBuffer = (int*)idBuffer[idBuffer_curr_index_ ^ 1].d_pointer();
    } else {
      launchParams.frame.prev_mask = nullptr;
      maskBuffer_prev_valid_ = true;
    }

    reservoirs_buffer_curr_index_ ^= 1;
    maskBuffer_curr_index_ ^= 1;
    idBuffer_curr_index_ ^= 1;
      
    launchParamsBuffer.upload(&launchParams,1);
    
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            program->Pipeline(),stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    for (uint8_t i = 0; i < restir_config_.num_spatial_reuse_pass; i++) {
      launchParams.restir.reservoirs = (Reservoir *)reservoirs_buffer_[reservoirs_buffer_curr_index_].d_pointer();
      launchParams.restir.prev_reservoirs = (Reservoir *)reservoirs_buffer_[reservoirs_buffer_curr_index_ ^ 1].d_pointer();
      reservoirs_buffer_curr_index_ ^= 1;

      launchParamsBuffer.upload(&launchParams,1);

      OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                              spatial_program->Pipeline(),stream,
                              /*! parameters and SBT */
                              launchParamsBuffer.d_pointer(),
                              launchParamsBuffer.sizeInBytes,
                              &sbt_spatial,
                              /*! dimensions of the launch: */
                              launchParams.frame.size.x,
                              launchParams.frame.size.y,
                              1
                              ));
      CUDA_SYNC_CHECK();
    }

    launchParamsBuffer.upload(&launchParams,1);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            light_program->Pipeline(),stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt_lighting,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));
    CUDA_SYNC_CHECK();

    launchParams.frame.accumID++;
  }

  /*! set camera to render with */
  void SampleRenderer::setCamera(const Camera &camera)
  {
    lastSetCamera = camera;
    if (reservoirs_buffer_prev_valid_) {
      launchParams.prev_camera.position  = launchParams.camera.position;
      launchParams.prev_camera.direction = launchParams.camera.direction;
      launchParams.prev_camera.horizontal = launchParams.camera.horizontal;
      launchParams.prev_camera.vertical = launchParams.camera.vertical;
    }
    else {
      launchParams.prev_camera.position  = camera.from;
      launchParams.prev_camera.direction = normalize(camera.at-camera.from);
      const float cosFovy = 0.66f;
      const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
      launchParams.prev_camera.horizontal
        = cosFovy * aspect * normalize(cross(launchParams.prev_camera.direction,
                                             camera.up));
      launchParams.prev_camera.vertical
        = cosFovy * normalize(cross(launchParams.prev_camera.horizontal,
                                    launchParams.prev_camera.direction));
    }
    // reservoirs_buffer_prev_valid_ = false;
    launchParams.camera.position  = camera.from;
    launchParams.camera.direction = normalize(camera.at-camera.from);
    const float cosFovy = 0.66f;
    const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
    launchParams.camera.horizontal
      = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                                           camera.up));
    launchParams.camera.vertical
      = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
  }
  
  void SampleRenderer::updateCamera() {
    if (reservoirs_buffer_prev_valid_) {
      launchParams.prev_camera.position  = launchParams.camera.position;
      launchParams.prev_camera.direction = launchParams.camera.direction;
      launchParams.prev_camera.horizontal = launchParams.camera.horizontal;
      launchParams.prev_camera.vertical = launchParams.camera.vertical;
    }
  }

  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {
    // if window minimized
    if (newSize.x == 0 || newSize.y == 0) return;
    
    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));
    maskBuffer[0].resize(newSize.x*newSize.y*sizeof(bool));
    maskBuffer[1].resize(newSize.x*newSize.y*sizeof(bool));
    posBuffer.resize(newSize.x*newSize.y*sizeof(vec3f));
    norBuffer.resize(newSize.x*newSize.y*sizeof(vec3f));
    diffuseBuffer.resize(newSize.x*newSize.y*sizeof(vec3f));

    idBuffer[0].resize(newSize.x*newSize.y*sizeof(int));
    idBuffer[1].resize(newSize.x*newSize.y*sizeof(int));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size  = newSize;
    launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
    launchParams.frame.mask = (bool*)maskBuffer[maskBuffer_curr_index_].d_pointer();
    launchParams.frame.prev_mask = (bool*)maskBuffer[maskBuffer_curr_index_ ^ 1].d_pointer();
    launchParams.frame.idBuffer = (int*)idBuffer[idBuffer_curr_index_].d_pointer();
    launchParams.frame.prev_idBuffer = (int*)idBuffer[idBuffer_curr_index_ ^ 1].d_pointer();
    // maskBuffer_prev_valid_ = false;
    launchParams.frame.posBuffer = (vec3f*)posBuffer.d_pointer();
    launchParams.frame.normalBuffer = (vec3f*)norBuffer.d_pointer();
    launchParams.frame.diffuseBuffer = (vec3f*)diffuseBuffer.d_pointer();

    const size_t reserviors_buffer_size = sizeof(Reservoir) * restir_config_.num_eveluated_samples * newSize.x * newSize.y;
    reservoirs_buffer_[0].resize(reserviors_buffer_size);
    reservoirs_buffer_[1].resize(reserviors_buffer_size);
    reservoirs_buffer_prev_valid_ = false;

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);
  }

  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    colorBuffer.download(h_pixels,
                         launchParams.frame.size.x*launchParams.frame.size.y);
  }
  
} // ::osc
