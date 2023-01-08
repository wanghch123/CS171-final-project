#include "optix_program.h"
#include "optix_stubs.h"

namespace osc {
    OptixProgram::OptixProgram(const OptixProgramConfig &config, const OptixDeviceContext optixContext) {
        // Create module
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount  = 50;
        moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur     = false;
        pipelineCompileOptions.numPayloadValues   = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
        
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = config.max_trace_depth;
        
        const std::string ptxCode = config.ptxCode;
        
        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                            &moduleCompileOptions,
                                            &pipelineCompileOptions,
                                            ptxCode.c_str(),
                                            ptxCode.size(),
                                            log,&sizeof_log,
                                            &module
                                            ));
        if (sizeof_log > 1) PRINT(log);

        // Create raygen program
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc    = {};
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module            = module;           
        pgDesc.raygen.entryFunctionName = config.raygen.c_str();

        // OptixProgramGroup raypg;
        // char log[2048];
        // size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &raygen_programs_
                                            ));
        if (sizeof_log > 1) PRINT(log);

        // Create miss programs
        miss_programs_.resize(config.miss.size());

        // char log[2048];
        // size_t sizeof_log = sizeof( log );

        pgOptions = {};
        pgDesc    = {};
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module              = module ;           

        for (size_t i = 0; i < config.miss.size(); i++) {
            pgDesc.miss.entryFunctionName = config.miss[i].c_str();
            OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                                &pgDesc,
                                                1,
                                                &pgOptions,
                                                log,&sizeof_log,
                                                &miss_programs_[i]
                                                ));
            if (sizeof_log > 1) PRINT(log);
        }

        // Create hitgroup programs
        hitgroup_programs_.resize(config.closesthit.size());
        // char log[2048];
        // size_t sizeof_log = sizeof( log );
        
        pgOptions  = {};
        pgDesc     = {};
        pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH            = module;           
        pgDesc.hitgroup.moduleAH            = module;

        for (size_t i = 0; i < config.closesthit.size(); i++) {
            pgDesc.hitgroup.entryFunctionNameCH = config.closesthit[i].c_str();
            pgDesc.hitgroup.entryFunctionNameAH = config.anyhit[i].c_str();
            OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                                &pgDesc,
                                                1,
                                                &pgOptions,
                                                log,&sizeof_log,
                                                &hitgroup_programs_[i]
                                                ));
            if (sizeof_log > 1) PRINT(log);
        }

        // Create pipeline
        std::vector<OptixProgramGroup> programGroups;
        programGroups.push_back(raygen_programs_);
        for (auto &miss : miss_programs_) programGroups.push_back(miss);
        for (auto &hitgroup : hitgroup_programs_) programGroups.push_back(hitgroup);

        // char log[2048];
        // size_t sizeof_log = sizeof( log );
        PING;
        PRINT(programGroups.size());
        OPTIX_CHECK(optixPipelineCreate(optixContext,
                                        &pipelineCompileOptions,
                                        &pipelineLinkOptions,
                                        programGroups.data(),
                                        (int)programGroups.size(),
                                        log,&sizeof_log,
                                        &pipeline
                                        ));
        if (sizeof_log > 1) PRINT(log);

        OPTIX_CHECK(optixPipelineSetStackSize
                    (/* [in] The pipeline to configure the stack size for */
                    pipeline, 
                    /* [in] The direct stack size requirement for direct
                        callables invoked from IS or AH. */
                    2*1024,
                    /* [in] The direct stack size requirement for direct
                        callables invoked from RG, MS, or CH.  */                 
                    2*1024,
                    /* [in] The continuation stack requirement. */
                    2*1024,
                    /* [in] The maximum depth of a traversable graph
                        passed to trace. */
                    1));
        if (sizeof_log > 1) PRINT(log);
    }
}