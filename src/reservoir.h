#pragma once

#include "gdt/math/vec.h"
#include "gdt/random/random.h"

namespace osc {
    struct ReservoirSample {
        gdt::vec3f position;
        gdt::vec3f normal;
        gdt::vec3f color;
        gdt::vec3f shade;
        float p_hat = 0.f;
    };
    struct Reservoir {
        ReservoirSample sample;
        float w_sum;
        uint32_t num_samples;
        float w;

#ifdef __CUDACC__
        __device__
#endif
        static Reservoir New() {
            Reservoir res;
            memset(&res, 0, sizeof(Reservoir));
            return res;
        }

#ifdef __CUDACC__
        __device__
#endif
        void Clear() {
            memset(this, 0, sizeof(Reservoir));
        }

#ifdef __CUDACC__
        __device__
#endif
        bool Update(const ReservoirSample &sample, float weight, gdt::LCG<16> random, uint32_t num_new_samples = 1) {
            num_samples += num_new_samples;
            w_sum += weight;
            float p = weight / fmax(w_sum, 0.001f);
            if (random() < p) {
                this->sample = sample;
                return true;
            }
            return false;
        }

#ifdef __CUDACC__
    __device__
#endif
        void CalcW() {
            w = w_sum / fmax(sample.p_hat * num_samples, 0.001f);
            // if (isnan(w)) {
            //     if (isnan(w_sum)) {
            //         printf("Fuck!!!!!!\n");
            //     }
            // }
        }

    };
}