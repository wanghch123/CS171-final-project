#pragma once

#include "gdt/math/vec.h"
#include "gdt/random/random.h"

namespace osc {
    struct ReservoirSample {
        gdt::vec3f position;
        gdt::vec3f normal;
        gdt::vec3f color;
        float p_hat;
    };
    struct Reservoir {
        ReservoirSample sample;
        float w_sum;
        uint32_t num_samples;

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
        bool Update(const ReservoirSample &sample, float weight, gdt::LCG<16> random) {
            if (num_samples < 1) {
                this->sample = sample;
                w_sum = weight;
                num_samples = 1;
                return true;
            }
            num_samples++;
            w_sum += weight;
            float p = weight / w_sum;
            if (random() < p) {
                this->sample = sample;
                return true;
            }
            return false;
        }
    };
}