#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"
#include "Geometries.h"
#include "Light.h"
#include "Config.h"

using namespace optix;

#define PI 3.14159265359

#define MIDPOINT 0.5 

rtBuffer<QuadLight> qlights;

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );
rtDeclareVariable(int1, reduceVariance, , );
rtDeclareVariable(int1, lightSamples, ,);
rtDeclareVariable(int1, lightStratify, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

float3 f(float3 w_i, float3 w_o, float3 n, float3 k_d, float3 k_s, float s) {
    float3 r = reflect(-w_o, n);
    return (k_d / PI) + k_s * ((s + 2) / (2 * PI)) * pow(fmaxf(dot(r, w_i), 0), s);
}

float G(float3 x, float3 x_prime, float3 surface_n, float3 light_n) {
    float R = length(x_prime - x);
    float3 w_i = normalize(x_prime - x);
    return (1 / (R * R)) * fmaxf(dot(surface_n, w_i), 0) * fmaxf(dot(light_n, w_i), 0);
}

RT_PROGRAM void closestHit() {

    MaterialValue mv = attrib.mv;

    float3 result = mv.ambient + mv.emission;

    Config cf = config[0];
    int N = lightSamples.x;

    int stratify = lightStratify.x;

    float3 x = attrib.intersection;
    float3 n = attrib.normal;
    float3 w_o = attrib.wo;

    for(int i = 0; i < qlights.size(); i++) {
        QuadLight qlight = qlights[i];

        float3 l_d = make_float3(0);
        
        float3 n_l = normalize(cross(qlight.ab, qlight.ac));
        float A = length(cross(qlight.ab, qlight.ac));

        if(stratify) {
            float sqrtN = sqrtf((float) N);

            float3 dAB = qlight.ab / sqrtN;
            float3 dAC = qlight.ac / sqrtN;

            for(int j = 0; j < sqrtN; j++) {
                for(int k = 0; k < sqrtN; k++) {
                    float u1 = reduceVariance.x ? MIDPOINT : rnd(payload.seed);
                    float u2 = reduceVariance.x ? MIDPOINT : rnd(payload.seed);

                    float3 x_prime = qlight.a + (j * dAB) + (k * dAC) + (u1 * dAB) + (u2 * dAC);
                    float3 w_i = normalize(x_prime - x);

                    float R = length(x_prime - x);

                    // Visibility
                    ShadowPayload shadowPayload;
                    shadowPayload.isVisible = true;

                    Ray shadowRay = make_Ray(x + (cf.epsilon * w_i), w_i, 1, cf.epsilon, R - 2 * cf.epsilon);
                    
                    rtTrace(root, shadowRay, shadowPayload);

                    if(shadowPayload.isVisible) {
                        l_d += f(w_i, w_o, n, mv.diffuse, mv.specular, mv.shininess) * G(x, x_prime, n, n_l);
                    }
                }
            }
        } else {
            for(int k = 0; k < N; k++) {
                float u1 = rnd(payload.seed);
                float u2 = rnd(payload.seed);

                float3 x_prime = qlight.a + (u1 * qlight.ab) + (u2 * qlight.ac);
                float3 w_i = normalize(x_prime - x);

                float R = length(x_prime - x);

                // Visibility
                ShadowPayload shadowPayload;
                shadowPayload.isVisible = true;

                Ray shadowRay = make_Ray(x + (cf.epsilon * w_i), w_i, 1, cf.epsilon, R - 2 * cf.epsilon);
                
                rtTrace(root, shadowRay, shadowPayload);

                if(shadowPayload.isVisible) {
                    l_d += f(w_i, w_o, n, mv.diffuse, mv.specular, mv.shininess) * G(x, x_prime, n, n_l);
                }
            }
        }

        l_d *= qlight.intensity * A / N;

        result += l_d;
    }

    payload.radiance = result;
}