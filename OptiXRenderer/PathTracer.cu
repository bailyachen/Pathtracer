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

// Sampling Modes
#define HEMISPHERE 0
#define COSINE 1
#define BRDF 2

#define PHONG 0
#define GGX 1

#define OFF 0
#define ON 1
#define MIS 2

rtBuffer<QuadLight> qlights;

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );

rtDeclareVariable(int1, NEE, , );
rtDeclareVariable(int1, RR, ,);
rtDeclareVariable(int1, importanceSampling, , );

rtBuffer<Config> config; // Config

// Declare attributes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

/*************************************************************
        RANDOM SAMPLING FUNCTIONS
*************************************************************/

float3 random_hemisphere_vector(unsigned int seed, float3 n) {
    float r0 = rnd(seed), r1 = rnd(seed);

    float theta = acos(r0);
    float phi = 2 * PI * r1;

    float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

    float3 w = normalize(n);
    float3 a = make_float3(1, 1, 1);

    float3 u = normalize(cross(a, w));
    float3 v = cross(w, u);

    return (s.x * u) + (s.y * v) + (s.z * w); 
}

float3 random_cosine_vector(unsigned int seed, float3 n) {
    float r0 = rnd(seed), r1 = rnd(seed);

    float theta  = acos(sqrt(r0));
    float phi = 2 * PI * r1;
    
    float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

    float3 w = normalize(n);
    float3 a = make_float3(1, 1, 1);

    float3 u = normalize(cross(a, w));
    float3 v = cross(w, u);

    return (s.x * u) + (s.y * v) + (s.z * w); 
}

float3 random_phong_vector(unsigned int seed, float3 w_o, float3 n, MaterialValue mv) {
    float3 k_s = mv.specular;
    float _k_s = (k_s.x + k_s.y + k_s.z) / 3.0f;
    float3 k_d = mv.diffuse;
    float _k_d = (k_d.x + k_d.y + k_d.z) / 3.0f;
    float t = _k_s / (_k_d + _k_s);

    float r0 = rnd(seed), r1 = rnd(seed), r2 = rnd(seed);

    float phi = 2 * PI * r2;
    float theta;

    float3 u, v, w; // Coordinate Basis

    if(r0 <= t) { // Specular
        float3 r = reflect(-w_o, n);
        w = normalize(r);
        theta = acos(pow(r1, 1 / (mv.shininess + 1)));
    }
    else { // Diffuse
        w = normalize(n);
        theta = acos(sqrt(r1));
    }

    float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

    float3 a = make_float3(1, 1, 1);
    
    u = normalize(cross(a, w));
    v = cross(w, u);

    return (s.x * u) + (s.y * v) + (s.z * w);
}

float3 random_GGX_vector(unsigned int seed, float3 w_o, float3 n, MaterialValue mv) {
    float3 k_s = mv.specular;
    float _k_s = (k_s.x + k_s.y + k_s.z) / 3.0f;
    float3 k_d = mv.diffuse;
    float _k_d = (k_d.x + k_d.y + k_d.z) / 3.0f;
    float t = _k_d == 0 && _k_s == 0 ? 1 : fmaxf(0.25, _k_s / (_k_d + _k_s));

    float r0 = rnd(seed), r1 = rnd(seed), r2 = rnd(seed);

    float3 s;

    float phi = 2 * PI * r2;
    float theta;

    float3 u, v, w; // Coordinate Basis

    if(r0 <= t) { // Specular
        w = normalize(n);
        theta = atan2(mv.roughness * sqrt(r1), sqrt(1 - r1));
        float3 h = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)); // Half Vector

        float3 a = make_float3(1, 1, 1);
    
        u = normalize(cross(a, w));
        v = normalize(cross(w, u));

        h = normalize((h.x * u) + (h.y * v) + (h.z * w));

        s = reflect(-w_o, h);
    } else { // Diffuse
        w = normalize(n);
        theta = acos(sqrtf(r1));
        s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

        float3 a = make_float3(1, 1, 1);
    
        u = normalize(cross(a, w));
        v = cross(w, u);
        s = (s.x * u) + (s.y * v) + (s.z * w);
    }

    return s;
}
/*************************************************************
        PHONG BRDF FUNCTIONS
*************************************************************/

float3 f_phong(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {
    float3 r = reflect(-w_o, n);

    float3 k_d = mv.diffuse;
    float3 k_s = mv.specular;
    float s = mv.shininess;

    return (k_d / PI) + k_s * ((s + 2) / (2 * PI)) * pow(fmaxf(dot(r, w_i), 0), s);
}

float G_phong(float3 x, float3 x_prime, float3 surface_n, float3 light_n) {
    float R = length(x_prime - x);
    float3 w_i = normalize(x_prime - x);
    return (1 / (R * R)) * fmaxf(dot(surface_n, w_i), 0) * fmaxf(dot(light_n, w_i), 0);
}


float PDF_phong(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {

    float3 k_s = mv.specular;
    float _k_s = (k_s.x + k_s.y + k_s.z) / 3.0f;
    float3 k_d = mv.diffuse;
    float _k_d = (k_d.x + k_d.y + k_d.z) / 3.0f;
    float s = mv.shininess;

    float t =_k_d == 0 && _k_s == 0 ? 1 : _k_s / (_k_d + _k_s);

    float3 r = reflect(-w_o, n);

    float result =  ((1 - t) * fmaxf(dot(n, w_i), 0) / PI);
    result += (t * (s + 1) / (2 * PI) * pow(fmaxf(dot(r, w_i), 0), s));

    return result;
}

/*************************************************************
        GGX BRDF FUNCTIONS
*************************************************************/

float power(float x, int n) {
    int zerodir;
    float factor;

    if(n < 0) {
        zerodir = 1;
        factor = 1.0f / x;
    } else {
        zerodir = -1;
        factor = x;
    }

    float result = 1;
    while(n) {
        if(n & 1) { // n % 2 != 0
            result *= factor;
            n += zerodir;
        } else {
            factor *= factor;
            n >>= 1; // n /= 2;
        }
    }
    return result;
}

float3 Fresnel(float3 w_i, float3 h, float3 n, MaterialValue mv) {
    float3 k_s = mv.specular;
    return k_s + (1 - k_s) * power(1 - clamp(dot(h, w_i), 0.0f, 1.0f), 5);
}

float G_1(float3 v, float3 n, MaterialValue mv) {
    if (dot(v, n) > 0) {
        float theta_v = acos(clamp(dot(v, n), 0.0f, 1.0f));
        float alpha = mv.roughness;
        float tan_theta_v = tan(theta_v);

        return 2 / (1 + sqrt(1 + power(alpha, 2) * power(tan_theta_v, 2)));
    }
    return 0;
}

float G_GGX(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {
    return G_1(w_i, n, mv) * G_1(w_o, n, mv);
}

float D(float3 h, float3 n, MaterialValue mv) {
    float alpha_2 = power(mv.roughness, 2);

    float theta_h = acos(clamp(dot(h, n), -1.0f, 1.0f));
    
    float tan_theta_h_2 = power(tan(theta_h), 2);
    float cos_theta_h_4 = power(cos(theta_h), 4);

    float alphatan_2 = power(alpha_2 + tan_theta_h_2, 2);

    float result =  alpha_2 / (PI * cos_theta_h_4 * alphatan_2);

    return result;
}

float3 _f_ggx(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {
    float3 h = normalize(w_i + w_o);

    float n_dot_wi = clamp(dot(n, w_i), 0.0f, 1.0f);
    float n_dot_wo = clamp(dot(n, w_o), 0.0f, 1.0f);

    if(n_dot_wi <= 0 || n_dot_wo <= 0)
        return make_float3(0);

    float3 result = make_float3(1);
    result *= Fresnel(w_i, h, n, mv);
    result *= G_GGX(w_i, w_o, n, mv);
    result *= D(h, n, mv);
    result /= 4 * n_dot_wi * n_dot_wo;

    return result;
}

float3 f_GGX(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {
    float3 k_d = mv.diffuse;
    return k_d / PI + _f_ggx(w_i, w_o, n, mv);
}

float PDF_GGX(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {
    float3 k_s = mv.specular;
    float _k_s = (k_s.x + k_s.y + k_s.z) / 3.0f;
    float3 k_d = mv.diffuse;
    float _k_d = (k_d.x + k_d.y + k_d.z) / 3.0f;

    float t = _k_d + _k_s <= 0 ? 1 : fmaxf(0.25, _k_s / (_k_d + _k_s));
    float3 h = normalize(w_i + w_o);
    float d = D(h, n, mv);
    
    float result = ((1 - t) * clamp(dot(n, w_i), 0.0f, 1.0f) / PI);
    result += t * d * clamp(dot(n, h), 0.0f, 1.0f) / (4 * clamp(dot(h, w_i), 0.0f, 1.0f));

    return result;
}

/*************************************************************
        LIGHTING
*************************************************************/

float3 directLight(float3 x, float3 n, float3 w_o, MaterialValue mv, Config cf) {
    float3 result = make_float3(0);
    for (int i = 0; i < qlights.size(); i++) {
        QuadLight qlight = qlights[i];

        float3 l_d = make_float3(0);

        float3 n_l = normalize(cross(qlight.ab, qlight.ac));
        float A = length(cross(qlight.ab, qlight.ac));

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

        if (shadowPayload.isVisible) {
            float3 f;
            if(mv.BRDFmode == GGX) {
                 f = f_GGX(w_i, w_o, n, mv);
            } else {
                f = f_phong(w_i, w_o, n, mv);
            }
            
            l_d += f * G_phong(x, x_prime, n, n_l);
        }

        l_d *= qlight.intensity * A;

        result += l_d;
    }
    return result;
}

float PDF_BRDF(float3 w_i, float3 w_o, float3 n, MaterialValue mv) {
    switch(importanceSampling.x) {
        case HEMISPHERE:
            return 1 / (2 * PI);
        case BRDF:
            switch(mv.BRDFmode) {
                case GGX:
                    return PDF_GGX(w_i, w_o, n, mv);
                case PHONG:
                default:
                    return PDF_phong(w_i, w_o, n, mv);
            }
        case COSINE:
        default:
            return clamp(dot(n, w_i), 0.0f, 1.0f) / PI;
    }
}

float PDF_NEE(unsigned int seed, float3 x, float3 w_i, Config cf) {
    float pdf = 0;

    for(int i = 0; i < qlights.size(); i++) {
        QuadLight qlight = qlights[i];

        Payload nPayload;
        nPayload.radiance = make_float3(0);
        nPayload.throughput = make_float3(1);
        nPayload.done = true;
        nPayload.depth = 1;
        nPayload.seed = seed;

        Ray ray = make_Ray(x + cf.epsilon * w_i, w_i, 0, cf.epsilon, RT_DEFAULT_MAX);
        
        rtTrace(root, ray, nPayload);

        float3 l_e = nPayload.radiance;
        float pdf_l = 0;

        if(length(l_e) > 0) {
            float3 x_prime = nPayload.intersection;
            float R = length(x_prime - x);
            float A = length(cross(qlight.ab, qlight.ac));
            float3 n_l = normalize(cross(qlight.ab, qlight.ac));

            if(dot(n_l, w_i) > 0) {
                pdf_l = power(R, 2) / (A * dot(n_l, w_i));
            }
        }

        pdf += pdf_l;
    }

    return pdf / qlights.size();
}

float3 direct_BRDF(unsigned int seed, float3 x, float3 w_o, float3 n, MaterialValue mv, Config cf) {
    float3 w_i;
    switch(importanceSampling.x) {
        case HEMISPHERE:
            w_i = random_hemisphere_vector(seed, n);
            break;
        case BRDF:
            switch(mv.BRDFmode) {
                case GGX:
                    w_i = random_GGX_vector(seed, w_o, n, mv);
                    break;
                case PHONG:
                default:
                    w_i = random_phong_vector(seed, w_o, n, mv);
                    break;
            }
            break;
        case COSINE:
        default:
            w_i = random_cosine_vector(seed, n);
            break;
    }

    float3 f;
    switch(mv.BRDFmode) {
        case GGX:
            f = f_GGX(w_i, w_o, n, mv);
            break;
        case PHONG:
        default:
            f = f_phong(w_i, w_o, n, mv);
            break;
    }

    float pdf = PDF_BRDF(w_i, w_o, n, mv);

    if(pdf == 0) return make_float3(0);

    float pdf_BRDF = PDF_BRDF(w_i, w_o, n, mv);
    float pdf_NEE = PDF_NEE(seed, x, w_i, cf);
    float weight = (pdf_BRDF * pdf_BRDF) / (pdf_BRDF * pdf_BRDF + pdf_NEE * pdf_NEE);

    Payload nPayload;
    nPayload.radiance = make_float3(0);

    nPayload.throughput = weight * f * clamp(dot(n, w_i), 0.0f, 1.0f) / pdf;
    nPayload.done = true;
    nPayload.depth = 1;
    nPayload.seed = seed;

    Ray ray = make_Ray(x + cf.epsilon * w_i, w_i, 0, cf.epsilon, RT_DEFAULT_MAX);
    rtTrace(root, ray, nPayload);

    return nPayload.radiance;
}

float3 direct_NEE(unsigned int seed, float3 x, float3 w_o, float3 n, MaterialValue mv, Config cf) {
    float3 result = make_float3(0);

    for(int i = 0; i < qlights.size(); i++) {
        QuadLight qlight = qlights[i];

        float u1 = rnd(seed), u2 = rnd(seed);
        float3 x_prime = qlight.a + (u1 * qlight.ab) + (u2 * qlight.ac);
        float3 w_i = normalize(x_prime - x);
        float R = length(x_prime - x);

        float3 f = make_float3(0);

        float pdf_l = 0;

        ShadowPayload shadowPayload;
        shadowPayload.isVisible = true;

        Ray shadowRay = make_Ray(x + cf.epsilon * w_i, w_i, 1, cf.epsilon, R - (2 * cf.epsilon));
        rtTrace(root, shadowRay, shadowPayload);

        float3 l_d = make_float3(0);

        if(shadowPayload.isVisible) {
            switch(mv.BRDFmode) {
                case GGX:
                    f = f_GGX(w_i, w_o, n, mv);
                    break;
                case PHONG:
                default:
                    f = f_phong(w_i, w_o, n, mv);
                    break;
            }

            float A = length(cross(qlight.ab, qlight.ac));
            float3 n_l = normalize(cross(qlight.ab, qlight.ac));

            if(dot(n_l, w_i) > 0) {
                pdf_l = power(R, 2) / (A * dot(n_l, w_i));
            } else {
                pdf_l = 0;
            }
        }

        float pdf_BRDF = PDF_BRDF(w_i, w_o, n, mv);
        float pdf_NEE = PDF_NEE(seed, x, w_i, cf);
        float weight = pdf_NEE * pdf_NEE / (pdf_BRDF * pdf_BRDF + pdf_NEE * pdf_NEE);

        if(pdf_l != 0) {
            l_d = weight * qlight.intensity * f * clamp(dot(n, w_i), 0.0f, 1.0f) / pdf_l;
            result += l_d;
        }
    }

    return result;
}

RT_PROGRAM void closestHit() {

    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = make_float3(0);

    float3 x = attrib.intersection;
    float3 n = attrib.normal;
    float3 w_o = attrib.wo;

    bool isEmissive = length(mv.emission) > 0;
    // NEE
    if(NEE.x == ON) {
        if(isEmissive) {
            payload.done = true;

            if (payload.depth == 0)
                payload.radiance = mv.emission;

            return;
        }

        float3 direct = directLight(x, n, w_o, mv, cf);

        payload.radiance += direct * payload.throughput;
    } else if (NEE.x == MIS) {
        if(payload.done) {
            if(dot(w_o, n) > 0) {
                payload.radiance = mv.emission * payload.throughput;

            } else {
                payload.radiance = make_float3(0);
            }
            payload.intersection = x;
            return;
        }
        if(payload.depth == 0) {
            if(dot(n, w_o) < 0) {
                result += mv.emission;
            }
        }
        if(isEmissive) {
            payload.radiance = result * payload.throughput;
            payload.done = true;
            return;
        }

        result += direct_BRDF(payload.seed, x, w_o, n, mv, cf);
        result += direct_NEE(payload.seed, x, w_o, n, mv, cf);

        payload.radiance = result * payload.throughput;
    } else {
        if(fmaxf(dot(n, w_o), 0) > 0)
            result += mv.emission;

        if(isEmissive) {
            payload.radiance = result * payload.throughput;
            payload.done = true;
            return;
        }
    }

    // RR
    if(RR.x) {
        float q = 1.f - fminf(fmaxf(payload.throughput), 1.f); // Probability of Termination
        float r = rnd(payload.seed);

        if(r <= q) {
            payload.done = true;
        } else {
            payload.throughput *= 1.f / (1.f - q);
        }
    }

    float3 w_i = make_float3(0, 1, 0); // Always UP
    float pdf = 0;

    switch(importanceSampling.x) {
        case HEMISPHERE:
            w_i = random_hemisphere_vector(payload.seed, n);
            payload.throughput *= 2 * PI * f_phong(w_i, w_o, n, mv) * fmaxf(dot(n, w_i), 0);
            break;
        case BRDF:
            switch(mv.BRDFmode) {
                case GGX:
                    w_i = random_GGX_vector(payload.seed, w_o, n, mv);
                    pdf = PDF_GGX(w_i, w_o, n, mv);
                    if(pdf == 0) {
                        payload.radiance = make_float3(0);
                        payload.done = true;
                    } else {
                        payload.throughput *= f_GGX(w_i, w_o, n, mv) * fmaxf(dot(n, w_i), 0) / pdf;
                    }
                    break;
                case PHONG:
                default:
                    w_i = random_phong_vector(payload.seed, w_o, n, mv);
                    payload.throughput *= f_phong(w_i, w_o, n, mv) * fmaxf(dot(n, w_i), 0) / PDF_phong(w_i, w_o, n, mv);
                    break;
            }
            break;
        case COSINE:
        default:
            w_i = random_cosine_vector(payload.seed, n);
            payload.throughput *= PI * f_phong(w_i, w_o, n, mv);
            break;
    }

    payload.origin = x;
    payload.dir = w_i; 
    payload.depth++;
}
