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

rtBuffer<QuadLight> qlights;

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void closestHit() {

    float3 result = attrib.mv.ambient + attrib.mv.emission;
    float3 r = attrib.intersection;

    // Diffuse Albedo
    float3 f = attrib.mv.diffuse / PI;

    float3 E = make_float3(0, 0, 0);

    for(int i = 0; i < qlights.size(); i++) {

        float3 v[] = { 
            qlights[i].a,                                 // A
            qlights[i].a + qlights[i].ab,                 // B
            qlights[i].a + qlights[i].ab + qlights[i].ac, // D
            qlights[i].a + qlights[i].ac                  // C
        };
        

        float3 irradiance = make_float3(0, 0, 0);
        

        for(int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            float Theta_k = acos(dot(normalize(v[k] - r), normalize(v[next] - r)));
            float3 Gamma_k = normalize(cross(v[k] - r, v[next] - r));

            irradiance += Gamma_k * Theta_k;
        }

        irradiance *= 0.5;

        E += qlights[i].intensity * dot(irradiance, attrib.normal);
    }


    result +=  f * E;
    payload.radiance = result;
}