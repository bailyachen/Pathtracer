#include "SceneLoader.h"

#define HEMISPHERE 0
#define COSINE 1
#define BRDF 2

#define PHONG 0
#define GGX 1

void SceneLoader::rightMultiply(const optix::Matrix4x4& M)
{
    optix::Matrix4x4& T = transStack.top();
    T = T * M;
}

optix::float3 SceneLoader::transformPoint(optix::float3 v)
{
    optix::float4 vh = transStack.top() * optix::make_float4(v, 1);
    return optix::make_float3(vh) / vh.w;
}

optix::float3 SceneLoader::transformNormal(optix::float3 n)
{
    return optix::make_float3(transStack.top() * make_float4(n, 0));
}

template <class T>
bool SceneLoader::readValues(std::stringstream& s, const int numvals, T* values)
{
    for (int i = 0; i < numvals; i++)
    {
        s >> values[i];
        if (s.fail())
        {
            std::cout << "Failed reading value " << i << " will skip" << std::endl;
            return false;
        }
    }
    return true;
}


std::shared_ptr<Scene> SceneLoader::load(std::string sceneFilename)
{

    // Attempt to open the scene file
    std::ifstream in(sceneFilename);
    if (!in.is_open())
    {
        // Unable to open the file. Check if the filename is correct.
        throw std::runtime_error("Unable to open scene file " + sceneFilename);
    }

    auto scene = std::make_shared<Scene>();
    Config& config = scene->config;
    scene->config.gamma = 1;

    MaterialValue mv, defaultMv;
    mv.ambient = optix::make_float3(0);
    mv.diffuse = optix::make_float3(0);
    mv.specular = optix::make_float3(0);
    mv.emission = optix::make_float3(0);
    mv.shininess = 1;
    mv.roughness = 1;
    mv.BRDFmode = PHONG;
    defaultMv = mv;

    optix::float3 attenuation = optix::make_float3(1, 0, 0);

    transStack.push(optix::Matrix4x4::identity());

    std::string str, cmd;

    // Read a line in the scene file in each iteration
    while (std::getline(in, str))
    {
        // Ruled out comment and blank lines
        if ((str.find_first_not_of(" \t\r\n") == std::string::npos) || (str[0] == '#'))
		{
			continue;
		}

        // Read a command
        std::stringstream s(str);
        s >> cmd;

        // Some arrays for storing values
        float fvalues[12];
        int ivalues[3];
        std::string svalues[1];

		if (cmd == "size" && readValues(s, 2, fvalues))
        {
			scene->width = (unsigned int)fvalues[0];
			scene->height = (unsigned int)fvalues[1];
			config.hSize = optix::make_float2(fvalues[0] / 2.f, fvalues[1] / 2.f);
        }
		else if (cmd == "maxdepth" && readValues(s, 1, fvalues))
        {
			config.maxDepth = (unsigned int)fvalues[0];
        }
		else if (cmd == "output" && readValues(s, 1, svalues))
        {
            scene->outputFilename = svalues[0];
        }
		else if (cmd == "integrator" && readValues(s, 1, svalues))
        {
            scene->integratorName = svalues[0];
        }
		else if (cmd == "camera" && readValues(s, 10, fvalues))
        {
            optix::float3 eye = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
            optix::float3 center = optix::make_float3(fvalues[3], fvalues[4], fvalues[5]);
            optix::float3 up = optix::make_float3(fvalues[6], fvalues[7], fvalues[8]);
            float fovy = fvalues[9];

            config.eye = eye;
            config.w = optix::normalize(eye - center);
            config.u = optix::normalize(optix::cross(up, config.w));
            config.v = optix::normalize(optix::cross(config.w, config.u));
            config.tanHFov.y = tanf(fovy / 180.f * M_PIf * 0.5f);
            config.tanHFov.x = config.tanHFov.y * config.hSize.x / config.hSize.y;
        }
		else if (cmd == "sphere" && readValues(s, 4, fvalues))
        {
            Sphere sphere;
            sphere.trans = transStack.top();
            sphere.trans *= optix::Matrix4x4::translate(
                optix::make_float3(fvalues[0], fvalues[1], fvalues[2]));
            sphere.trans *= optix::Matrix4x4::scale(
                optix::make_float3(fvalues[3]));
            sphere.mv = mv;
            scene->spheres.push_back(sphere);
        }
		else if (cmd == "maxverts" && readValues(s, 1, fvalues))
        {

        }
		else if (cmd == "vertex" && readValues(s, 3, fvalues))
        {
            scene->vertices.push_back(
                optix::make_float3(fvalues[0], fvalues[1], fvalues[2]));
        }
		else if (cmd == "tri" && readValues(s, 3, ivalues))
        {
            Triangle tri;
            optix::Matrix4x4 trans = transStack.top();
            tri.v1 = transformPoint(scene->vertices[ivalues[0]]);
            tri.v2 = transformPoint(scene->vertices[ivalues[1]]);
            tri.v3 = transformPoint(scene->vertices[ivalues[2]]);
            tri.normal = optix::normalize(cross(tri.v2 - tri.v1, tri.v3 - tri.v1));
            tri.mv = mv;
            scene->triangles.push_back(tri);
        }
		else if (cmd == "translate" && readValues(s, 3, fvalues))
        {
            optix::Matrix4x4 T = optix::Matrix4x4::translate(
                optix::make_float3(fvalues[0], fvalues[1], fvalues[2]));
            rightMultiply(T);
        }
		else if (cmd == "scale" && readValues(s, 3, fvalues))
        {
            optix::Matrix4x4 S = optix::Matrix4x4::scale(
                optix::make_float3(fvalues[0], fvalues[1], fvalues[2]));
            rightMultiply(S);
        }
		else if (cmd == "rotate" && readValues(s, 4, fvalues))
        {
            float radians = fvalues[3] * M_PIf / 180.f;
            optix::Matrix4x4 R = optix::Matrix4x4::rotate(
                radians, optix::make_float3(fvalues[0], fvalues[1], fvalues[2]));
            rightMultiply(R);
        }
		else if (cmd == "pushTransform")
        {
            transStack.push(transStack.top());
        }
		else if (cmd == "popTransform")
        {
            if (transStack.size() <= 1)
            {
                std::cerr << "Stack has no elements. Cannot pop" << std::endl;
            }
            else
            {
                transStack.pop();
            }
        }
		else if (cmd == "attenuation" && readValues(s, 3, fvalues))
        {
            attenuation = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
        }
		else if (cmd == "ambient" && readValues(s, 3, fvalues))
        {
            mv.ambient = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
        }
		else if (cmd == "diffuse" && readValues(s, 3, fvalues))
        {
            mv.diffuse = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
        }
		else if (cmd == "specular" && readValues(s, 3, fvalues))
        {
            mv.specular = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
        }
		else if (cmd == "emission" && readValues(s, 3, fvalues))
        {
            mv.emission = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
        }
		else if (cmd == "shininess" && readValues(s, 1, fvalues))
        {
            mv.shininess = fvalues[0];
        }
        else if(cmd == "brdf" && readValues(s, 1, svalues))
        {
            mv.BRDFmode = svalues[0] == "ggx" ? GGX : PHONG;
        }
        else if(cmd == "roughness" && readValues(s, 1, fvalues))
        {
            mv.roughness = fvalues[0];
        }
		else if (cmd == "directional" && readValues(s, 6, fvalues))
        {
            DirectionalLight dlight;
            dlight.direction = optix::normalize(
                optix::make_float3(fvalues[0], fvalues[1], fvalues[2]));
            dlight.color = optix::make_float3(fvalues[3], fvalues[4], fvalues[5]);
            scene->dlights.push_back(dlight);
        }
		else if (cmd == "point" && readValues(s, 6, fvalues))
        {
            PointLight plight;
            plight.location = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
            plight.color = optix::make_float3(fvalues[3], fvalues[4], fvalues[5]);
            plight.attenuation = attenuation;
            scene->plights.push_back(plight);
        }
		else if (cmd == "quadLight" && readValues(s, 12, fvalues))
        {
            QuadLight qlight;
            qlight.a = optix::make_float3(fvalues[0], fvalues[1], fvalues[2]);
            qlight.ab = optix::make_float3(fvalues[3], fvalues[4], fvalues[5]);
            qlight.ac = optix::make_float3(fvalues[6], fvalues[7], fvalues[8]);
            qlight.intensity = optix::make_float3(fvalues[9], fvalues[10], fvalues[11]);
            scene->qlights.push_back(qlight);

            Triangle t1, t2;

            optix::float3 A = qlight.a;
            optix::float3 B  = qlight.a + qlight.ab;
            optix::float3 C  = qlight.a + qlight.ac;
            optix::float3 D  = qlight.a + qlight.ab + qlight.ac;

            t1.v1 = transformPoint(A);
            t1.v2 = transformPoint(D);
            t1.v3 = transformPoint(C);
            t1.normal = optix::normalize(optix::cross(t1.v2 - t1.v1, t1.v3 - t1.v1));
            t1.mv = defaultMv;
            t1.mv.emission = qlight.intensity;

            scene->triangles.push_back(t1);

            t2.v1 = transformPoint(A);
            t2.v2 = transformPoint(B);
            t2.v3 = transformPoint(D);
            t2.normal = optix::normalize(optix::cross(t1.v2 - t1.v1, t1.v3 - t1.v1));
            t2.mv = defaultMv;
            t2.mv.emission = qlight.intensity;

            scene->triangles.push_back(t2);
        }
		else if (cmd == "lightsamples" && readValues(s, 1, ivalues))
        {
            scene->lightSamples = ivalues[0];
        }
		else if (cmd == "lightstratify" && readValues(s, 1, svalues))
        {
            scene->lightStratify = svalues[0] == "on";
        }
		else if (cmd == "reducevariance" && readValues(s, 1, svalues))
		{
            scene->reduceVariance = svalues[0] == "on";
        }
		else if(cmd == "spp" && readValues(s, 1, ivalues))
		{
			scene->samplesPerPixel = ivalues[0];
		}
        else if(cmd == "nexteventestimation" && readValues(s, 1, svalues)) {
            if(svalues[0] == "mis") {
                scene->NEE = 2;
            } else if(svalues[0] == "on") {
                scene->NEE = 1;
            } else {
                scene->NEE = 0;
            }
            std::cerr << "Set NEE to " << scene->NEE << " for " << svalues[0] << std::endl;
        }
        else if(cmd == "russianroulette" && readValues(s, 1, svalues)) {
            scene->RR = svalues[0] == "on";
        }
        else if(cmd == "importancesampling" && readValues(s, 1, svalues)) {
            if (svalues[0] == "cosine") {
                scene->importanceSampling = COSINE;
            } 
            else if(svalues[0] == "brdf") {
                scene->importanceSampling = BRDF;
            } else {
                scene->importanceSampling = HEMISPHERE;
            }
        }
        else if(cmd == "gamma" && readValues(s, 1, fvalues)) {
            scene->config.gamma = fvalues[0];
        }
    }

    in.close();

    return scene;
}
