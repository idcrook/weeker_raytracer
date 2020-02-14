#ifndef IO_SCENE_H
#define IO_SCENE_H

#include <vector>
#include <iostream>

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometry.h"
#include "geometry/ioGeometryInstance.h"
#include "geometry/ioGeometryGroup.h"

#include "geometry/ioSphere.h"
#include "geometry/ioAARect.h"

#include "texture/ioTexture.h"
#include "material/ioNormalMaterial.h"
#include "material/ioLambertianMaterial.h"
#include "material/ioMetalMaterial.h"
#include "material/ioDielectricMaterial.h"
#include "material/ioDiffuseLightMaterial.h"

#include "scene/ioCamera.h"

#include "random.cuh"

class ioScene
{
public:
    ioScene() { }

    void init(optix::Context& context, int Nx, int Ny) {

        //context["skyLight"]->setInt(true);
        context["skyLight"]->setInt(false);

        ioTexture *nullTexture = new ioNullTexture();
        ioTexture *fiftyPercentGrey = new ioConstantTexture(make_float3(0.5f, 0.5f, 0.5f));
        ioTexture *constantGrey = new ioConstantTexture(make_float3(0.7f, 0.7f, 0.7f));
        ioTexture *constantGreen = new ioConstantTexture(make_float3(0.2f, 0.3f, 0.1f));
        ioTexture *constantPurple = new ioConstantTexture(make_float3(0.4f, 0.2f, 0.9f));
        ioTexture *saddleBrown = new ioConstantTexture(make_float3(139/255.f,  69/255.f,  19/255.f));
        ioTexture *ivory =       new ioConstantTexture(make_float3(255/255.f, 255/255.f, 240/255.f));
        ioTexture *reallyDarkGrey = new ioConstantTexture(make_float3( 9/255.f,  9/255.f,  9/255.f));
        ioTexture *black = new ioConstantTexture(make_float3( 0/255.f,  0/255.f,  0/255.f));

        ioTexture *checkered = new ioCheckerTexture(reallyDarkGrey, saddleBrown);
        //ioTexture *checkered = new ioCheckerTexture(constantGrey, saddleBrown);

        ioTexture *noise2 = new ioNoiseTexture(2.f);
        ioTexture *noise4 = new ioNoiseTexture(4.f);

        ioTexture *earthGlobeImage = new ioImageTexture("assets/earthmap.jpg");

        ioTexture* light4 =  new ioConstantTexture(make_float3(4.f, 4.f, 4.f));
        ioTexture* light8 =  new ioConstantTexture(make_float3(8.f, 8.f, 8.f));

        // Big Sphere
        geometryList.push_back(new ioSphere(0.0f, -1000.0f, 0.0, 1000.0f));
        //materialList.push_back(new ioLambertianMaterial(noise2));
        materialList.push_back(new ioLambertianMaterial(checkered));

        // Medium Spheres
        geometryList.push_back(new ioSphere(-4.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(0.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(4.0f, 1.0f, 0.0, 1.0f));

        materialList.push_back(new ioMetalMaterial(constantGrey, 0.4f));
        materialList.push_back(new ioDielectricMaterial(1.5f));
        materialList.push_back(new ioLambertianMaterial(earthGlobeImage));
        //materialList.push_back(new ioLambertianMaterial(noise2));

        geometryList.push_back(new ioAARect(3.f, 5.f, 0.5f, 2.5f, -2.f, false, Z_AXIS));
        materialList.push_back(new ioDiffuseLightMaterial(light8));


        // Small Spheres
        uint32_t seed = 0x6314759;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float chooseMat = randf(seed);
                float x = a + 0.8f*randf(seed);
                float y = 0.2f;
                float z = b + 0.9f*randf(seed);
                float z_squared = (z)*(z);
                float dist = sqrtf(
                    (x-4.0f)*(x-4.0f) +
                    //(y-0.2f)*(y-0.2f) +
                    z_squared
                    );

                // keep out area near medium spheres
                if ((dist > 0.9f) ||
                    ((z_squared > 0.7f) && ((x*x - 16.0f) > -2.f)))
                {
                    if (chooseMat < 0.70f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioLambertianMaterial(
                                                   new ioConstantTexture(make_float3(randf(seed), randf(seed), randf(seed))
                                                       )));
                    }
                    else if (chooseMat < 0.85f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioMetalMaterial(
                                                   new ioConstantTexture(make_float3(0.5f*(1.0f-randf(seed)),
                                                                                     0.5f*(1.0f-randf(seed)),
                                                                                     0.5f*(1.0f-randf(seed)))),
                                                   0.5f*randf(seed)));
                    }
                    else if (chooseMat < 0.93f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                    }
                    else
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                        geometryList.push_back(new ioSphere(x,y,z, -(0.2f-0.007f)));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                    }
                }
            }
        }

        // init all geometry
        for(int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }
        // GeometryInstance
        geoInstList.resize(geometryList.size());

        // Taking advantage of geometryList.size == materialList.size
        for (int i = 0; i < geoInstList.size(); i++)
        {
            //std::cerr << i << std::endl;
            geoInstList[i] = ioGeometryInstance();
            geoInstList[i].init(context);
            geoInstList[i].setGeometry(*geometryList[i]);
            materialList[i]->assignTo(geoInstList[i].get(), context);
        }

        // World & Acceleration
        geometryGroup.init(context);  // init() sets acceleration
        for (int i = 0; i < geoInstList.size(); i++)
            geometryGroup.addChild(geoInstList[i]);

        // Setting World Variable
        context["sysWorld"]->set(geometryGroup.get());

        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            13.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            20.0f, float(Nx) / float(Ny),
            /*aperture*/0.12f,
            /*focus_distance*/10.f
            );

        // camera = new ioOrthographicCamera(
        //     13.0f, 2.0f, 3.0f,
        //     -1.2f, 0.0f, 0.0f,
        //     0.0f, 1.0f, 0.0f,
        //     3.5f, 7.f
        //     );

        // camera = new ioEnvironmentCamera(
        //     -1.0f, 2.0f, 1.0f,
        //     0.0f, 0.0f, -2.0f,
        //     0.0f, 1.0f, 0.0f
        //     );

        camera->init(context);
    }

    void destroy()
        {
            for(int i = 0; i < materialList.size(); i++)
                materialList[i]->destroy();

            for(int i = 0; i < geometryList.size(); i++)
                geometryList[i]->destroy();

            for(int i = 0; i < geoInstList.size(); i++)
                geoInstList[i].destroy();

            geometryGroup.destroy();

            camera->destroy();
            delete camera;
        }

public:
    std::vector<ioMaterial*> materialList;
    std::vector<ioGeometry*> geometryList;
    //std::vector<ioTexture*>  textureList;
    std::vector<ioGeometryInstance> geoInstList;
    ioGeometryGroup geometryGroup;

    ioCamera* camera;
};

#endif //!IO_SCENE_H
