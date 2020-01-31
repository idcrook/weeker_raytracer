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

#include "texture/ioTexture.h"
#include "material/ioNormalMaterial.h"
#include "material/ioLambertianMaterial.h"
#include "material/ioMetalMaterial.h"
#include "material/ioDielectricMaterial.h"

#include "scene/ioCamera.h"

#include "random.cuh"

class ioScene
{
public:
    ioScene() { }

    void init(optix::Context& context) {
        ioTexture *nullTexture = new ioNullTexture();
        ioTexture *constantGrey = new ioConstantTexture(make_float3(0.7f, 0.7f, 0.7f));
        ioTexture *constantGreen = new ioConstantTexture(make_float3(0.2f, 0.3f, 0.1f));
        ioTexture *constantPurple = new ioConstantTexture(make_float3(0.4f, 0.2f, 0.9f));
        ioTexture *saddleBrown = new ioConstantTexture(make_float3(139/255.f,  69/255.f,  19/255.f));
        ioTexture *ivory =       new ioConstantTexture(make_float3(255/255.f, 255/255.f, 240/255.f));
        ioTexture *reallyDarkGrey = new ioConstantTexture(make_float3( 9/255.f,  9/255.f,  9/255.f));
        ioTexture *black = new ioConstantTexture(make_float3( 0/255.f,  0/255.f,  0/255.f));

        ioTexture *checkered = new ioCheckerTexture(reallyDarkGrey, saddleBrown);
        //ioTexture *checkered = new ioCheckerTexture(black, saddleBrown);

        // Big Sphere
        geometryList.push_back(new ioSphere(0.0f, -1000.0f, 0.0, 1000.0f));
        materialList.push_back(new ioLambertianMaterial(0.5f, 0.5f, 0.5f));
        textureList.push_back(checkered);

        // Medium Spheres
        geometryList.push_back(new ioSphere(-4.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(0.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(4.0f, 1.0f, 0.0, 1.0f));

        materialList.push_back(new ioLambertianMaterial(0.4f, 0.2f, 0.2f));
        materialList.push_back(new ioDielectricMaterial(1.5f));
        materialList.push_back(new ioMetalMaterial(0.7f, 0.6f, 0.5f, 0.2f));

        textureList.push_back(constantPurple);
        textureList.push_back(nullptr);
        textureList.push_back(constantGrey);


        // Small Spheres
        uint32_t seed = 0x314759;
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
                        materialList.push_back(new ioLambertianMaterial(1.f, 1.f, 1.f));
                        textureList.push_back(new ioConstantTexture(
                                                  make_float3(randf(seed), randf(seed), randf(seed))));

                    }
                    else if (chooseMat < 0.85f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioMetalMaterial(
                            0.5f, 0.5f, 0.5f, 0.5f*randf(seed))
                            );
                        textureList.push_back(new ioConstantTexture(
                                                  make_float3(0.5f*(1.0f-randf(seed)),
                                                              0.5f*(1.0f-randf(seed)),
                                                              0.5f*(1.0f-randf(seed))))
                                                  );

                    }
                    else if (chooseMat < 0.93f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                        textureList.push_back(nullptr);
                    }
                    else
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                        textureList.push_back(nullptr);
                        geometryList.push_back(new ioSphere(x,y,z, -(0.2f-0.007f)));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                        textureList.push_back(nullptr);
                    }
                }
            }
        }

        // init all geometry
        for(int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
            // init all materials
            for(int i = 0; i < geometryList.size(); i++)
                materialList[i]->init(context);
        }
        // GeometryInstance
        geoInstList.resize(geometryList.size());
        // Taking advantage of geometryList.size == materialList.size == textureList.size
        for (int i = 0; i < geoInstList.size(); i++)
        {
            geoInstList[i] = ioGeometryInstance();
            geoInstList[i].init(context);
            geoInstList[i].setGeometry(*geometryList[i]);
            geoInstList[i].setMaterial(*materialList[i]);
            if (textureList[i]) {
                textureList[i]->assignTo(geoInstList[i].get(), context);
            } else {
                nullTexture->assignTo(geoInstList[i].get(), context);
            }
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
            20.0f, 2.0f
            );
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
    std::vector<ioTexture*>  textureList;
    std::vector<ioGeometryInstance> geoInstList;
    ioGeometryGroup geometryGroup;

    ioCamera* camera;
};

#endif //!IO_SCENE_H
