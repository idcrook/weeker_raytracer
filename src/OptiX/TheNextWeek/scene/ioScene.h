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
#include "geometry/ioAABox.h"

#include "texture/ioTexture.h"
#include "material/ioNormalMaterial.h"
#include "material/ioLambertianMaterial.h"
#include "material/ioMetalMaterial.h"
#include "material/ioDielectricMaterial.h"
#include "material/ioDiffuseLightMaterial.h"

#include "scene/ioCamera.h"

// needed for randf()
#include "../lib/random.cuh"

class ioScene
{
public:
    ioScene() { }

    int init(optix::Context& context, int Nx, int Ny, int Nscene) {

        optix::GeometryGroup world;

        switch(Nscene){
        case 0:
            // Nx = Ny = 1080;
            world = CornellBox(context, Nx, Ny);
            break;
        case 1:
            world = MovingSpheres(context, Nx, Ny);
            break;
        case 2:
            world = InOneWeekend(context, Nx, Ny);
            break;
        case 3:
            world = InOneWeekendLight(context, Nx, Ny);
            break;
        default:
            printf("Error: scene unknown.\n");
            //system("PAUSE");
            return 1;
        }

        // created in each scene
        camera->init(context);

        // Setting World Variable
        context["sysWorld"]->set(world);
    }

    optix::GeometryGroup InOneWeekend(optix::Context& context, int Nx, int Ny)  {}
    optix::GeometryGroup MovingSpheres(optix::Context& context, int Nx, int Ny) {}

    optix::GeometryGroup InOneWeekendLight(optix::Context& context, int Nx, int Ny)  {

        sceneDescription = "IOW Scene with a light box";

        //ioTexture *fiftyPercentGrey = new ioConstantTexture(make_float3(0.5f, 0.5f, 0.5f));
        ioTexture *constantGrey = new ioConstantTexture(make_float3(0.7f, 0.7f, 0.7f));
        // ioTexture *constantGreen = new ioConstantTexture(make_float3(0.2f, 0.3f, 0.1f));
        // ioTexture *constantPurple = new ioConstantTexture(make_float3(0.4f, 0.2f, 0.9f));
        ioTexture *saddleBrown = new ioConstantTexture(make_float3(139/255.f,  69/255.f,  19/255.f));
        ioTexture *reallyDarkGrey = new ioConstantTexture(make_float3( 9/255.f,  9/255.f,  9/255.f));
        ioTexture *black = new ioConstantTexture(make_float3( 0/255.f,  0/255.f,  0/255.f));

        ioTexture *checkered = new ioCheckerTexture(reallyDarkGrey, saddleBrown);

        //ioTexture *noise4 = new ioNoiseTexture(4.f);

        ioTexture *earthGlobeImage = new ioImageTexture("assets/earthmap.jpg");

        //ioTexture* light4 =  new ioConstantTexture(make_float3(4.f, 4.f, 4.f));
        ioTexture* light8 =  new ioConstantTexture(make_float3(8.f, 8.f, 8.f));
        //ioTexture* light12 =  new ioConstantTexture(make_float3(12.f, 12.f, 12.f));

        // Big Sphere
        geometryList.push_back(new ioSphere(0.0f, -1000.0f, 0.0, 1000.0f));
        //materialList.push_back(new ioLambertianMaterial(noise4));
        materialList.push_back(new ioLambertianMaterial(checkered));

        // Medium Spheres
        geometryList.push_back(new ioSphere(-4.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(0.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(4.0f, 1.0f, 0.0, 1.0f));

        materialList.push_back(new ioMetalMaterial(constantGrey, 0.4f));
        materialList.push_back(new ioLambertianMaterial(earthGlobeImage));
        materialList.push_back(new ioDielectricMaterial(1.5f));
        //materialList.push_back(new ioLambertianMaterial(noise2));

        geometryList.push_back(new ioAARect(3.f, 5.f, 0.8f+0.2f, 2.8f+0.7f, -2.f, false, Z_AXIS));
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

        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            13.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            20.0f, float(Nx) / float(Ny),
            /*aperture*/0.12f,
            /*focus_distance*/10.f
            );

        // Have a light in scene
        context["skyLight"]->setInt(false);

        return geometryGroup.get();

    }

    optix::GeometryGroup CornellBox(optix::Context& context, int Nx, int Ny) {

        sceneDescription = "Cornell box";

        ioMaterial *wallRed = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.65f, 0.05f, 0.05f)));
        ioMaterial *wallGreen = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.12f, 0.45f, 0.15f)));
        ioMaterial *wallWhite = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.73f, 0.73f, 0.73f)));
        ioTexture* light15 =  new ioConstantTexture(make_float3(15.f, 15.f, 15.f));

        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f, true, X_AXIS)); // left wall
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 0.f, false, X_AXIS));  // right wall
        geometryList.push_back(new ioAARect(213.f, 343.f, 227.f, 332.f, 554.f, false, Y_AXIS)); // light
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f, true, Y_AXIS)); // roof
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 0.f, false, Y_AXIS));  //floor
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f, true, Z_AXIS)); // back wall

        materialList.push_back(wallGreen);
        materialList.push_back(wallRed);
        materialList.push_back(new ioDiffuseLightMaterial(light15));
        materialList.push_back(wallWhite);
        materialList.push_back(wallWhite);
        materialList.push_back(wallWhite);

        // place some objects in the box

        //  Sphere
        //geometryList.push_back(new ioSphere(265.f, 165.f, 295.f, 165.f));
        geometryList.push_back(new ioSphere(185.f, 75.f, 155.f, 75.f));
        materialList.push_back(wallWhite);

        // bigger Sphere
        geometryList.push_back(new ioSphere(365.f, 165.f, 295.f, 165.f));
        materialList.push_back(wallWhite);

        // box ioAABox not working
        // float3 p0 = make_float3(265.f, 0.f, 295.f);
        // float3 p1 = make_float3(165.f, 330.f, 165.f);
        // geometryList.push_back(new ioAABox(p0, p1));
        // materialList.push_back(wallWhite);

        uint32_t seed = 0x6314759;

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

        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            278.f, 278.f, -800.f,
            278.f, 278.f, 0.f,
            0.0f, 1.0f, 0.0f,
            40.0f, float(Nx) / float(Ny),
            /*aperture*/0.f,
            /*focus_distance*/10.f
            );

        //context["skyLight"]->setInt(true);
        context["skyLight"]->setInt(false);

        return geometryGroup.get();
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
    std::vector<ioGeometryInstance> geoInstList;
    ioGeometryGroup geometryGroup;
    ioCamera* camera;
    //optix::GeometryGroup world;
    std::string sceneDescription;
};

#endif //!IO_SCENE_H
