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

#include "random.cuh"

class ioScene
{
public:
    ioScene() { }

    void init(optix::Context& context, int Nx, int Ny) {

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

        // box
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

        // Setting World Variable
        context["sysWorld"]->set(geometryGroup.get());

        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            278.f, 278.f, -800.f,
            278.f, 278.f, 0.f,
            0.0f, 1.0f, 0.0f,
            40.0f, float(Nx) / float(Ny),
            /*aperture*/0.f,
            /*focus_distance*/10.f
            );

        // camera = new ioOrthographicCamera(
        //     13.0f, 2.0f, 3.0f,
        //     -1.2f, 0.0f, 0.0f,
        //     0.0f, 1.0f, 0.0f,
        //     3.5f, 7.f
        //     );

        camera->init(context);

        //context["skyLight"]->setInt(true);
        context["skyLight"]->setInt(false);
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
