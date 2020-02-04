#ifndef IO_SCENE_H
#define IO_SCENE_H

#include <vector>

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometry.h"
#include "geometry/ioGeometryInstance.h"
#include "geometry/ioGeometryGroup.h"

#include "geometry/ioSphere.h"
#include "material/ioNormalMaterial.h"
#include "material/ioLambertianMaterial.h"
#include "material/ioMetalMaterial.h"
#include "material/ioDielectricMaterial.h"

#include "scene/ioCamera.h"

class ioScene
{
public:
  ioScene() { }

  void init(optix::Context& context)
    {
      // Base Sphere
      geometryList.push_back(new ioSphere(0.0f, -100.5f, -1.0f, 100.0f));
      materialList.push_back(new ioLambertianMaterial(0.8f, 0.8f, 0.0f));

      geometryList.push_back(new ioSphere(0.0f, 0.0f, -1.0f, 0.5f));
      materialList.push_back(new ioLambertianMaterial(0.1f, 0.2f, 0.5f));

      geometryList.push_back(new ioSphere(1.0f, 0.0f, -1.0f, 0.5f));
      materialList.push_back(new ioMetalMaterial(0.8f, 0.6f, 0.2f, 0.5f));

      geometryList.push_back(new ioSphere(-1.0f, 0.0f, -1.0f, 0.5f));
      materialList.push_back(new ioDielectricMaterial(1.5f));

      // init all geometry
      for(int i = 0; i < geometryList.size(); i++)
        geometryList[i]->init(context);
      // init all materials
      for(int i = 0; i < geometryList.size(); i++)
        materialList[i]->init(context);

      // GeometryInstance
      geoInstList.resize(geometryList.size());
      // Taking advantage of geometryList.size == materialList.size
      for (int i = 0; i < geoInstList.size(); i++)
      {
        geoInstList[i] = ioGeometryInstance();
        geoInstList[i].init(context);
        geoInstList[i].setGeometry(*geometryList[i]);
        geoInstList[i].setMaterial(*materialList[i]);
      }

      // World & Acceleration
      geometryGroup.init(context);
      for (int i = 0; i < geoInstList.size(); i++)
        geometryGroup.addChild(geoInstList[i]);

      // Setting World Variable
      context["sysWorld"]->set(geometryGroup.get());

      // Create and Init our scene camera
      camera = new ioEnvironmentCamera(
        -2.0f, 2.0f, 1.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 0.0f
        );

      // camera = new ioPerspectiveCamera(
      // -2.0f, 2.0f, 1.0f,
      // 0.0f, 0.0f, -1.0f,
      // 0.0f, 1.0f, 0.0f,
      // 90.0f, 2.0f
      // );

      // camera = new ioOrthographicCamera(
      //   -2.0f, 2.0f, 1.0f,
      //   0.0f, 0.0f, -1.0f,
      //   0.0f, 1.0f, 0.0f,
      //   5.0f, 10.0f
      //   );

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
  std::vector<ioGeometryInstance> geoInstList;
  ioGeometryGroup geometryGroup;

  ioCamera* camera;
};

#endif //!IO_SCENE_H
