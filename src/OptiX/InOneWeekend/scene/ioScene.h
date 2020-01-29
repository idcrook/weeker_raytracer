#ifndef IO_SCENE_H
#define IO_SCENE_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometry.h"
#include "geometry/ioGeometryInstance.h"
#include "geometry/ioGeometryGroup.h"

#include "geometry/ioSphere.h"
#include "material/ioNormalMaterial.h"
#include "material/ioLambertianMaterial.h"

class ioScene
{
public:
  ioScene() { }

  void init(optix::Context& context)
    {
      // Geometry
      geometryList.push_back(new ioSphere(
                               0.0f, 0.0f, -1.0f,
                               0.5f
                               ));
      geometryList.push_back(new ioSphere(
                               0.0, -100.5, -1.0f,
                               100.0f
                               ));
      geometryList[0]->init(context);
      geometryList[1]->init(context);

      // Material
      materialList.push_back(new ioNormalMaterial());
      //materialList.push_back(new ioLambertianMaterial(0.7f, 0.5f, 0.0f));
      materialList.push_back(new ioLambertianMaterial(0.5f, 0.5f, 0.5f));
      materialList[0]->init(context);
      materialList[1]->init(context);

      // GeometryInstance
      //  Normal Sphere
      geoInstList.push_back(ioGeometryInstance());
      geoInstList[0].init(context);
      geoInstList[0].setGeometry(*geometryList[0]);
      geoInstList[0].setMaterial(*materialList[0]);
      // Lambertian Sphere
      geoInstList.push_back(ioGeometryInstance());
      geoInstList[1].init(context);
      geoInstList[1].setGeometry(*geometryList[1]);
      geoInstList[1].setMaterial(*materialList[1]);

      // World & Acceleration
      geometryGroup.init(context);
      geometryGroup.addChild(geoInstList[0]);
      geometryGroup.addChild(geoInstList[1]);

      // Setting World Variable
      context["sysWorld"]->set(geometryGroup.get());
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
    }

public:
  std::vector<ioMaterial*> materialList;
  std::vector<ioGeometry*> geometryList;
  std::vector<ioGeometryInstance> geoInstList;
  ioGeometryGroup geometryGroup;
};

#endif //!IO_SCENE_H
