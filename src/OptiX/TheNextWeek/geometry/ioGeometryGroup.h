#ifndef IO_GEOMETRY_GROUP_H
#define IO_GEOMETRY_GROUP_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioAARect.h"
#include "geometry/ioGeometryInstance.h"

class ioGeometryGroup
{
public:
  ioGeometryGroup() { }

  void init(optix::Context& context)
    {
      m_gg = context->createGeometryGroup();
      // NoAccel, Bvh, Sbvh, Trbvh
      //m_gg->setAcceleration(context->createAcceleration("NoAccel")); // faster than Trbvh with early empty Cornell box
      m_gg->setAcceleration(context->createAcceleration("Trbvh"));
      m_gg->setChildCount(0);
    }


    // Utility function - box made of rectangle primitives
   static optix::GeometryGroup createBox(const float3& p0, const float3& p1, ioMaterial* material, optix::Context &context){
        std::vector<ioGeometry*> geometryList;
        std::vector<ioGeometryInstance> geoInstList;

        geometryList.push_back(new ioAARect(p0.x, p1.x, p0.y, p1.y, p0.z, true,  Z_AXIS)); // left wall
        geometryList.push_back(new ioAARect(p0.x, p1.x, p0.y, p1.y, p1.z, false, Z_AXIS)); // right wall

        geometryList.push_back(new ioAARect(p0.x, p1.x, p0.z, p1.z, p0.y, true,  Y_AXIS)); // roof
        geometryList.push_back(new ioAARect(p0.x, p1.x, p0.z, p1.z, p1.y, false, Y_AXIS)); // floor

        geometryList.push_back(new ioAARect(p0.y, p1.y, p0.z, p1.z, p0.x, true,  X_AXIS)); // back wall
        geometryList.push_back(new ioAARect(p0.y, p1.y, p0.z, p1.z, p1.x, false, X_AXIS)); // front wall

        // init all geometry
        for(int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }
        // GeometryInstance
        geoInstList.resize(geometryList.size());
        for (int i = 0; i < geoInstList.size(); i++)
        {
            //std::cerr << i << std::endl;
            geoInstList[i] = ioGeometryInstance();
            geoInstList[i].init(context);
            geoInstList[i].setGeometry(*geometryList[i]);
            material->assignTo(geoInstList[i].get(), context);
        }

        optix::GeometryGroup d_world = context->createGeometryGroup();
        d_world->setAcceleration(context->createAcceleration("Trbvh"));
        d_world->setChildCount((int)geometryList.size());
        for (int i = 0; i < geoInstList.size(); i++)
            d_world->setChild(i, geoInstList[i].get());

        return d_world;
    }




  void destroy()
    {
      m_gg->destroy();
    }

  void addChild(ioGeometryInstance& gi)
    {
      m_gg->addChild(gi.get());
    }

  optix::GeometryGroup get()
    {
      return m_gg;
    }

private:
  optix::GeometryGroup m_gg;
};

#endif //!IO_GEOMETRY_GROUP_H
