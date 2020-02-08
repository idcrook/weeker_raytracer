#ifndef IO_GROUP_H
#define IO_GROUP_H

#include <optix.h>
#include <optixu/optixpp.h>

class ioGroup
{
public:
    void init(optix::Context& context) {
        optix::Group group = context->createGroup();
        group->setAcceleration(context->createAcceleration("Trbvh"));
        m_group = group;
    }


    // groups and primitives to the hierarchy
    void addChild(optix::GeometryInstance gi, optix::Group &d_world, optix::Context &g_context){
        optix::GeometryGroup test = g_context->createGeometryGroup();
        test->setAcceleration(g_context->createAcceleration("Trbvh"));
        test->setChildCount(1);
        test->setChild(0, gi);

        int i = d_world->getChildCount();
        d_world->setChildCount(i + 1);
        d_world->setChild(i, test);
        d_world->getAcceleration()->markDirty();
    }

    void addChild(optix::GeometryGroup gg, optix::Group &d_world, optix::Context &g_context){
        int i = d_world->getChildCount();
        d_world->setChildCount(i + 1);
        d_world->setChild(i, gg);
        d_world->getAcceleration()->markDirty();
    }

    void addChild(optix::Transform gi, optix::Group &d_world, optix::Context &g_context){
        int i = d_world->getChildCount();
        d_world->setChildCount(i + 1);
        d_world->setChild(i, gi);
        d_world->getAcceleration()->markDirty();
    }


    virtual void destroy() {
        if (m_group) {
            m_group->destroy();
            m_group = nullptr;
        }
    }

    optix::Group get()    {
        return m_group;
    }

protected:
    optix::Group m_group;
};

#endif //!IO_GROUP_H
