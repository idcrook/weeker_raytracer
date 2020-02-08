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
    void addChild(optix::GeometryInstance gi,  optix::Context &g_context){
        optix::GeometryGroup gg = g_context->createGeometryGroup();
        gg->setAcceleration(g_context->createAcceleration("Trbvh"));
        gg->setChildCount(1);
        gg->setChild(0, gi);

        int i = m_group->getChildCount();
        m_group->setChildCount(i + 1);
        m_group->setChild(i, gg);
        m_group->getAcceleration()->markDirty();
    }

    void addChild(optix::GeometryGroup gg, optix::Context &g_context){
        int i = m_group->getChildCount();
        m_group->setChildCount(i + 1);
        m_group->setChild(i, gg);
        m_group->getAcceleration()->markDirty();
    }

    void addChild(optix::Transform gi,  optix::Context &g_context){
        int i = m_group->getChildCount();
        m_group->setChildCount(i + 1);
        m_group->setChild(i, gi);
        m_group->getAcceleration()->markDirty();
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
