#ifndef IO_GROUP_H
#define IO_GROUP_H

#include <iostream>

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
    void addChild(optix::GeometryInstance gi,  optix::Context &context){
        optix::GeometryGroup gg = context->createGeometryGroup();
        gg->setAcceleration(context->createAcceleration("Trbvh"));
        //gg->setAcceleration(context->createAcceleration("NoAccel"));
        gg->setChildCount(1);
        gg->setChild(0, gi);

        int i = m_group->getChildCount();
        // std::cerr << "DEBUG: Group child count: " << i  << std::endl;
        m_group->setChildCount(i + 1);
        m_group->setChild(i, gg);
        m_group->getAcceleration()->markDirty();
    }

    void addChild(optix::GeometryGroup gg, optix::Context &context){
        int i = m_group->getChildCount();
        // std::cerr << "DEBUG: Group child count: " << i  << std::endl;
        m_group->setChildCount(i + 1);
        m_group->setChild(i, gg);
        m_group->getAcceleration()->markDirty();
    }

    void addChild(optix::Transform t, optix::Context &context){
        int i = m_group->getChildCount();
        // std::cerr << "DEBUG: Group child count: " << i  << std::endl;
        m_group->setChildCount(i + 1);
        m_group->setChild(i, t);
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
