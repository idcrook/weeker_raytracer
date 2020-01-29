// std
#include <iostream>

// optix
#include <optix.h>
#include <optixu/optixpp.h>

#define RT_CHECK( func )                                        \
  do                                                            \
  {                                                             \
    RTresult code = func;                                       \
    if( code != RT_SUCCESS ) {                                  \
      const char* errorString;                                  \
      rtContextGetErrorString( context, code, &errorString );   \
      std::cout << "RT_ERROR: " << __FILE__ << ":" << __LINE__  \
                << " - 0x" <<  std::hex << code << std::dec     \
                << " - " << errorString << std::endl;           \
    }                                                           \
  } while(0)

extern "C" const char raygen_ptx_c[];
extern "C" const char miss_ptx_c[];
extern "C" const char sphere_ptx_c[];
extern "C" const char material_ptx_c[];

int main(int argc, char* argv[])
{
  // Primary RT Objects
  RTcontext context = 0;
  //    Programs and Variables
  RTprogram rayGenProgram;
  RTvariable rtvResultBuffer;
  RTvariable rtvWorld;
  RTprogram missProgram;
  RTprogram sphereBoundingBoxProgram;
  RTprogram sphereIntersectionProgram;
  RTvariable rtvSphereCenter;
  RTvariable rtvSphereRadius;
  RTprogram sphereMatClosestHitProgram;
  //    Result Buffer
  RTbuffer resultBuffer;
  //    The geometry of the world (scene)
  RTgeometrygroup world;
  RTacceleration acceleration;
  //    The sole geometry occupying the world, a sphere
  RTgeometry sphere;
  RTgeometryinstance sphereInstance;
  //    The Material of the Sphere
  RTmaterial sphereMaterial;

  // Parameters
  int width = 1200;
  int height = 600; // @writing explain change from 800 to 600

  // Create Objects
  //   Context
  RT_CHECK( rtContextCreate( &context ));
  RT_CHECK( rtContextSetRayTypeCount( context, 1 ));
  RT_CHECK( rtContextSetEntryPointCount( context, 1 ));
  //   Buffer
  RT_CHECK( rtBufferCreate( context, RT_BUFFER_OUTPUT, &resultBuffer ));
  RT_CHECK( rtBufferSetFormat( resultBuffer, RT_FORMAT_FLOAT3 ));
  RT_CHECK( rtBufferSetSize2D( resultBuffer, width, height));
  //   Acceleration
  RT_CHECK( rtAccelerationCreate( context, &acceleration ));
  RT_CHECK( rtAccelerationSetBuilder( acceleration, "bvh" ));
  //   World
  RT_CHECK( rtGeometryGroupCreate( context, &world ));
  RT_CHECK( rtGeometryGroupSetAcceleration( world, acceleration ));

  //   Sphere
  RT_CHECK( rtGeometryCreate(context, &sphere ));
  //     Set the number of primitives in the sphere geometry (1, just the sphere)
  RT_CHECK( rtGeometrySetPrimitiveCount( sphere, 1 ));
  //     Add both of the sphere programs
  RT_CHECK( rtProgramCreateFromPTXString(
              context,
              sphere_ptx_c,
              "getBounds",
              &sphereBoundingBoxProgram ));
  RT_CHECK( rtGeometrySetBoundingBoxProgram( sphere, sphereBoundingBoxProgram ));
  RT_CHECK( rtProgramCreateFromPTXString(
              context,
              sphere_ptx_c,
              "intersection",
              &sphereIntersectionProgram ));
  RT_CHECK( rtGeometrySetIntersectionProgram( sphere, sphereIntersectionProgram ));
  //     Add Variables
  RT_CHECK( rtContextDeclareVariable( context, "center", &rtvSphereCenter ));
  RT_CHECK( rtContextDeclareVariable( context, "radius", &rtvSphereRadius ));
  RT_CHECK( rtVariableSet3f( rtvSphereCenter, 0.0f, 0.0f, -1.0f ));
  RT_CHECK( rtVariableSet1f( rtvSphereRadius, 0.5f ));
  //   Material
  RT_CHECK( rtMaterialCreate(context, &sphereMaterial ));
  //     Set material closest hit program
  RT_CHECK( rtProgramCreateFromPTXString(
              context,
              material_ptx_c,
              "closestHit",
              &sphereMatClosestHitProgram ));
  RT_CHECK( rtMaterialSetClosestHitProgram( sphereMaterial, 0, sphereMatClosestHitProgram ));

  // Add the sphere to the world
  //   Put sphere within a GeometryInstance
  RT_CHECK( rtGeometryInstanceCreate( context, &sphereInstance ));
  RT_CHECK( rtGeometryInstanceSetGeometry( sphereInstance, sphere ));
  //   Add a Material to the GeometryInstance
  RT_CHECK( rtGeometryInstanceSetMaterialCount( sphereInstance, 1 ));
  RT_CHECK( rtGeometryInstanceSetMaterial( sphereInstance, 0, sphereMaterial ));
  //   Add that GeometryInstance to our world
  RT_CHECK( rtGeometryGroupSetChildCount( world, 1 ));
  RT_CHECK( rtGeometryGroupSetChild(
              world,          // GeometryGroup
              0,              // Index of child
              sphereInstance  // GeometryInstance to add
              ));

  // Connecting Buffer to Context
  RT_CHECK( rtContextDeclareVariable( context, "resultBuffer", &rtvResultBuffer ));
  RT_CHECK( rtVariableSetObject( rtvResultBuffer, resultBuffer ));
  // Connecting World to Context
  RT_CHECK( rtContextDeclareVariable( context, "world", &rtvWorld ));
  RT_CHECK( rtVariableSetObject( rtvWorld, world ));

  // Set ray generation program
  RT_CHECK( rtProgramCreateFromPTXString(
              context,
              raygen_ptx_c,
              "rayGenProgram",
              &rayGenProgram ));
  RT_CHECK( rtContextSetRayGenerationProgram( context, 0, rayGenProgram ));

  // Set miss program
  RT_CHECK( rtProgramCreateFromPTXString(
              context,
              miss_ptx_c,
              "missProgram",
              &missProgram ));
  RT_CHECK( rtContextSetMissProgram( context, 0, missProgram ));

  // RUN
  RT_CHECK( rtContextValidate( context ));
  RT_CHECK( rtContextLaunch2D( context, 0,      // Entry Point
                               width, height ));

  // Print out to terminal as PPM
  //   Get pointer to output buffer data
  void* bufferData;
  RT_CHECK( rtBufferMap( resultBuffer, &bufferData ));
  //   Print ppm header
  std::cout << "P3\n" << width << " " << height << "\n255\n";
  //   Parse through bufferdata
  for (int j = height - 1; j >= 0;  j--)
  {
    for (int i = 0; i < width; i++)
    {
      float* floatData = ((float*)bufferData) + (3*(width*j + i));
      float r = floatData[0];
      float g = floatData[1];
      float b = floatData[2];
      int ir = int(255.99f * r);
      int ig = int(255.99f * g);
      int ib = int(255.99f * b);
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  RT_CHECK( rtBufferUnmap( resultBuffer ));

  // Explicitly destroy our objects
  RT_CHECK( rtGeometryDestroy( sphere ));
  RT_CHECK( rtAccelerationDestroy( acceleration ));
  RT_CHECK( rtMaterialDestroy( sphereMaterial ));
  RT_CHECK( rtGeometryInstanceDestroy( sphereInstance ));
  RT_CHECK( rtGeometryGroupDestroy( world ));
  RT_CHECK( rtBufferDestroy( resultBuffer ));
  RT_CHECK( rtProgramDestroy( missProgram ));
  RT_CHECK( rtProgramDestroy( rayGenProgram ));
  RT_CHECK( rtProgramDestroy( sphereBoundingBoxProgram ));
  RT_CHECK( rtProgramDestroy( sphereIntersectionProgram ));
  RT_CHECK( rtProgramDestroy( sphereMatClosestHitProgram ));
  // RT_CHECK( rtContextDestroy( context ));
}
