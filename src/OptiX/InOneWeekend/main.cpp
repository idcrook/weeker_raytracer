// std
#include <iostream>

// optix
#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char raygen_ptx_c[];
extern "C" const char miss_ptx_c[];
extern "C" const char sphere_ptx_c[];
extern "C" const char material_ptx_c[];

int main(int argc, char* argv[])
{
  int width = 1200;
  int height = 600;

  // Context
  optix::Context context = optix::Context::create();
  context->setRayTypeCount(1);

  // RayGen & Miss
  context->setEntryPointCount(1);
  context->setRayGenerationProgram(
    0,
    context->createProgramFromPTXString(
      raygen_ptx_c, "rayGenProgram"
      )
    );
  context->setMissProgram(
    0,
    context->createProgramFromPTXString(
      miss_ptx_c, "missProgram"
      )
    );

  // Buffer
  optix::Buffer resultBuffer = context->createBuffer(RT_BUFFER_OUTPUT);
  resultBuffer->setFormat(RT_FORMAT_FLOAT3);
  resultBuffer->setSize(width, height);
  // Setting Buffer Variable
  context["resultBuffer"]->set(resultBuffer);

  // Sphere
  //   Sphere Geometry
  optix::Geometry sphere = context->createGeometry();
  sphere->setPrimitiveCount(1);
  sphere->setBoundingBoxProgram(
    context->createProgramFromPTXString(sphere_ptx_c, "getBounds")
    );
  sphere->setIntersectionProgram(
    context->createProgramFromPTXString(sphere_ptx_c, "intersection")
    );
  sphere["center"]->setFloat(0.0f, 0.0f, -1.0f);
  sphere["radius"]->setFloat(0.5f);
  //   Sphere Material
  optix::Material material = context->createMaterial();
  optix::Program materialHit = context->createProgramFromPTXString(
    material_ptx_c, "closestHit"
    );
  material->setClosestHitProgram(0, materialHit);
  //   Sphere GeometryInstance
  optix::GeometryInstance gi = context->createGeometryInstance();
  gi->setGeometry(sphere);
  gi->setMaterialCount(1);
  gi->setMaterial(0, material);

  // World & Acceleration
  optix::GeometryGroup world = context->createGeometryGroup();
  world->setAcceleration(context->createAcceleration("Bvh"));
  world->setChildCount(1);
  world->setChild(0, gi);
  // Setting World Variable
  context["world"]->set(world);

  // Run
  context->validate();
  context->launch(
    0,         // Program ID
    width,     // launch dimension x
    height     // launch dimension y
    );

  // Print Out
  void* bufferData = resultBuffer->map();
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
  resultBuffer->unmap();

  // Destroy our objects
  world->destroy();
  gi->destroy();
  sphere->destroy();
  material->destroy();
  resultBuffer->destroy();
  context->destroy();
}
