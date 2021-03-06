// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../lib/raydata.cuh"
#include "../lib/sampling.cuh"

// Ray state variables
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, sysWorld, , );

// The point and normal of intersection and UV parms
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

/*! and finally - that particular material's parameters */
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );


/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
// inline __device__ bool scatter(const optix::Ray &ray_in,
//                                DRand48 &rndState,
//                                vec3f &scattered_origin,
//                                vec3f &scattered_direction,
//                                vec3f &attenuation) {
//   return false;
// }

inline __device__ float3 emitted(){
  return sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
}

RT_PROGRAM void closestHit() {
  thePrd.emitted = emitted();
  thePrd.scatterEvent = Ray_Cancel;
    // = scatter(ray,
    //           *prd.in.randState,
    //           prd.out.scattered_origin,
    //           prd.out.scattered_direction,
    //           prd.out.attenuation)
    // ? rayGotBounced
    // : rayGotCancelled;
}
