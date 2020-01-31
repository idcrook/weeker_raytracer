#ifndef CALLABLEPROG_CUH
#define CALLABLEPROG_CUH

/***
https://raytracing-docs.nvidia.com/optix6/guide_6_5/index.html#programs#callable-programs

It is recommended to replace all uses of the macro version of rtCallableProgram
with the templated version, rtCallableProgramX.

In addition, if the preprocessor macro RT_USE_TEMPLATED_RTCALLABLEPROGRAM is
defined then the old rtCallableProgram macro is supplanted by a definition that
uses rtCallableProgramX.

Therefore the define below is commented out and this entire file is effectively
empty.  Leaving in the repo to document this style guideline.

*/


#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM

#endif
