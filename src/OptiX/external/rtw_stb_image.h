#pragma once

#ifndef RTWEEKEND_STB_IMAGE_H
#define RTWEEKEND_STB_IMAGE_H


// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
    // Microsoft Visual C++ Compiler
    #pragma warning (push, 0)
#endif


#pragma GCC system_header
// Implementation must #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// Implementation must #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



// Restore warning levels.
#ifdef _MSC_VER
    // Microsoft Visual C++ Compiler
    #pragma warning (pop)
#endif

#endif
