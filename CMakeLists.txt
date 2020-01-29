#-------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#-------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.11 )

project ( RTWeekend
  VERSION 2.95.0
  )

# set to match your path
set(OptiX_INSTALL_DIR "/usr/local/nvidia/NVIDIA-OptiX-SDK-6.5.0-linux64/"
  CACHE PATH "Path to OptiX installed location.")

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  #set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # for multi-target pre-compilations

  # Defines OptiX utilities and variables
  INCLUDE(cmake/configure_optix.cmake)
else()
  message(STATUS "No CUDA support")
endif()

# if(NOT SET_UP_CONFIGURATIONS_DONE)
#   set(SET_UP_CONFIGURATIONS_DONE 1)

#   # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
#   # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
#   if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
#     set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile" CACHE STRING "" FORCE)
#   else()
#     if(NOT CMAKE_BUILD_TYPE)
#       #            message("Defaulting to release build.")
#       set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
#     endif()
#     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
#     # set the valid options for cmake-gui drop-down list
#     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;Profile")
#   endif()
# endif()
# MARK_AS_ADVANCED(CUDA_SDK_ROOT_DIR)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Source
set ( COMMON_ALL
  src/common/rtweekend.h
  src/common/camera.h
  src/common/ray.h
  src/common/vec3.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  src/InOneWeekend/hittable.h
  src/InOneWeekend/hittable_list.h
  src/InOneWeekend/material.h
  src/InOneWeekend/sphere.h
  src/InOneWeekend/main.cc
)

set ( SOURCE_NEXT_WEEK
  ${COMMON_ALL}
  src/common/aabb.h
  src/common/external/stb_image.h
  src/common/perlin.h
  src/common/rtw_stb_image.h
  src/common/texture.h
  src/TheNextWeek/aarect.h
  src/TheNextWeek/box.h
  src/TheNextWeek/bvh.h
  src/TheNextWeek/constant_medium.h
  src/TheNextWeek/hittable.h
  src/TheNextWeek/hittable_list.h
  src/TheNextWeek/material.h
  src/TheNextWeek/moving_sphere.h
  src/TheNextWeek/sphere.h
  src/TheNextWeek/main.cc
)

set ( SOURCE_REST_OF_LIFE
  ${COMMON_ALL}
  src/common/aabb.h
  src/common/external/stb_image.h
  src/common/perlin.h
  src/common/rtw_stb_image.h
  src/common/texture.h
  src/RestOfLife/aarect.h
  src/RestOfLife/box.h
  src/RestOfLife/bvh.h
  src/RestOfLife/hittable.h
  src/RestOfLife/hittable_list.h
  src/RestOfLife/material.h
  src/RestOfLife/moving_sphere.h
  src/RestOfLife/onb.h
  src/RestOfLife/pdf.h
  src/RestOfLife/sphere.h
  src/RestOfLife/main.cc
)

set ( COMMON_ALL_CUDA
  src/commonCuda/rtweekend.cuh
  src/commonCuda/camera.cuh
  src/commonCuda/ray.cuh
  src/commonCuda/vec3.cuh
)

set ( SOURCE_ONE_WEEKEND_CUDA
  ${COMMON_ALL_CUDA}
  src/InOneWeekendCuda/hittable.cuh
  src/InOneWeekendCuda/hittable_list.cuh
  src/InOneWeekendCuda/material.cuh
  src/InOneWeekendCuda/sphere.cuh
  src/InOneWeekendCuda/main.cu
)

set ( SOURCE_NEXT_WEEK_CUDA
  ${COMMON_ALL_CUDA}
  src/commonCuda/aabb.cuh
  # src/commonCuda/external/stb_image.h
  # src/commonCuda/perlin.h
  # src/commonCuda/rtw_stb_image.h
  src/commonCuda/texture.cuh
  # src/TheNextWeekCuda/aarect.h
  # src/TheNextWeekCuda/box.h
  # src/TheNextWeekCuda/bvh.cuh
  # src/TheNextWeekCuda/constant_medium.h
  src/TheNextWeekCuda/moving_sphere.cuh
  src/TheNextWeekCuda/hittable.cuh
  src/TheNextWeekCuda/hittable_list.cuh
  src/TheNextWeekCuda/material.cuh
  src/TheNextWeekCuda/sphere.cuh
  src/TheNextWeekCuda/main.cu
)


# Executables
add_executable(inOneWeekend ${SOURCE_ONE_WEEKEND})
add_executable(theNextWeek  ${SOURCE_NEXT_WEEK})
add_executable(restOfLife   ${SOURCE_REST_OF_LIFE})
add_executable(inOneWeekendCuda  ${SOURCE_ONE_WEEKEND_CUDA})
add_executable(theNextWeekCuda   ${SOURCE_NEXT_WEEK_CUDA})

target_include_directories(inOneWeekend PRIVATE src)
target_include_directories(theNextWeek  PRIVATE src)
target_include_directories(restOfLife   PRIVATE src)

target_include_directories(inOneWeekendCuda PRIVATE src)
target_include_directories(theNextWeekCuda  PRIVATE src)

# Handled within each
ADD_SUBDIRECTORY( src/OptiX/InOneWeekend )
