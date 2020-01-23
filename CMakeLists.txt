#-------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#-------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.11 )

project ( RTWeekend
  VERSION 2.95.0
  LANGUAGES CXX
)

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
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
else()
  message(STATUS "No CUDA support")
endif()

# Source
set ( COMMON_ALL
  src/common/rtweekend.h
  src/common/camera.h
  src/common/ray.h
  src/common/vec3.h
)

set ( COMMON_ALL_CUDA
  src/commonCuda/rtweekend.h
  src/commonCuda/camera.h
  src/commonCuda/ray.h
  src/commonCuda/vec3.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  src/InOneWeekend/hittable.h
  src/InOneWeekend/hittable_list.h
  src/InOneWeekend/material.h
  src/InOneWeekend/sphere.h
  src/InOneWeekend/main.cc
)

set ( SOURCE_ONE_WEEKEND_CUDA
  ${COMMON_ALL_CUDA}
  src/InOneWeekendCuda/hittable.h
  src/InOneWeekendCuda/hittable_list.h
  # src/InOneWeekendCuda/material.h
  src/InOneWeekendCuda/sphere.h
  src/InOneWeekendCuda/main.cu
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

# Executables
add_executable(inOneWeekend ${SOURCE_ONE_WEEKEND})
add_executable(theNextWeek  ${SOURCE_NEXT_WEEK})
add_executable(restOfLife   ${SOURCE_REST_OF_LIFE})
# add_executable(cos_cubed         src/RestOfLife/cos_cubed.cc         ${COMMON_ALL})
# add_executable(cos_density       src/RestOfLife/cos_density.cc       ${COMMON_ALL})
# add_executable(integrate_x_sq    src/RestOfLife/integrate_x_sq.cc    ${COMMON_ALL})
# add_executable(pi                src/RestOfLife/pi.cc                ${COMMON_ALL})
# add_executable(sphere_importance src/RestOfLife/sphere_importance.cc ${COMMON_ALL})
# add_executable(sphere_plot       src/RestOfLife/sphere_plot.cc       ${COMMON_ALL})
add_executable(inOneWeekendCuda ${SOURCE_ONE_WEEKEND_CUDA})


target_include_directories(inOneWeekend PRIVATE src)
target_include_directories(theNextWeek  PRIVATE src)
target_include_directories(restOfLife   PRIVATE src)
# target_include_directories(cos_cubed         PRIVATE src)
# target_include_directories(cos_density       PRIVATE src)
# target_include_directories(integrate_x_sq    PRIVATE src)
# target_include_directories(pi                PRIVATE src)
# target_include_directories(sphere_importance PRIVATE src)
# target_include_directories(sphere_plot       PRIVATE src)
target_include_directories(inOneWeekendCuda PRIVATE src)
