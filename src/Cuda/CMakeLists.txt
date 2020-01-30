cmake_minimum_required ( VERSION 3.11 )

project ( RTWeekendCuda
  VERSION 2.95.0
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
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # for multi-target pre-compilations
else()
  message(STATUS "No CUDA support")
endif()



set ( COMMON_ALL_CUDA
  commonCuda/rtweekend.cuh
  commonCuda/camera.cuh
  commonCuda/ray.cuh
  commonCuda/vec3.cuh
)

set ( SOURCE_ONE_WEEKEND_CUDA
  ${COMMON_ALL_CUDA}
  InOneWeekend/hittable.cuh
  InOneWeekend/hittable_list.cuh
  InOneWeekend/material.cuh
  InOneWeekend/sphere.cuh
  InOneWeekend/main.cu
)


set ( SOURCE_NEXT_WEEK_CUDA
  ${COMMON_ALL_CUDA}
  commonCuda/aabb.cuh
  # commonCuda/external/stb_image.h
  # commonCuda/perlin.h
  # commonCuda/rtw_stb_image.h
  commonCuda/texture.cuh
  # TheNextWeekCuda/aarect.h
  # TheNextWeekCuda/box.h
  # TheNextWeekCuda/bvh.cuh
  # TheNextWeekCuda/constant_medium.h
  TheNextWeek/moving_sphere.cuh
  TheNextWeek/hittable.cuh
  TheNextWeek/hittable_list.cuh
  TheNextWeek/material.cuh
  TheNextWeek/sphere.cuh
  TheNextWeek/main.cu
  )

add_executable(inOneWeekendCuda  ${SOURCE_ONE_WEEKEND_CUDA})

add_executable(theNextWeekCuda   ${SOURCE_NEXT_WEEK_CUDA})

target_include_directories(inOneWeekendCuda  PRIVATE .)

target_include_directories(theNextWeekCuda  PRIVATE .)
