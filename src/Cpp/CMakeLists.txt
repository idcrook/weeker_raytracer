
cmake_minimum_required ( VERSION 3.8 )

project ( RTWeekendCpp
  VERSION 2.95.0
  )

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

# Source

set ( COMMON_ALL
  common/rtweekend.h
  common/camera.h
  common/ray.h
  common/vec3.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  InOneWeekend/hittable.h
  InOneWeekend/hittable_list.h
  InOneWeekend/material.h
  InOneWeekend/sphere.h
  InOneWeekend/main.cc
)

set ( SOURCE_NEXT_WEEK
  ${COMMON_ALL}
  common/aabb.h
  common/external/stb_image.h
  common/perlin.h
  common/rtw_stb_image.h
  common/texture.h
  TheNextWeek/aarect.h
  TheNextWeek/box.h
  TheNextWeek/bvh.h
  TheNextWeek/constant_medium.h
  TheNextWeek/hittable.h
  TheNextWeek/hittable_list.h
  TheNextWeek/material.h
  TheNextWeek/moving_sphere.h
  TheNextWeek/sphere.h
  TheNextWeek/main.cc
)

set ( SOURCE_REST_OF_LIFE
  ${COMMON_ALL}
  common/aabb.h
  common/external/stb_image.h
  common/perlin.h
  common/rtw_stb_image.h
  common/texture.h
  RestOfLife/aarect.h
  RestOfLife/box.h
  RestOfLife/bvh.h
  RestOfLife/hittable.h
  RestOfLife/hittable_list.h
  RestOfLife/material.h
  RestOfLife/moving_sphere.h
  RestOfLife/onb.h
  RestOfLife/pdf.h
  RestOfLife/sphere.h
  RestOfLife/main.cc
)

# Executables
add_executable(inOneWeekendCpp ${SOURCE_ONE_WEEKEND})
add_executable(theNextWeekCpp  ${SOURCE_NEXT_WEEK})
add_executable(restOfLifeCpp   ${SOURCE_REST_OF_LIFE})

target_include_directories(inOneWeekendCpp PRIVATE .)
target_include_directories(theNextWeekCpp  PRIVATE .)
target_include_directories(restOfLifeCpp   PRIVATE .)
