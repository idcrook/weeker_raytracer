INCLUDE_DIRECTORIES(.)

cuda_compile_and_embed(  drawColor_ptx_c         basics/drawColor.cu  )
cuda_compile_and_embed(  drawColorGradient_ptx_c basics/drawColorGradient.cu   )
add_executable(basics_drawColor
  basics/main.cpp
  # Compiled cuda ptx programs
  ${drawColor_ptx_c}
  )

add_executable(basics_drawColorGradient
  basics/main2.cpp
  # Compiled cuda ptx programs
  ${drawColorGradient_ptx_c}
  )
target_link_libraries(basics_drawColor          ${optix_LIBRARY})
target_link_libraries(basics_drawColorGradient  ${optix_LIBRARY})


cuda_compile_and_embed(
  raygen_ptx_c   raygen/raygen.cu
  )

cuda_compile_and_embed(
  miss_ptx_c     miss/miss.cu
  )

cuda_compile_and_embed(
  sphere_ptx_c   geometry/sphere.cu
  )

cuda_compile_and_embed(
  normal_material_ptx_c     material/normalMaterial.cu
  )

cuda_compile_and_embed(
  lambertian_material_ptx_c material/lambertianMaterial.cu
  )

cuda_compile_and_embed(
  metal_material_ptx_c      material/metalMaterial.cu
)

add_executable(inOneWeekendOptix
  introOptix.h
  introOptix.cpp
  main.cpp
  # Compiled cuda ptx programs
  ${raygen_ptx_c}
  ${miss_ptx_c}
  ${sphere_ptx_c}
  ${normal_material_ptx_c}
  ${lambertian_material_ptx_c}
  ${metal_material_ptx_c}
  )


target_link_libraries(inOneWeekendOptix  ${optix_LIBRARY})
