INCLUDE_DIRECTORIES(.)

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
  moving_sphere_ptx_c   geometry/moving_sphere.cu
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

cuda_compile_and_embed(
  dielectric_material_ptx_c material/dielectricMaterial.cu
  )

cuda_compile_and_embed(
  constant_texture_ptx_c    texture/constantTexture.cu
  )

cuda_compile_and_embed(
  checkered_texture_ptx_c     texture/checkeredTexture.cu
  )


add_executable(theNextWeekOptix
  Director.h
  Director.cpp
  main.cpp
  # Compiled cuda ptx programs
  ${raygen_ptx_c}
  ${miss_ptx_c}
  ${sphere_ptx_c}
  ${moving_sphere_ptx_c}
  ${normal_material_ptx_c}
  ${lambertian_material_ptx_c}
  ${metal_material_ptx_c}
  ${dielectric_material_ptx_c}
  ${constant_texture_ptx_c}
  ${checkered_texture_ptx_c}
  )


target_link_libraries(theNextWeekOptix  ${optix_LIBRARY})
