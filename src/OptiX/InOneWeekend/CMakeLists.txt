INCLUDE_DIRECTORIES(.)


cuda_compile_and_embed(  drawColor_ptx_c         basics/drawColor.cu  )
cuda_compile_and_embed(  drawColorGradient_ptx_c basics/drawColorGradient.cu   )

cuda_compile_and_embed(
  raygen_ptx_c raygen.cu
  )

cuda_compile_and_embed(
  miss_ptx_c miss.cu  )

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

add_executable(inOneWeekendOptix
  main.cpp
  # Compiled cuda ptx programs
  ${raygen_ptx_c}
  ${miss_ptx_c}
  )

target_link_libraries(basics_drawColor          ${optix_LIBRARY})
target_link_libraries(basics_drawColorGradient  ${optix_LIBRARY})
target_link_libraries(inOneWeekendOptix  ${optix_LIBRARY})
