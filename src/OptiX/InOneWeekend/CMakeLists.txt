INCLUDE_DIRECTORIES(.)

cuda_compile_and_embed(
  drawColor_ptx_c drawColor.cu
)

add_executable(inOneWeekendOptix

  main.cpp
  # Compiled cuda ptx programs
  ${drawColor_ptx_c}
)

target_link_libraries(inOneWeekendOptix
  ${optix_LIBRARY}
  )
