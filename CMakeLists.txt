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

# Handled within each
ADD_SUBDIRECTORY( src/Cpp )

if(CMAKE_CUDA_COMPILER)   # Only if CUDA is available
  ADD_SUBDIRECTORY( src/Cuda )
  ADD_SUBDIRECTORY( src/OptiX/InOneWeekend )
  ADD_SUBDIRECTORY( src/OptiX/TheNextWeek )
endif()
