cmake_minimum_required(VERSION 3.21)
project(KataGomoInference LANGUAGES CXX)
set(COMPILE_MAX_BOARD_LEN 21 CACHE STRING "Maximum supported board edge length")
add_compile_definitions("$<$<COMPILE_LANGUAGE:C,CXX>:COMPILE_MAX_BOARD_LEN=${COMPILE_MAX_BOARD_LEN}>")
if(USE_BACKEND STREQUAL "DISPATCH")
  include(dispatcher.cmake)
  return()
endif()
if(USE_BACKEND STREQUAL "METAL")
  enable_language(Swift)
endif()
set(NO_GIT_REVISION ON CACHE BOOL "" FORCE)
set(BUILD_DISTRIBUTED OFF CACHE BOOL "" FORCE)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
add_subdirectory("${UPSTREAM_KATAGO}/cpp" upstream EXCLUDE_FROM_ALL)
if(USE_BACKEND STREQUAL "METAL")
  foreach(source metalbackend.cpp metalbackend.swift metallayers.swift)
    configure_file("${UPSTREAM_KATAGO}/cpp/neuralnet/${source}"
      "${CMAKE_CURRENT_BINARY_DIR}/${source}" COPYONLY)
  endforeach()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-transformer.patch")
  execute_process(COMMAND patch -s -p0 -d "${CMAKE_CURRENT_BINARY_DIR}"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/metal-transformer.patch"
    COMMAND_ERROR_IS_FATAL ANY)
  get_target_property(metal_sources KataGoSwift SOURCES)
  list(REMOVE_ITEM metal_sources neuralnet/metalbackend.swift neuralnet/metallayers.swift)
  list(APPEND metal_sources
    "${CMAKE_CURRENT_BINARY_DIR}/metalbackend.swift"
    "${CMAKE_CURRENT_BINARY_DIR}/metallayers.swift"
    "${CMAKE_CURRENT_SOURCE_DIR}/benty-mixed.swift")
  set_property(TARGET KataGoSwift PROPERTY SOURCES "${metal_sources}")
endif()
get_target_property(upstream_sources katago SOURCES)
foreach(source IN LISTS upstream_sources)
  if(source MATCHES "^(core/|game/|dataio/homedata.cpp$|neuralnet/(desc|modelversion|nninputs|sgfmetadata|opencl.*|metalbackend)\\.cpp$)")
    list(APPEND inference_sources "${UPSTREAM_KATAGO}/cpp/${source}")
  endif()
endforeach()
if(USE_BACKEND STREQUAL "METAL")
  list(REMOVE_ITEM inference_sources "${UPSTREAM_KATAGO}/cpp/neuralnet/metalbackend.cpp")
  list(APPEND inference_sources "${CMAKE_CURRENT_BINARY_DIR}/metalbackend.cpp")
endif()
if(USE_BACKEND STREQUAL "OPENCL")
  set(packed_source "${CMAKE_CURRENT_BINARY_DIR}/openclbackend.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl-y-packed.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl-output-reads.patch"
    "${UPSTREAM_KATAGO}/cpp/neuralnet/openclbackend.cpp")
  execute_process(COMMAND patch -s -o "${packed_source}"
    "${UPSTREAM_KATAGO}/cpp/neuralnet/openclbackend.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/opencl-y-packed.patch"
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND patch -s "${packed_source}"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl-output-reads.patch"
    COMMAND_ERROR_IS_FATAL ANY)
  list(REMOVE_ITEM inference_sources "${UPSTREAM_KATAGO}/cpp/neuralnet/openclbackend.cpp")
  list(APPEND inference_sources "${packed_source}")
endif()
include(benty-graph.cmake)
add_library(katagoinference SHARED bridge.cpp ${inference_sources})
target_include_directories(katagoinference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(katagoinference PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_BINARY_DIR}/desc.h>")
target_include_directories(katagoinference PRIVATE "${UPSTREAM_KATAGO}/cpp" "${UPSTREAM_KATAGO}/cpp/neuralnet")
foreach(property INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES)
  get_target_property(value katago ${property})
  if(value)
    set_property(TARGET katagoinference APPEND PROPERTY ${property} "${value}")
  endif()
endforeach()
string(TOLOWER "${USE_BACKEND}" backend_name)
set_target_properties(katagoinference PROPERTIES OUTPUT_NAME "katagoinference_${backend_name}"
  CXX_STANDARD 17 CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
if(USE_BACKEND STREQUAL "METAL")
  set_target_properties(katagoinference PROPERTIES LINKER_LANGUAGE Swift)
endif()
