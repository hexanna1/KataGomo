# Bent-Y graph support is an overlay; the supplied upstream source stays intact.
foreach(source desc.h desc.cpp)
  configure_file("${UPSTREAM_KATAGO}/cpp/neuralnet/${source}"
    "${CMAKE_CURRENT_BINARY_DIR}/${source}" COPYONLY)
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/benty-graph-desc.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/benty-graph-opencl.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/benty-graph-metal.patch")
execute_process(COMMAND patch -s -p0 -d "${CMAKE_CURRENT_BINARY_DIR}"
  -i "${CMAKE_CURRENT_SOURCE_DIR}/benty-graph-desc.patch"
  COMMAND_ERROR_IS_FATAL ANY)
list(REMOVE_ITEM inference_sources "${UPSTREAM_KATAGO}/cpp/neuralnet/desc.cpp")
list(APPEND inference_sources "${CMAKE_CURRENT_BINARY_DIR}/desc.cpp")
if(USE_BACKEND STREQUAL "METAL")
  execute_process(COMMAND patch -s -p0 -d "${CMAKE_CURRENT_BINARY_DIR}"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/benty-graph-metal.patch"
    COMMAND_ERROR_IS_FATAL ANY)
elseif(USE_BACKEND STREQUAL "OPENCL")
  foreach(source openclkernels.cpp opencltuner.cpp)
    configure_file("${UPSTREAM_KATAGO}/cpp/neuralnet/${source}"
      "${CMAKE_CURRENT_BINARY_DIR}/${source}" COPYONLY)
  endforeach()
  execute_process(COMMAND patch -s -p0 -d "${CMAKE_CURRENT_BINARY_DIR}"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/benty-graph-opencl.patch"
    COMMAND_ERROR_IS_FATAL ANY)
  foreach(source openclkernels.cpp opencltuner.cpp)
    list(REMOVE_ITEM inference_sources "${UPSTREAM_KATAGO}/cpp/neuralnet/${source}")
    list(APPEND inference_sources "${CMAKE_CURRENT_BINARY_DIR}/${source}")
  endforeach()
endif()
