add_library(katagoinference SHARED dispatcher.cpp)
target_compile_definitions(katagoinference PRIVATE
  KG_OPENCL_LIBRARY_NAME="${CMAKE_SHARED_LIBRARY_PREFIX}katagoinference_opencl${CMAKE_SHARED_LIBRARY_SUFFIX}"
  KG_METAL_LIBRARY_NAME="${CMAKE_SHARED_LIBRARY_PREFIX}katagoinference_metal${CMAKE_SHARED_LIBRARY_SUFFIX}")
target_link_libraries(katagoinference PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
  target_compile_definitions(katagoinference PRIVATE NOMINMAX)
endif()
set_target_properties(katagoinference PROPERTIES CXX_STANDARD 17
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
