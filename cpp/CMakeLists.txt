cmake_minimum_required(VERSION 3.14)
project("got.cpp" C CXX)


set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_EXAMPLES ON)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(llama.cpp)

set(TARGET libocr)
add_library(${TARGET} STATIC src/libocr.cpp)

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME libocr)
install(TARGETS ${TARGET}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava )
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE OCR_BUILD)

#target_include_directories(${TARGET} PUBLIC llama.cpp/common)
#target_include_directories(${TARGET} PUBLIC llama.cpp/examples/llava)
#target_include_directories(${TARGET} PUBLIC llama.cpp/examples)
#target_include_directories(${TARGET} PUBLIC llama.cpp/include)
#target_include_directories(${TARGET} PUBLIC llama.cpp/ggml/include)


