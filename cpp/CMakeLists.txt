cmake_minimum_required(VERSION 3.14)
project("got.cpp" C CXX)


set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_EXAMPLES ON)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # 根据构建类型设置运行时库
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # 使用 /MTd（Debug 模式下的静态链接）
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    else()
        # 使用 /MT（Release 模式下的静态链接）
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    endif()
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(llama.cpp)

#set(TARGET libocr)
#add_library(${TARGET} OBJECT src/libocr.cpp)
#target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
#target_include_directories(${TARGET} PUBLIC src)
#target_compile_features(${TARGET} PRIVATE cxx_std_17)
#
#set(TARGET libocr_static)
#add_library(${TARGET} STATIC $<TARGET_OBJECTS:libocr>)
#set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME libocr)
#install(TARGETS ${TARGET}
#        ARCHIVE DESTINATION lib
#        LIBRARY DESTINATION lib
#        RUNTIME DESTINATION bin)
#target_compile_features(${TARGET} PRIVATE cxx_std_17)
#
#set(TARGET libocr_shared)
#add_library(${TARGET} SHARED $<TARGET_OBJECTS:libocr>)
#set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME gotocr)
#target_compile_definitions(${TARGET} PRIVATE OCR_BUILD)
#target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
#install(TARGETS ${TARGET}
#        ARCHIVE DESTINATION lib
#        LIBRARY DESTINATION lib
#        RUNTIME DESTINATION bin)
#target_compile_features(${TARGET} PRIVATE cxx_std_17)



set(TARGET libocr)
add_library(${TARGET} SHARED src/libocr.cpp)

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME libocr)
install(TARGETS ${TARGET}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PUBLIC src)
target_compile_definitions(${TARGET} PRIVATE OCR_BUILD)

#target_include_directories(${TARGET} PUBLIC llama.cpp/common)
#target_include_directories(${TARGET} PUBLIC llama.cpp/examples/llava)
#target_include_directories(${TARGET} PUBLIC llama.cpp/examples)
#target_include_directories(${TARGET} PUBLIC llama.cpp/include)
#target_include_directories(${TARGET} PUBLIC llama.cpp/ggml/include)


