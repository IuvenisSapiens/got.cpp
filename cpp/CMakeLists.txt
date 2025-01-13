cmake_minimum_required(VERSION 3.14)
project("got.cpp" C CXX)


set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_EXAMPLES ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib/static)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    endif()
endif()



add_subdirectory(llama.cpp)


set(TARGET libocr)

add_library(${TARGET}_shared SHARED src/libocr.cpp)
set_target_properties(${TARGET}_shared PROPERTIES
        OUTPUT_NAME libocr
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/shared"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/shared"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/shared"
)
install(TARGETS ${TARGET}_shared
        ARCHIVE DESTINATION lib/shared
        LIBRARY DESTINATION lib/shared
        RUNTIME DESTINATION bin/shared
)
target_link_libraries(${TARGET}_shared PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET}_shared PRIVATE cxx_std_17)
target_include_directories(${TARGET}_shared PUBLIC src)
target_compile_definitions(${TARGET}_shared PRIVATE OCR_BUILD)
target_compile_definitions(${TARGET}_shared PRIVATE OCR_SHARED)
target_compile_definitions(${TARGET}_shared PRIVATE _CRT_SECURE_NO_WARNINGS)


add_library(${TARGET}_static STATIC src/libocr.cpp)
set_target_properties(${TARGET}_static PROPERTIES
        OUTPUT_NAME libocr
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/static"
)
install(TARGETS ${TARGET}_static
        ARCHIVE DESTINATION lib/static
        LIBRARY DESTINATION lib/static
        RUNTIME DESTINATION bin/static
)
target_link_libraries(${TARGET}_static PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET}_static PRIVATE cxx_std_17)
target_include_directories(${TARGET}_static PUBLIC src)
target_compile_definitions(${TARGET}_static PRIVATE OCR_BUILD)
target_compile_definitions(${TARGET}_static PRIVATE _CRT_SECURE_NO_WARNINGS)



