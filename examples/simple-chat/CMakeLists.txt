set(TARGET llama-simple-chat)
add_library(${TARGET} SHARED simple-chat.cpp)
install(TARGETS ${TARGET} RUNTIME LIBRARY)

# Export symbols for DLL
target_compile_definitions(${TARGET} PRIVATE LLAMA_SHARED_LIB LLAMA_EXPORTS)
if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE _WINDLL)
endif()

# Link against llama library
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Generate export header
include(GenerateExportHeader)
generate_export_header(${TARGET})

# Add include directories
target_include_directories(${TARGET} PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}  # For the export header
    ${CMAKE_SOURCE_DIR}/include  # For llama.h in the include directory
)

# Install headers needed for integration
install(FILES 
    simple-chat.h
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_export.h
    DESTINATION include
)
