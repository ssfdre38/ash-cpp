cmake_minimum_required(VERSION 3.20)
project(ash_engine VERSION 0.1.0 LANGUAGES C CXX)

# C++20 required for coroutines
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# SQLite library
add_library(sqlite3 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/sqlite-amalgamation-3530000/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/sqlite-amalgamation-3530000)

# Source files
set(SOURCES
    src/main.cpp
    src/event_loop.cpp
    src/logger.cpp
    src/model_loader.cpp
    src/emotional_state.cpp
    src/memory_store.cpp
    src/decision_engine.cpp
    src/context_manager.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
)

# Executable
add_executable(ash_engine ${SOURCES})

# Model test executable
add_executable(model_test 
    src/model_test.cpp
    src/logger.cpp
    src/model_loader.cpp
)

# Emotional state test executable
add_executable(emotional_test
    src/emotional_test.cpp
    src/logger.cpp
    src/emotional_state.cpp
    src/event_loop.cpp
)

# Memory test executable
add_executable(memory_test
    src/memory_test.cpp
    src/logger.cpp
    src/memory_store.cpp
    src/emotional_state.cpp
)

# Include directories
target_include_directories(ash_engine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(model_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(emotional_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(memory_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(ash_engine PRIVATE Threads::Threads sqlite3)
target_link_libraries(model_test PRIVATE Threads::Threads)
target_link_libraries(emotional_test PRIVATE Threads::Threads)
# Decision test executable
add_executable(decision_test
    src/decision_test.cpp
    src/logger.cpp
    src/memory_store.cpp
    src/emotional_state.cpp
    src/decision_engine.cpp
)

target_include_directories(decision_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Context test executable
add_executable(context_test
    src/context_test.cpp
    src/logger.cpp
    src/memory_store.cpp
    src/emotional_state.cpp
    src/context_manager.cpp
)

target_include_directories(context_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Tensor test executable
add_executable(tensor_test
    src/tensor_test.cpp
    src/logger.cpp
    src/tensor.cpp
)

target_include_directories(tensor_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tensor_test PRIVATE Threads::Threads)

# Matrix operations test executable
add_executable(matrix_ops_test
    src/matrix_ops_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
)

target_include_directories(matrix_ops_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(matrix_ops_test PRIVATE Threads::Threads)

# GGUF parser test executable
add_executable(gguf_parser_test
    src/gguf_parser_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/gguf_parser.cpp
)

target_include_directories(gguf_parser_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gguf_parser_test PRIVATE Threads::Threads)

# Tokenizer test executable
add_executable(tokenizer_test
    src/tokenizer_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
)

target_include_directories(tokenizer_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tokenizer_test PRIVATE Threads::Threads)

# Attention test executable
add_executable(attention_test
    src/attention_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/attention.cpp
)

target_include_directories(attention_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(attention_test PRIVATE Threads::Threads)

# Installation
install(TARGETS ash_engine DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
