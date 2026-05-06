cmake_minimum_required(VERSION 3.20)
project(omega_pipeline LANGUAGES CXX)

option(OMEGA_PIPELINE_ENABLE_AVX512 "Enable AVX-512 evaluator implementation" ON)

add_library(omega_pipeline
  src/block_predictor.cpp
  src/evaluator.cpp
  src/ingestion.cpp
  src/npu_scorer.cpp
)

target_include_directories(omega_pipeline PUBLIC include)
target_compile_features(omega_pipeline PUBLIC cxx_std_20)
target_compile_options(omega_pipeline PRIVATE -Wall -Wextra -Wpedantic)

if (OMEGA_PIPELINE_ENABLE_AVX512)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx512f" CXX_SUPPORTS_AVX512F)
  if (CXX_SUPPORTS_AVX512F)
    target_compile_definitions(omega_pipeline PRIVATE OMEGA_PIPELINE_ENABLE_AVX512=1)
    target_compile_options(omega_pipeline PRIVATE -mavx512f)
  endif()
endif()

add_executable(omega_pipeline_demo src/demo.cpp)
target_link_libraries(omega_pipeline_demo PRIVATE omega_pipeline)

add_executable(omega_pipeline_tests tests/pipeline_tests.cpp)
target_link_libraries(omega_pipeline_tests PRIVATE omega_pipeline)

enable_testing()
add_test(NAME omega_pipeline_tests COMMAND omega_pipeline_tests)
