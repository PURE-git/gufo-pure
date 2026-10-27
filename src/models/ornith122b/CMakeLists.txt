# Ornith 1.5 122B (qwen35moe) model package.
add_library(gufo_ornith122b STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_ornith122b PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ornith122b PUBLIC gufo_core Threads::Threads)
