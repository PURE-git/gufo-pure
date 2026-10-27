add_executable(ornith122b_load_test load_test.cpp)
target_link_libraries(ornith122b_load_test PRIVATE gufo_ornith122b gufo_core)
add_test(NAME ornith122b.load COMMAND ornith122b_load_test)
set_tests_properties(ornith122b.load PROPERTIES LABELS "ornith122b;cpu;model")
