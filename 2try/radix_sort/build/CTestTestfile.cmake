# CMake generated Testfile for 
# Source directory: /Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort
# Build directory: /Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unit_tests "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/radix_tests")
set_tests_properties(unit_tests PROPERTIES  _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;125;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_benchmark_scatter "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--mode" "benchmark" "--generate" "2M" "--threads" "4" "--repeats" "2" "--verify")
set_tests_properties(cli_benchmark_scatter PROPERTIES  _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;128;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_benchmark_fill "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--mode" "benchmark" "--generate" "2M" "--threads" "3" "--algorithm" "fill" "--repeats" "2" "--verify")
set_tests_properties(cli_benchmark_fill PROPERTIES  _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;130;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_single_thread "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--mode" "parallel" "--generate" "100k" "--threads" "1" "--distribution" "few-values" "--repeats" "1" "--verify")
set_tests_properties(cli_single_thread PROPERTIES  _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;132;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_empty_array "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--mode" "benchmark" "--generate" "0" "--threads" "4" "--verify")
set_tests_properties(cli_empty_array PROPERTIES  _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;134;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_rejects_bad_arguments "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--threads" "0")
set_tests_properties(cli_rejects_bad_arguments PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;136;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_quiet_mode "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--mode" "serial" "--generate" "1M" "--quiet" "--verify" "--repeats" "1")
set_tests_properties(cli_quiet_mode PROPERTIES  _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;139;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
add_test(cli_rejects_oversized_n "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/build/sort_app" "--mode" "serial" "--generate" "5G")
set_tests_properties(cli_rejects_oversized_n PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;143;add_test;/Users/alexandr/Documents/GitHub/belov-parallel-radix-sort-coursework/2try/radix_sort/CMakeLists.txt;0;")
