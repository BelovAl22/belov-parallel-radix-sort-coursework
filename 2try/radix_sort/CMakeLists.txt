cmake_minimum_required(VERSION 3.16)

project(radix_sort
        VERSION 1.0.0
        DESCRIPTION "Serial and OpenMP-parallel radix (counting) sort of uint16_t arrays"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Опции
# ---------------------------------------------------------------------------
option(RADIX_WIDE_COUNTERS "Use 64-bit counters (needed only for N > 4294967295 elements)" OFF)
option(RADIX_NATIVE_ARCH   "Add -march=native on GCC/Clang (the binary will not be portable)" OFF)
option(RADIX_BUILD_TESTS   "Build the test suite (radix_tests) and register it in CTest" ON)
option(RADIX_LARGE_TESTS   "Also register the 10^8-element test in CTest (needs ~400 MB of RAM)" OFF)
option(ENABLE_GOOGLE_BENCHMARK
       "Build the optional Google Benchmark target 'radix_gbench' (skipped if the library is not found)" OFF)

# ---------------------------------------------------------------------------
# Стандарт, тип сборки
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# OpenMP (обязательная зависимость). На GCC/Clang добавляет -fopenmp, на MSVC — /openmp:
# вручную прописывать флаги не нужно.
#
# macOS: Apple Clang поставляется БЕЗ OpenMP, поэтому find_package(OpenMP) сам его не находит.
# Нужен libomp из Homebrew (brew install libomp). Ниже его путь определяется автоматически
# (brew --prefix libomp, затем стандартные каталоги Homebrew для Apple Silicon и Intel) и
# подставляются те же флаги, которые иначе пришлось бы задавать вручную через
# -DOpenMP_CXX_FLAGS / -DOpenMP_CXX_LIB_NAMES / -DOpenMP_omp_LIBRARY. Если эти переменные уже
# заданы пользователем, они не перезаписываются.
# ---------------------------------------------------------------------------
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND NOT OpenMP_CXX_FLAGS)
  set(_radix_libomp_prefix "")
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE _radix_brew_prefix
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  foreach(_radix_candidate IN ITEMS "${_radix_brew_prefix}" "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    if(NOT _radix_libomp_prefix AND _radix_candidate AND EXISTS "${_radix_candidate}/include/omp.h")
      set(_radix_libomp_prefix "${_radix_candidate}")
    endif()
  endforeach()

  if(_radix_libomp_prefix)
    message(STATUS "Apple Clang has no built-in OpenMP; using libomp from ${_radix_libomp_prefix}")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_radix_libomp_prefix}/include"
        CACHE STRING "OpenMP compile flags (libomp from Homebrew)" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP library names" FORCE)
    set(OpenMP_omp_LIBRARY "${_radix_libomp_prefix}/lib/libomp.dylib"
        CACHE FILEPATH "libomp from Homebrew" FORCE)
  else()
    message(FATAL_ERROR
      "Apple Clang has no built-in OpenMP and libomp was not found.\n"
      "Install it and configure again from a clean build directory:\n"
      "    brew install libomp\n"
      "    rm -rf build\n"
      "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
      "(Alternatively use a compiler with OpenMP, e.g. 'brew install gcc' and -DCMAKE_CXX_COMPILER=g++-14.)")
  endif()
endif()

find_package(OpenMP REQUIRED)

# Предупреждения компилятора и особенности MSVC для конкретной цели.
function(radix_configure_target target)
  if(MSVC)
    # /utf-8: исходники в UTF-8 (русские комментарии); C4324: структуры с alignas(64) дополняются
    # до кэш-линии намеренно.
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /wd4324)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if(RADIX_NATIVE_ARCH)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Библиотека с алгоритмами, вводом-выводом, генератором, проверкой и бенчмарком
# ---------------------------------------------------------------------------
add_library(radix_core STATIC
  src/radix_sort.cpp
  src/io.cpp
  src/data_gen.cpp
  src/verify.cpp
  src/sysinfo.cpp
  src/benchmark.cpp)
target_include_directories(radix_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(radix_core PUBLIC cxx_std_17)
target_link_libraries(radix_core PUBLIC OpenMP::OpenMP_CXX)
if(RADIX_WIDE_COUNTERS)
  # PUBLIC: тип Count обязан совпадать во всех единицах трансляции (иначе нарушение ODR).
  target_compile_definitions(radix_core PUBLIC RADIX_WIDE_COUNTERS)
endif()
radix_configure_target(radix_core)

# ---------------------------------------------------------------------------
# Консольное приложение
# ---------------------------------------------------------------------------
add_executable(sort_app src/main.cpp)
target_link_libraries(sort_app PRIVATE radix_core)
radix_configure_target(sort_app)
install(TARGETS sort_app RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Тесты (CTest)
# ---------------------------------------------------------------------------
if(RADIX_BUILD_TESTS)
  enable_testing()

  add_executable(radix_tests tests/test_radix_sort.cpp)
  target_link_libraries(radix_tests PRIVATE radix_core)
  radix_configure_target(radix_tests)

  add_test(NAME unit_tests COMMAND radix_tests)

  # Сквозные проверки приложения (ненулевой код возврата = провал теста).
  add_test(NAME cli_benchmark_scatter
           COMMAND sort_app --mode benchmark --generate 2M --threads 4 --repeats 2 --verify)
  add_test(NAME cli_benchmark_fill
           COMMAND sort_app --mode benchmark --generate 2M --threads 3 --algorithm fill --repeats 2 --verify)
  add_test(NAME cli_single_thread
           COMMAND sort_app --mode parallel --generate 100k --threads 1 --distribution few-values --repeats 1 --verify)
  add_test(NAME cli_empty_array
           COMMAND sort_app --mode benchmark --generate 0 --threads 4 --verify)
  add_test(NAME cli_rejects_bad_arguments
           COMMAND sort_app --threads 0)
  set_tests_properties(cli_rejects_bad_arguments PROPERTIES WILL_FAIL TRUE)
  add_test(NAME cli_quiet_mode
           COMMAND sort_app --mode serial --generate 1M --quiet --verify --repeats 1)
  if(NOT RADIX_WIDE_COUNTERS)
    # N, не помещающееся в 32-битные счётчики, отвергается сразу — до генерации данных.
    add_test(NAME cli_rejects_oversized_n
             COMMAND sort_app --mode serial --generate 5G)
    set_tests_properties(cli_rejects_oversized_n PROPERTIES WILL_FAIL TRUE)
  endif()

  if(RADIX_LARGE_TESTS)
    add_test(NAME unit_tests_large COMMAND radix_tests --large)
    set_tests_properties(unit_tests_large PROPERTIES LABELS large TIMEOUT 1800)
  endif()
endif()

# ---------------------------------------------------------------------------
# Необязательный Google Benchmark. Отсутствие библиотеки НЕ ломает сборку основного проекта.
# ---------------------------------------------------------------------------
if(ENABLE_GOOGLE_BENCHMARK)
  find_package(benchmark CONFIG QUIET)
  if(benchmark_FOUND)
    add_executable(radix_gbench bench/gbench_radix_sort.cpp)
    target_link_libraries(radix_gbench PRIVATE radix_core benchmark::benchmark)
    radix_configure_target(radix_gbench)
  else()
    message(WARNING
      "ENABLE_GOOGLE_BENCHMARK=ON, but Google Benchmark was not found: "
      "target 'radix_gbench' is skipped. sort_app and radix_tests are built as usual.")
  endif()
endif()
