cmake_minimum_required(VERSION 3.5.0)
cmake_policy(SET CMP0069 NEW)
project(dhnsw VERSION 0.1.0 LANGUAGES C CXX)

# ============================================
# Compiler settings
# ============================================
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-O3 -march=native -funroll-loops -ffast-math -fno-math-errno)
add_definitions(-DBOOST_COROUTINES_NO_DEPRECATION_WARNING)

# Link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
add_link_options(-flto)
message(STATUS "IPO (LTO) is enabled: ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")

# ============================================
# Include / link directories
# ============================================
# System and local paths (adjust LOCAL_PREFIX if deps are installed elsewhere)
set(LOCAL_PREFIX "$ENV{HOME}/local" CACHE PATH "Prefix for locally installed libraries")
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${LOCAL_PREFIX}/include
    /usr/include
    /usr/local/include
)
link_directories(
    ${LOCAL_PREFIX}/lib
    /usr/local/lib
)

# IBVerbs (RDMA)
set(IBVerbs_INCLUDE_DIR /usr/include)
set(IBVerbs_LIBRARY /usr/lib/libibverbs.so)
include_directories(${IBVerbs_INCLUDE_DIR})

# ============================================
# Project directories
# ============================================
set(dhnsw_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/dhnsw)
set(dhnsw_bench_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/bench)
set(deps_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
set(util_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/util)

include_directories(
    "${dhnsw_PUBLIC_INCLUDE_DIR}"
    "${deps_PUBLIC_INCLUDE_DIR}"
    "${util_PUBLIC_INCLUDE_DIR}"
    "${dhnsw_bench_INCLUDE_DIR}"
    "./"
)

# ============================================
# Source file collection
# ============================================
file(GLOB_RECURSE FAISS_SOURCES "${deps_PUBLIC_INCLUDE_DIR}/faiss/*.cpp")
file(GLOB_RECURSE RLIBV2_SOURCES "${deps_PUBLIC_INCLUDE_DIR}/rlibv2/*.cpp")
file(GLOB_RECURSE UTIL_SOURCES "${util_PUBLIC_INCLUDE_DIR}/*.cpp")
file(GLOB_RECURSE DHNSW_SOURCES "${dhnsw_PUBLIC_INCLUDE_DIR}/*.cpp")

set(ALL_SOURCES ${FAISS_SOURCES} ${RLIBV2_SOURCES} ${DHNSW_SOURCES} ${UTIL_SOURCES})

# ============================================
# Dependencies
# ============================================
add_subdirectory(deps/r2)

# FAISS library
add_library(faiss ${FAISS_SOURCES})
set_target_properties(faiss PROPERTIES LINK_SEARCH_START_STATIC TRUE)
target_compile_options(faiss PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(faiss ${OpenBLAS_LIBRARIES})

# Core DHNSW library
add_library(dhnsw_library ${ALL_SOURCES})
target_include_directories(dhnsw_library PRIVATE ${dhnsw_PUBLIC_INCLUDE_DIR} ${deps_PUBLIC_INCLUDE_DIR})

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dhnsw_library PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

# BLAS & LAPACK
find_package(BLAS REQUIRED)
target_link_libraries(dhnsw_library PUBLIC ${BLAS_LIBRARIES})

find_package(LAPACK REQUIRED)
target_link_libraries(dhnsw_library PUBLIC ${LAPACK_LIBRARIES})

# Protobuf & gRPC
find_package(Protobuf REQUIRED)
find_package(gflags REQUIRED)

# Try CMake config first; fall back to pkg-config for system-installed gRPC
# Exclude deps/grpc build dir which may contain incomplete config files
set(CMAKE_IGNORE_PATH "${CMAKE_SOURCE_DIR}/deps/grpc/cmake/build")
find_package(gRPC CONFIG QUIET)
unset(CMAKE_IGNORE_PATH)

if(gRPC_FOUND AND TARGET gRPC::grpc++)
    message(STATUS "Found gRPC via CMake config")
    set(GRPC_LIBRARIES gRPC::grpc++ gRPC::grpc)
    set(GRPC_REFLECTION_LIB gRPC::grpc++_reflection)
else()
    message(STATUS "gRPC CMake config not found, trying pkg-config...")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPC REQUIRED grpc++ grpc)
    include_directories(${GRPC_INCLUDE_DIRS})
    link_directories(${GRPC_LIBRARY_DIRS})
    set(GRPC_LIBRARIES ${GRPC_LINK_LIBRARIES})
    set(GRPC_REFLECTION_LIB grpc++_reflection)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(src/generated)

target_link_libraries(dhnsw_library PUBLIC protobuf::libprotobuf protobuf::libprotoc ${GRPC_LIBRARIES})
target_link_libraries(dhnsw_library PUBLIC pthread dl)

# ============================================
# Build mode option
# ============================================
# OFF = search-only (run_server + run_client)
# ON  = full build with reconstruction executables
option(ENABLE_RECONSTRUCTION "Build reconstruction executables" OFF)
message(STATUS "ENABLE_RECONSTRUCTION: ${ENABLE_RECONSTRUCTION}")

# ============================================
# Common source fragments
# ============================================
set(PROTO_SOURCES src/generated/dhnsw.grpc.pb.cc src/generated/dhnsw.pb.cc)
set(COMMON_DEPS deps/r2/src/logging.cc)
set(COMMON_LINK_LIBS
    gflags
    dhnsw_library
    ${GRPC_LIBRARIES}
    ${GRPC_REFLECTION_LIB}
    protobuf::libprotobuf
    pthread
    dl
    ${IBVerbs_LIBRARY}
    r2
    boost_coroutine
    boost_context
)

# ============================================
# Executables: Server & Client (always built)
# ============================================
set(SERVER_SOURCES src/bench/search_server.cc  ${PROTO_SOURCES} ${COMMON_DEPS})
add_executable(run_server ${SERVER_SOURCES})
target_link_libraries(run_server ${COMMON_LINK_LIBS})

set(CLIENT_SOURCES src/bench/search_client_pipelined_reuse_thread.cc ${PROTO_SOURCES} ${COMMON_DEPS})
add_executable(run_client ${CLIENT_SOURCES})
target_link_libraries(run_client ${COMMON_LINK_LIBS})

# ============================================
# Executables: Reconstruction
# ============================================
if(ENABLE_RECONSTRUCTION)
    # Reconstruction implementation library
    add_library(reconstruction_lib src/dhnsw/reconstruction.cpp)
    target_link_libraries(reconstruction_lib dhnsw_library)

    # Memory server with reconstruction support
    set(RECONSTRUCTION_SERVER_SOURCES src/bench/search_server_reconstruction.cc ${PROTO_SOURCES} ${COMMON_DEPS})
    add_executable(reconstruction_server ${RECONSTRUCTION_SERVER_SOURCES})
    target_link_libraries(reconstruction_server ${COMMON_LINK_LIBS} reconstruction_lib)

    # Master insert client with reconstruction triggering
    set(MASTER_INSERT_SOURCES src/bench/client_insert_reconstruction.cc ${PROTO_SOURCES} ${COMMON_DEPS})
    add_executable(master_insert ${MASTER_INSERT_SOURCES})
    target_link_libraries(master_insert ${COMMON_LINK_LIBS} reconstruction_lib)

    # Worker search client with reconstruction handling
    set(WORKER_SEARCH_SOURCES src/bench/search_client_reconstruction.cc ${PROTO_SOURCES} ${COMMON_DEPS})
    add_executable(worker_search ${WORKER_SEARCH_SOURCES})
    target_link_libraries(worker_search ${COMMON_LINK_LIBS} reconstruction_lib)
endif()

# ============================================
# Packaging
# ============================================
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
