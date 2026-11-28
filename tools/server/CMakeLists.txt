include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-cache.cpp
    server-cache.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
    server-realtime.cpp
    server-realtime.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
    server-slot-sched.cpp
    server-slot-sched.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE vendor::hash)

# the monitor page (monitor.html) as bytes in a generated header, no UI download involved
set(MONITOR_HTML ${CMAKE_CURRENT_SOURCE_DIR}/monitor.html)
set(MONITOR_HPP  ${CMAKE_CURRENT_BINARY_DIR}/monitor-page.hpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MONITOR_HTML})
file(READ ${MONITOR_HTML} MONITOR_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," MONITOR_HEX "${MONITOR_HEX}")
file(WRITE ${MONITOR_HPP} "static const unsigned char monitor_page[] = {${MONITOR_HEX}};\n")

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-connect.cpp
    server-connect.h
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
    server-models-sched.cpp
    server-models-sched.h
    server-monitor.cpp
    server-monitor.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
