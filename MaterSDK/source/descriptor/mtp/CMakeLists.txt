# 3. 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtp)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtp)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtp)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. compile



# 6. compile test
if (BUILD_TEST)
    add_executable(test_radial_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_radial_basis.cc)
    target_link_libraries(test_radial_basis gtest gtest_main)

    add_executable(test_angular_part ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_angular_part.cc)
    target_link_libraries(test_angular_part gtest gtest_main)

    add_executable(test_level ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_level.cc)
    target_link_libraries(test_level gtest gtest_main)


    gtest_discover_tests(test_radial_basis)
    gtest_discover_tests(test_angular_part)
endif()