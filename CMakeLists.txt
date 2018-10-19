message(STATUS "Universal CMakeLists.txt, version 1.0.0")
message(STATUS "For more information, please see https://martin.iat.tu-berlin.de/projects/cmakepackage/wiki")

cmake_policy(SET CMP0057 NEW)

macro(main)
	cmake_minimum_required(VERSION 3.10.2)
	include_config_version_file(${CMAKE_SOURCE_DIR})
	set_compiler()
	set_variables()
	set_options()
	include_config_file(${CMAKE_SOURCE_DIR})
	set_flags()
	set_environment_paths()
	create_doxygen_project(${PROJECT_NAME} ${CMAKE_SOURCE_DIR})
	#create_installer(${PROJECT_NAME})
endmacro()

macro(include_config_version_file DIRECTORY)
	get_config_version_file(${DIRECTORY})
	if(DEFINED CONFIG_VERSION_FILE)
		include(${CONFIG_VERSION_FILE})
	endif()
endmacro()

macro(include_config_file DIRECTORY)
	get_config_file(${DIRECTORY})
	if(DEFINED CONFIG_FILE)
		include(${CONFIG_FILE})
	endif()
endmacro()

macro(get_config_file DIRECTORY)
	unset(CONFIG_FILE)
	file(GLOB CONFIG_FILES ${DIRECTORY}/*Config.cmake)
	list(LENGTH CONFIG_FILES CONFIG_FILES_LENGTH)
	if(CONFIG_FILES_LENGTH GREATER 0)
		list(GET CONFIG_FILES 0 CONFIG_FILE)
	endif()
endmacro()

macro(get_config_version_file DIRECTORY)
	unset(CONFIG_VERSION_FILE)
	file(GLOB CONFIG_VERSION_FILES ${DIRECTORY}/*ConfigVersion.cmake)
	list(LENGTH CONFIG_VERSION_FILES CONFIG_VERSION_FILES_LENGTH)
	if(CONFIG_VERSION_FILES_LENGTH GREATER 0)
		list(GET CONFIG_VERSION_FILES 0 CONFIG_VERSION_FILE)
	endif()
endmacro()

macro(set_variables)
	set(Debug_KEYWORD debug)
	set(Debug_LOWER_CASE debug)
	set(Debug_UPPER_CASE DEBUG)
	set(Release_KEYWORD optimized)
	set(Release_LOWER_CASE release)
	set(Release_UPPER_CASE RELEASE)
	set(RelWithDebInfo_KEYWORD optimized)
	set(RelWithDebInfo_LOWER_CASE relwithdebinfo)
	set(RelWithDebInfo_UPPER_CASE RELWITHDEBINFO)
	set(MinSizeRel_KEYWORD optimized)
	set(MinSizeRel_LOWER_CASE minsizerel)
	set(MinSizeRel_UPPER_CASE MINSIZEREL)
	set(dynamic_UPPER_CASE DYNAMIC)
	set(dynamic_KEYWORD SHARED)
	set(static_UPPER_CASE STATIC)
	set(static_KEYWORD STATIC)
	set(DEPENDENCY_ARGUMENTS
		COMPONENTS
		HEADERS
		DEBUG_DYNAMIC_LIBS DEBUG_DYNAMIC_DLLS DEBUG_DYNAMIC_PDBS DEBUG_DYNAMIC_SOS
		DEBUG_STATIC_LIBS DEBUG_STATIC_PDBS DEBUG_STATIC_AS
		RELEASE_DYNAMIC_LIBS RELEASE_DYNAMIC_DLLS RELEASE_DYNAMIC_SOS
		RELEASE_STATIC_LIBS RELEASE_STATIC_AS
	)
	set(DEFAULT_HEADERS
		*.h
		*.hpp
		src/*.h
		src/*.hpp
		include/*.h
		include/*.hpp
	)
	set(DEFAULT_SOURCES
		*.cpp
		*.py
		*.C
		src/*.cpp
		src/*.py
		src/*.C
	)
	#set(INCLUDE_DUBLICTES)
	#set(LINK_DUBLICTES)
endmacro()

macro(set_compiler)
	if(MINGW)
		set(COMPILER "mingw")
	elseif(MSVC)
		if(MSVC_VERSION EQUAL 1700)
			set(COMPILER "vc110")
			set(COMPILER_VERSION "11.0")
		elseif(MSVC_VERSION EQUAL 1800)
			set(COMPILER "vc120")
			set(COMPILER_VERSION "12.0")
		elseif(MSVC_VERSION EQUAL 1900)
			set(COMPILER "vc140")
			set(COMPILER_VERSION "14.0")
		elseif(MSVC_VERSION EQUAL 1910)
			set(COMPILER "vc141")
			set(COMPILER_VERSION "15.0")
		else()
			message(FATAL_ERROR "The selected Microsoft Visual Studio version is currently not supported." )
		endif()
	elseif(UNIX)
		set(COMPILER "gcc")
	else()
		message(FATAL_ERROR "This generator is currently not supported." )
	endif()
endmacro()

macro(set_architecture TARGET_NAME)
	if(${ARGC} LESS 2)
		set(ARCHITECTURES x64 x86)
	else()
		set(ARCHITECTURES ${ARGV1})
	endif() 
	list(GET ARCHITECTURES 0 DEFAULT_ARCHITECTURE)
	set(${TARGET_NAME}_ARCHITECTURE ${DEFAULT_ARCHITECTURE} CACHE STRING "${TARGET_NAME} architecture")
	set_property(CACHE ${TARGET_NAME}_ARCHITECTURE PROPERTY STRINGS "${ARCHITECTURES}")
	if (${TARGET_NAME}_ARCHITECTURE STREQUAL "x86")
		add_definitions(-D_USE_32BIT_TIME_T)
	endif()	
	message(STATUS "${TARGET_NAME} architecture: ${${TARGET_NAME}_ARCHITECTURE}")
endmacro()

macro(set_options)
	option(GENERAL_COPY_DEPENDENCIES_TO_BUILD_DIRECTORY "Copy dependencies to install directory?" ON)
	option(GENERAL_COPY_DEPENDENCIES_TO_INSTALL_DIRECTORY "Copy dependencies to install directory?" ON)
	option(GENERAL_CREATE_DOXYGEN_DOCUMENTATION "Create Doxygen documentation?" ON)
	option(GENERAL_RECURSIVELY_ADD_PREBUILT_DEPENDENCIES "Recursively add prebuilt dependencies" OFF)
endmacro()

macro(set_flags)
	set_compiler_flags()
	set_linker_flags()
endmacro()

macro(set_compiler_flags)
	if(NOT MSVC)
		set(CMAKE_CXX_FLAGS "-Wall") # -std=c++11 is specified where needed
		set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
		set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	else()
		if (${PROJECT_NAME}_CHARACTER_SET STREQUAL "Unicode")
			set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D _UNICODE /D UNICODE")
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D _UNICODE /D UNICODE")
		endif()
		# DO NOT ENABLE THIS IN CASE OF USING CUDA RELATED FUNCTIONS! (cannot be compiled then because compiler flags seem to change something else also):
		# set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG /Od")
		# set(CMAKE_CXX_FLAGS_RELEASE "/Ox")
	endif()
	add_definitions(-DBOOST_ALL_NO_LIB)
endmacro()

macro(set_linker_flags)
	if (${PROJECT_NAME}_RUNTIME STREQUAL static)
		string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
		string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	endif()
endmacro()

macro(create_project TARGET_NAME TARGET_SOURCE_DIR)
	if(NOT TARGET ${TARGET_NAME})
		if(NOT ${TARGET_NAME}_TYPE STREQUAL Project)
			set(${TARGET_NAME}_INCLUDE_DIRS ${TARGET_SOURCE_DIR} ${TARGET_SOURCE_DIR}/src ${TARGET_SOURCE_DIR}/include)
			get_header_files(${TARGET_NAME} ${TARGET_SOURCE_DIR})
			get_source_files(${TARGET_NAME} ${TARGET_SOURCE_DIR})
			create_target(${TARGET_NAME} ${TARGET_SOURCE_DIR})
			install_target(${TARGET_NAME})
			create_test_projects(${TARGET_NAME})
		endif()
	endif()
endmacro()

macro(create_target TARGET_NAME TARGET_SOURCE_DIR)
	if(${TARGET_NAME}_TYPE STREQUAL Application)
		create_application(${TARGET_NAME} ${TARGET_SOURCE_DIR})
	elseif(${TARGET_NAME}_TYPE STREQUAL Library)
		create_library(${TARGET_NAME} ${TARGET_SOURCE_DIR})
	else()
		message(FATAL_ERROR "Please set the TYPE to Application or Library in the file ${TARGET_NAME}Config.cmake" )
	endif()
	if(NOT ${TARGET_NAME}_LIBRARY_TYPE STREQUAL interface)
		target_compile_features(${TARGET_NAME} PUBLIC cxx_range_for)
	endif()
endmacro()

macro(get_header_files TARGET_NAME TARGET_SOURCE_DIR)
	list(APPEND ${TARGET_NAME}_HEADERS ${DEFAULT_HEADERS})
	unset(${TARGET_NAME}_HEADER_FILES)
	foreach(${TARGET_NAME}_HEADER_EXPRESSION ${${TARGET_NAME}_HEADERS})
		file(GLOB ${TARGET_NAME}_HEADER_EXPRESSION_FILES ${TARGET_SOURCE_DIR}/${${TARGET_NAME}_HEADER_EXPRESSION})
		list(APPEND ${TARGET_NAME}_HEADER_FILES ${${TARGET_NAME}_HEADER_EXPRESSION_FILES})
	endforeach()
	set(${TARGET_NAME}_HEADERS ${${TARGET_NAME}_HEADER_FILES})
endmacro()

macro(get_source_files TARGET_NAME TARGET_SOURCE_DIR)
	list(APPEND ${TARGET_NAME}_SOURCES ${DEFAULT_SOURCES})
	unset(${TARGET_NAME}_SOURCE_FILES)
	foreach(${TARGET_NAME}_SOURCE_EXPRESSION ${${TARGET_NAME}_SOURCES})
		file(GLOB ${TARGET_NAME}_SOURCE_EXPRESSION_FILES ${TARGET_SOURCE_DIR}/${${TARGET_NAME}_SOURCE_EXPRESSION})
		list(APPEND ${TARGET_NAME}_SOURCE_FILES ${${TARGET_NAME}_SOURCE_EXPRESSION_FILES})
	endforeach()
	set(${TARGET_NAME}_SOURCES ${${TARGET_NAME}_SOURCE_FILES})
endmacro()

macro(create_application TARGET_NAME TARGET_SOURCE_DIR)
	add_executable(${TARGET_NAME} "${${TARGET_NAME}_HEADERS}" ${${TARGET_NAME}_SOURCES})
endmacro()

macro(create_library TARGET_NAME TARGET_SOURCE_DIR)
	if(${TARGET_NAME}_LIBRARY_TYPE STREQUAL interface)
		add_interface_library(${TARGET_NAME} ${TARGET_SOURCE_DIR})
	else()
		#message("add_library")
		#message(${TARGET_NAME})
		add_library(${TARGET_NAME} ${${${TARGET_NAME}_LIBRARY_TYPE}_KEYWORD} ${${TARGET_NAME}_HEADERS} ${${TARGET_NAME}_SOURCES})
	endif()
	set(${TARGET_NAME}_LIBRARIES ${TARGET_NAME})
endmacro()

macro(add_interface_library TARGET_NAME TARGET_SOURCE_DIR)
	message(${TARGET_NAME})
	add_library(${TARGET_NAME} INTERFACE)
	target_include_directories(${TARGET_NAME} INTERFACE ${TARGET_SOURCE_DIR})
	target_sources(${TARGET_NAME} INTERFACE ${${TARGET_NAME}_HEADERS})
endmacro()

macro(add_dependency TARGET_NAME DEPENDENCY)
	#message("add_dependency")
	#message("TARGET " ${TARGET_NAME})
	#message("DEPENDENCY " ${DEPENDENCY})
	#message("PROJECT_NAME " ${PROJECT_NAME})
	string(TOLOWER ${DEPENDENCY} ${DEPENDENCY}_LOWER_CASE)
	set(${DEPENDENCY}_TARGET ${TARGET_NAME})
	if((${DEPENDENCY} STREQUAL "ws2_32.lib") OR (${DEPENDENCY} STREQUAL "opengl32.lib") OR (DEPENDENCY STREQUAL ${PROJECT_NAME}))
		#message("link_library_files_to_target ")
		link_library_files_to_target(${TARGET_NAME} "${DEPENDENCY}" )
	else()
		#message("find_prebuilt_or_subproject_dependency ")
		set(${DEPENDENCY}_DIR_CACHE ${${DEPENDENCY}_DIR})	
		find_prebuilt_or_subproject_dependency(${TARGET_NAME} ${DEPENDENCY})
		if(${DEPENDENCY}_FOUND)
			#message("FOUND")
			if(${DEPENDENCY}_PREBUILT)
				#message("add_prebuilt_dependency ")
				add_prebuilt_dependency(${TARGET_NAME} ${DEPENDENCY})
			else()
				#message("add_subproject_dependency ")
				add_subproject_dependency(${TARGET_NAME} ${DEPENDENCY})
			endif()
		else()
			#message("find_installed_dependency ")
			#find_deprecated_prebuilt_dependency(${DEPENDENCY})
			#if(${DEPENDENCY}_FOUND)
			#	configure_default_library(${DEPENDENCY})
			#	add_prebuilt_dependency(${TARGET_NAME} ${DEPENDENCY})
			#else()
				find_installed_dependency(${DEPENDENCY})
				if(${DEPENDENCY}_FOUND)
					configure_default_library(${DEPENDENCY})
					add_installed_dependency(${TARGET_NAME} ${DEPENDENCY})
				else()
					message("Please specify ${DEPENDENCY}_DIR.")
				endif()
			#endif()
		endif()
	endif()
endmacro()

macro(configure_default_library LIBRARY_NAME)
	configure(
		${LIBRARY_NAME}
		TYPE Library
		PREBUILT
		ARCHITECTURES ${${PROJECT_NAME}_ARCHITECTURE}
		LIBRARY_TYPES dynamic static
		RUNTIMES ${${PROJECT_NAME}_RUNTIMES}
		BUILD_TYPES ${${PROJECT_NAME}_BUILD_TYPES}
		MULTI_THREADED
	)		
endmacro()

macro(find_prebuilt_or_subproject_dependency TARGET_NAME DEPENDENCY)
	set(DEFAULT_${DEPENDENCY}_DIRS
		H:/repos/CameraAPI/install/cameraapi
		${${TARGET_NAME}_DIR}/${DEPENDENCY}
		C:/local/prebuilt/${${DEPENDENCY}_LOWER_CASE}-${${DEPENDENCY}_VERSION}-${COMPILER}-${${PROJECT_NAME}_ARCHITECTURE}
		C:/local/prebuilt/${${DEPENDENCY}_LOWER_CASE}-${${DEPENDENCY}_VERSION}-${COMPILER}
		C:/local/prebuilt/${${DEPENDENCY}_LOWER_CASE}-${${DEPENDENCY}_VERSION}
		C:/local/prebuilt/${${DEPENDENCY}_LOWER_CASE}
		/opt/${${DEPENDENCY}_LOWER_CASE}
		${${TARGET_NAME}_DIR}/../${DEPENDENCY}/install/${${DEPENDENCY}_LOWER_CASE}
		${${TARGET_NAME}_DIR}/../${DEPENDENCY}
		${PROJECT_SOURCE_DIR}/${DEPENDENCY}
		${PROJECT_SOURCE_DIR}/../${DEPENDENCY}/install/${${DEPENDENCY}_LOWER_CASE}
		${PROJECT_SOURCE_DIR}/../${DEPENDENCY}
	)
	
	#message("FIND DEPENDENCY: " ${DEPENDENCY})
	#message("FIND VERSION: " ${${DEPENDENCY}_VERSION})
	#message("DEFAULT: " ${DEFAULT_${DEPENDENCY}_DIRS})
	#message("DEPENDENCY_DIR: ${DEPENDENCY_DIR}")
	
	
	find_package(
		${DEPENDENCY}
		${${DEPENDENCY}_VERSION}
		COMPONENTS ${${DEPENDENCY}_COMPONENTS}
		QUIET
		CONFIG
		PATHS ${DEFAULT_${DEPENDENCY}_DIRS}
	)
	

	if(${DEPENDENCY}_FOUND)
		#message("FOUND2: ${DEPENDENCY}")
	else()
		#message("NO FOUND2: ${DEPENDENCY}")
	endif()		
	
endmacro()

macro(checkout_branch DEPENDENCY BRANCH)
	foreach(DEFAULT_${DEPENDENCY}_DIR ${DEFAULT_${DEPENDENCY}_DIRS})
		if(EXISTS ${DEFAULT_${DEPENDENCY}_DIR}/.git)
			if(${DEPENDENCY}_VERSION)
				execute_process(
					COMMAND git branch --list ${BRANCH}
					WORKING_DIRECTORY ${DEFAULT_${DEPENDENCY}_DIR}
					OUTPUT_VARIABLE BRANCH_EXISTS
				)
				if(BRANCH_EXISTS)
					execute_process(
						COMMAND git checkout ${BRANCH}
						WORKING_DIRECTORY ${DEFAULT_${DEPENDENCY}_DIR}
						OUTPUT_VARIABLE BRANCH_CHECKOUT
					)
				endif()
			endif()
		endif()
	endforeach()
endmacro()

macro(add_subproject_dependency TARGET_NAME DEPENDENCY)
	
	message("add_subproject_dependency: ")
	message(${TARGET_NAME})
	message(${DEPENDENCY})
	
	
	
	if(${DEPENDENCY}_LIBRARY_TYPE STREQUAL interface)
		#message("interface")
		include_directories(${${DEPENDENCY}_INCLUDE_DIR} ${${DEPENDENCY}_INCLUDE_DIRS})
	else()
		#message("no interface")
		#message("${${DEPENDENCY}_INCLUDE_DIR}")
		#message("${${DEPENDENCY}_INCLUDE_DIRS}")
		#message("TEST")
		#message(${DEPENDENCY})
		#message("${${DEPENDENCY}_INCLUDE_DIRS}")
		#message("${${DEPENDENCY}_INCLUDE_DIR}")
		#target_include_directories(${DEPENDENCY} PUBLIC ${${DEPENDENCY}_INCLUDE_DIR} ${${DEPENDENCY}_INCLUDE_DIRS})
		target_include_directories(${TARGET_NAME} PUBLIC ${${DEPENDENCY}_INCLUDE_DIR} ${${DEPENDENCY}_INCLUDE_DIRS})
	endif()
	#link_library_files_to_target(${TARGET_NAME} "${${DEPENDENCY}_LIBRARIES}")
	target_link_libraries(${TARGET_NAME} ${${DEPENDENCY}_LIBRARIES})
	
	add_dependencies(${TARGET_NAME} ${DEPENDENCY})
	
	
	message("add_dependencies ${TARGET_NAME} ${DEPENDENCY}")
	message("LIBRARIES ${${DEPENDENCY}_LIBRARIES}")
	
endmacro()

macro(add_installed_dependency TARGET_NAME DEPENDENCY)
	set(DEPENDENCY_TARGET ${TARGET_NAME})
	while(${DEPENDENCY_TARGET}_PREBUILT)
		set(DEPENDENCY_TARGET ${${DEPENDENCY_TARGET}_TARGET})
	endwhile()
	if(NOT ${DEPENDENCY_TARGET}_TYPE STREQUAL "Project")
		target_include_directories(${DEPENDENCY_TARGET} PUBLIC "${${DEPENDENCY}_INCLUDE_DIR}" "${${DEPENDENCY}_INCLUDE_DIRS}")
		link_dependency(${DEPENDENCY_TARGET} ${DEPENDENCY})
	endif()
	copy_dependency_to_build_directory(${DEPENDENCY})
	install_dependency(${DEPENDENCY_TARGET} ${DEPENDENCY})
endmacro()

macro(find_deprecated_prebuilt_dependency DEPENDENCY)
	if(EXISTS "${${DEPENDENCY}_DIR}/info")
		set(${DEPENDENCY}_FOUND TRUE)
	else()
		if(EXISTS "${${DEPENDENCY}_DIR_CACHE}/info")
			set(${DEPENDENCY}_DIR ${${DEPENDENCY}_DIR_CACHE} CACHE PATH "The directory containing ${DEPENDENCY}." FORCE)
			set(${DEPENDENCY}_FOUND TRUE)
		else()
			set(${DEPENDENCY}_FOUND FALSE)
		endif()
	endif()
endmacro()

macro(add_prebuilt_dependency TARGET_NAME DEPENDENCY)
	message("add_prebuilt_dependency")
	set(DEPENDENCY_TARGET ${TARGET_NAME})
	while(${DEPENDENCY_TARGET}_PREBUILT)
		set(DEPENDENCY_TARGET ${${DEPENDENCY_TARGET}_TARGET})
	endwhile()
	if(NOT ${DEPENDENCY_TARGET}_TYPE STREQUAL "Project")
	
		#message("TEST2")
		#message("${${DEPENDENCY}_DIR}")
		set(${DEPENDENCY}_INCLUDE_DIR ${${DEPENDENCY}_DIR}/include)
		#message("TEST1")
		#message("${${DEPENDENCY}_INCLUDE_DIR}")
		#message("TEST1X")
		
		# FoB
		#target_include_directories(${DEPENDENCY_TARGET} PUBLIC ${${DEPENDENCY}_INCLUDE_DIR})
		SUBDIRLIST(INCLUDE_SUBDIRS ${MY_CURRENT_DIR} ${${DEPENDENCY}_INCLUDE_DIR})
		FOREACH(item ${INCLUDE_SUBDIRS})
			if (NOT ${item} IN_LIST ${TARGET_NAME}_INCLUDE_DUBLICTES)
				target_include_directories(${DEPENDENCY_TARGET} PUBLIC ${${DEPENDENCY}_INCLUDE_DIR}/${item})
				#include_directories(${${DEPENDENCY}_INCLUDE_DIR}/${item})
				LIST(APPEND ${TARGET_NAME}_INCLUDE_DUBLICTES ${item})
				#message("SUBDIRLIST")
				#message("DEPENDENCY_TARGETX: ${DEPENDENCY_TARGET}")
				#message("TARGET_NAMEX: ${TARGET_NAME}")
			endif()
		ENDFOREACH()
		# FoB

		target_include_dependency_headers(${DEPENDENCY_TARGET} PUBLIC ${DEPENDENCY})
		#target_include_subdirectories(${DEPENDENCY_TARGET} ${${DEPENDENCY}_INCLUDE_DIR})
		set_prebuilt_dependency_dirs(${DEPENDENCY})
		find_dependency_files(${DEPENDENCY_TARGET} ${DEPENDENCY})
		#message("BUILD_TYPE")
		#message("${BUILD_TYPE}")
		link_libraries_to_target(${DEPENDENCY_TARGET} ${DEPENDENCY} ${BUILD_TYPE})
	endif()
	copy_dependency_to_build_directory(${DEPENDENCY})
	install_dependency(${DEPENDENCY_TARGET} ${DEPENDENCY})
endmacro()

macro(target_include_dependency_headers TARGET_NAME KEYWORD DEPENDENCY)
	foreach(HEADER ${${DEPENDENCY}_HEADERS})
		get_filename_component(DIRECTORY ${HEADER} PATH)
		target_include_directories(${TARGET_NAME} PUBLIC ${${DEPENDENCY}_DIR}/${DIRECTORY})
	endforeach()
endmacro()

macro(target_include_subdirectories DEPENDENCY_TARGET DIRECTORY)
	file(GLOB_RECURSE HEADERS ${DIRECTORY}/*.h)
    foreach(HEADER ${HEADERS})
        get_filename_component(SUBDIRECTORY ${HEADER} PATH)
        target_include_directories(${DEPENDENCY_TARGET} PUBLIC ${SUBDIRECTORY})
    endforeach()

#	file(GLOB_RECURSE ITEMS ${DIRECTORY}/*)
#    foreach(ITEM ${ITEMS})
#		get_filename_component(SUBDIRECTORY ${ITEM} PATH)
#		target_include_directories(${DEPENDENCY_TARGET} PUBLIC ${SUBDIRECTORY})
#    endforeach()
endmacro()

macro(find_installed_dependency DEPENDENCY)
	find_package(
		${DEPENDENCY}
		${${DEPENDENCY}_VERSION}
		COMPONENTS ${${DEPENDENCY}_COMPONENTS}
		QUIET
		MODULE
	)
	string(TOUPPER ${DEPENDENCY} ${DEPENDENCY}_UPPER_CASE)
	if(${${DEPENDENCY}_UPPER_CASE}_FOUND)
		set(${DEPENDENCY}_FOUND ${${${DEPENDENCY}_UPPER_CASE}_FOUND})
		
		set(${DEPENDENCY}_INCLUDE_DIRS ${${${DEPENDENCY}_UPPER_CASE}_INCLUDE_DIRS})
		set(${DEPENDENCY}_LIBRARIES ${${${DEPENDENCY}_UPPER_CASE}_LIBRARIES})
	endif()
endmacro()

macro(link_dependency TARGET_NAME DEPENDENCY)
	#message("TEST3")
	set_dependency_dirs(${DEPENDENCY})
	find_dependency_files(${TARGET_NAME} ${DEPENDENCY})
	
	link_libraries_to_target(${DEPENDENCY_TARGET} ${DEPENDENCY} ${BUILD_TYPE})
endmacro()

macro(set_dependency_dirs DEPENDENCY)
	foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
		if(DEFINED ${DEPENDENCY}_LIBRARY_DIR_${${BUILD_TYPE}_UPPER_CASE})
			set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR ${${DEPENDENCY}_LIBRARY_DIR_${${BUILD_TYPE}_UPPER_CASE}})
			set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR ${${DEPENDENCY}_LIBRARY_DIR_${${BUILD_TYPE}_UPPER_CASE}})
		endif()
	endforeach()
endmacro()

macro(link_library_files_to_target TARGET_NAME LIBRARIES)
	#message("link_library_files_to_target")
	#message("TARGET_NAME: " ${TARGET_NAME})
	#message("LIBRARIES: " ${LIBRARIES})
	if(NOT ${TARGET_NAME}_PREBUILT)
		if(NOT "${LIBRARIES}" STREQUAL "")
			if((${TARGET_NAME}_TYPE STREQUAL "Application") OR (${TARGET_NAME}_TYPE STREQUAL "Library"))
				#message("ARGV2")
				#message("${LIBRARIES}")
				#message(${${ARGV2}_KEYWORD})
				set(FILENAME)
				foreach(ITEM ${LIBRARIES})
					get_filename_component(FILENAME ${ITEM} NAME)
					if (NOT ${FILENAME} IN_LIST ${TARGET_NAME}_LINK_DUBLICTES)
						target_link_libraries(${TARGET_NAME} ${${ARGV2}_KEYWORD} ${ITEM})
						LIST(APPEND ${TARGET_NAME}_LINK_DUBLICTES ${FILENAME})
					endif()
				endforeach()
				#target_link_libraries(${TARGET_NAME} ${${ARGV2}_KEYWORD} "${LIBRARIES}")
			endif()
		endif()
	endif()
endmacro()


macro(link_libraries_to_target TARGET_NAME DEPENDENCY)
	if(WIN32)
	
		
		message("link_libraries_to_target")
		message(${TARGET_NAME})
		message(${DEPENDENCY})
		message("TYPE: ${${DEPENDENCY}_TYPE}")
		
		#message(${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS})
		# FoB: ERROR: Variable LIBRARIES is overridden by next build type without calling "link_library_files_to_target" to add libraries to project 
		foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
			#message(${BUILD_TYPE}_UPPER_CASE})
			#message("${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS}")
			set(LIBRARIES "${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS}")
			link_library_files_to_target(${TARGET_NAME} "${LIBRARIES}" ${BUILD_TYPE})
		endforeach()
	else()
		if(${DEPENDENCY}_LIBRARY_TYPE STREQUAL "dynamic")
			foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
				set(LIBRARIES "${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS}")
			endforeach()
		else()
			foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
				set(LIBRARIES "${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS}")
			endforeach()
		endif()
	endif()
	#message("LIBRARIES")
	#message("${ARGV2}")
	#link_library_files_to_target(${TARGET_NAME} "${LIBRARIES}" ${ARGV2})
endmacro()
		
macro(set_prebuilt_dependency_dirs DEPENDENCY)
	foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
		set_configuration_dir(${${DEPENDENCY}_ARCHITECTURE} ${${DEPENDENCY}_RUNTIME} ${BUILD_TYPE} ${${DEPENDENCY}_THREADS} ${${DEPENDENCY}_LIBRARY_TYPE})
		set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR ${${DEPENDENCY}_DIR}/lib/${CONFIGURATION_DIR})
		set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR ${${DEPENDENCY}_DIR}/bin/${CONFIGURATION_DIR})
	endforeach()
endmacro()

macro(set_configuration_dir ARCHITECTURE RUNTIME BUILD_TYPE THREADS)
	set(CONFIGURATION_DIR ${COMPILER}/${ARCHITECTURE}/${ARGV4}/${RUNTIME}_runtime/${${BUILD_TYPE}_LOWER_CASE}/${THREADS})
	#set(CONFIGURATION_DIR ${COMPILER}/${${BUILD_TYPE}_LOWER_CASE}/${ARCHITECTURE}/${LIBRARY_TYPE}/${THREADS})
endmacro()

macro(copy_dependency_to_build_directory DEPENDENCY)
	if(GENERAL_COPY_DEPENDENCIES_TO_BUILD_DIRECTORY)
		foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
			file(
				COPY
					${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS}
					${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS}
					${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS}
				DESTINATION
					${CMAKE_BINARY_DIR}/${BUILD_TYPE}
			)
			copy_resolved_symlinks_to_build_directory(
				"${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS}"
				${CMAKE_BINARY_DIR}
			)
			file(
				COPY
					${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS}
				DESTINATION
					${CMAKE_BINARY_DIR}
			)
		endforeach()
	endif()
endmacro()

macro(copy_resolved_symlinks_to_build_directory SYMLINKS DESTINATION)
	foreach(SYMLINK ${SYMLINKS})
		get_filename_component(REALPATH ${SYMLINK} REALPATH)
		file(COPY ${REALPATH} DESTINATION ${DESTINATION})
	endforeach()
endmacro()

macro(create_test_projects TARGET_NAME)
	set(${TARGET_NAME}_TEST_DIR "${${TARGET_NAME}_DIR}/src/test")
	if(EXISTS ${${TARGET_NAME}_TEST_DIR})
		create_test_project(${TARGET_NAME} ${${TARGET_NAME}_TEST_DIR})	
		file(GLOB ITEMS ${${TARGET_NAME}_TEST_DIR}/*)
		foreach(ITEM ${ITEMS})
			if(IS_DIRECTORY ${ITEM})
				create_test_project(${TARGET_NAME} ${ITEM})
			endif()
		endforeach()
	endif()
endmacro()

macro(create_test_project TARGET_NAME TEST_DIR)
	set(TARGET_NAME_DIR ${TEST_DIR})
	include_config_version_file(${TEST_DIR})
	include_config_file(${TEST_DIR})
endmacro()

macro(create_doxygen_project TARGET_NAME TARGET_SOURCE_DIR)
	if(NOT ${TARGET_NAME}_TYPE STREQUAL "Project")
		if(GENERAL_CREATE_DOXYGEN_DOCUMENTATION)
			find_package(Doxygen)
			if(DOXYGEN_FOUND)
				configure_file(
					${TARGET_SOURCE_DIR}/doc/template/Doxyfile.in
					${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
				)
				add_custom_target(
					DOCUMENTATION
					${DOXYGEN_EXECUTABLE}
					${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
					COMMENT "Generating API documentation with Doxygen"
					VERBATIM
				)
			endif()
		endif()
	endif()
endmacro()

macro(install_target TARGET_NAME)
	install_headers(${TARGET_NAME})
	install_target_files(${TARGET_NAME})
	install_info_dir()
	install_data_dir()
	install_doc_dir()	
	create_config_file()
	create_config_version_file()
endmacro()

macro(install_info_dir)
	if(EXISTS ${PROJECT_SOURCE_DIR}/info)
		install(DIRECTORY info DESTINATION ${${PROJECT_NAME}_INSTALL_PREFIX})
	endif()
endmacro()

macro(install_data_dir)
	if(EXISTS ${CMAKE_BINARY_DIR}/data)
		install(DIRECTORY ${CMAKE_BINARY_DIR}/data DESTINATION ${${PROJECT_NAME}_INSTALL_PREFIX})
	endif()
endmacro()

macro(install_doc_dir)
	if(EXISTS ${CMAKE_BINARY_DIR}/doc)
		install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION ${${PROJECT_NAME}_INSTALL_PREFIX})
	endif()
endmacro()

macro(install_headers TARGET_NAME)
	foreach(HEADER ${${TARGET_NAME}_HEADERS})
		get_header_path(${TARGET_NAME} ${HEADER})
		#message("HEADER")
		#message("${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${HEADER_PATH}${PROJECT_NAME}/")
		# FoB chage path of install location for include headers
		#install(FILES ${HEADER} DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${HEADER_PATH})
		install(FILES ${HEADER} DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${HEADER_PATH}${PROJECT_NAME}/)
	endforeach()
endmacro()

macro(get_header_path TARGET_NAME HEADER)
	string(REPLACE "${${TARGET_NAME}_DIR}/include/src/" "" HEADER_PATH ${HEADER})
	string(REPLACE "${${TARGET_NAME}_DIR}/src/" "" HEADER_PATH ${HEADER})
	string(REPLACE "${${TARGET_NAME}_DIR}/" "" HEADER_PATH ${HEADER_PATH})
	get_filename_component(HEADER_PATH ${HEADER_PATH} PATH)
endmacro()

macro(install_target_files TARGET_NAME)
	foreach(BUILD_TYPE ${${TARGET_NAME}_BUILD_TYPES})
		set_configuration_dir(${${PROJECT_NAME}_ARCHITECTURE} ${${TARGET_NAME}_RUNTIME} ${BUILD_TYPE} ${${TARGET_NAME}_THREADS} ${${PROJECT_NAME}_LIBRARY_TYPE})
		install(
			TARGETS ${TARGET_NAME} 
			ARCHIVE DESTINATION ${${PROJECT_NAME}_INSTALL_LIBDIR}/${CONFIGURATION_DIR}
			LIBRARY DESTINATION ${${PROJECT_NAME}_INSTALL_BINDIR}/${CONFIGURATION_DIR}
			RUNTIME DESTINATION ${${PROJECT_NAME}_INSTALL_BINDIR}/${CONFIGURATION_DIR}
			ARCHIVE CONFIGURATIONS ${BUILD_TYPE}
			LIBRARY CONFIGURATIONS ${BUILD_TYPE}
			RUNTIME CONFIGURATIONS ${BUILD_TYPE}
		)
	endforeach()
endmacro()

macro(install_dependency TARGET_NAME DEPENDENCY)
	if(GENERAL_COPY_DEPENDENCIES_TO_INSTALL_DIRECTORY)
		install_dependency_files(${TARGET_NAME} ${DEPENDENCY})
		install_dependency_headers(${TARGET_NAME} ${DEPENDENCY})
	endif()
endmacro()

macro(install_dependency_files TARGET_NAME DEPENDENCY)
	foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
		set_configuration_dir(${${PROJECT_NAME}_ARCHITECTURE} ${${TARGET_NAME}_RUNTIME} ${BUILD_TYPE} ${${TARGET_NAME}_THREADS} ${${PROJECT_NAME}_LIBRARY_TYPE})
		install_dependency_libs(${DEPENDENCY} ${BUILD_TYPE} ${CONFIGURATION_DIR})
		install_dependency_pdbs(${DEPENDENCY} ${BUILD_TYPE} ${CONFIGURATION_DIR})
		install_dependency_dlls(${DEPENDENCY} ${BUILD_TYPE} ${CONFIGURATION_DIR})
		install_dependency_sos(${DEPENDENCY} ${BUILD_TYPE} ${CONFIGURATION_DIR})
		install_dependency_as(${DEPENDENCY} ${BUILD_TYPE} ${CONFIGURATION_DIR})
	endforeach()
endmacro()

macro(install_dependency_libs DEPENDENCY BUILD_TYPE CONFIGURATION_DIR)
	install(
		FILES ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_LIBDIR}/${CONFIGURATION_DIR}
		CONFIGURATIONS ${BUILD_TYPE}
	)
endmacro()

macro(install_dependency_pdbs DEPENDENCY BUILD_TYPE CONFIGURATION_DIR)
	install(
		FILES ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_LIBDIR}/${CONFIGURATION_DIR}
		CONFIGURATIONS ${BUILD_TYPE}
	)
endmacro()

macro(install_dependency_dlls DEPENDENCY BUILD_TYPE CONFIGURATION_DIR)
	install(
		FILES ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_BINDIR}/${CONFIGURATION_DIR}
		CONFIGURATIONS ${BUILD_TYPE}
	)
endmacro()

macro(install_dependency_sos DEPENDENCY BUILD_TYPE CONFIGURATION_DIR)
	install(
		FILES ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_BINDIR}/${CONFIGURATION_DIR}
		CONFIGURATIONS ${BUILD_TYPE}
	)
endmacro()

macro(install_dependency_as DEPENDENCY BUILD_TYPE CONFIGURATION_DIR)
	install(
		FILES ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_BINDIR}/${CONFIGURATION_DIR}
		CONFIGURATIONS ${BUILD_TYPE}
	)
endmacro()

macro(find_dependency_files TARGET_NAME DEPENDENCY)
	get_dependency_files(${DEPENDENCY})
	foreach(BUILD_TYPE ${${DEPENDENCY}_BUILD_TYPES})
		if(
			(NOT DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS) AND
			(NOT DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS) AND
			(NOT DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS)
		)
			find_dependency_components_files(${DEPENDENCY} ${BUILD_TYPE})
		endif()
		get_named_dependency_files(${DEPENDENCY} ${BUILD_TYPE})
		if(
			(NOT DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS) AND
			(NOT DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS) AND
			(NOT DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS)
		)
			get_all_dependency_files(${DEPENDENCY} ${BUILD_TYPE})
		endif()	
		list(APPEND ${PROJECT_NAME}_ENVIRONMENT_PATHS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR})
	endforeach()
endmacro()

macro(find_dependency_components_files DEPENDENCY BUILD_TYPE)
	if(DEFINED ${DEPENDENCY}_COMPONENTS)
		separate_arguments(${DEPENDENCY}_COMPONENTS)
		foreach(COMPONENT ${${DEPENDENCY}_COMPONENTS})
			string(TOUPPER ${COMPONENT} ${COMPONENT}_UPPER_CASE)
			find_dependency_component_files(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
		endforeach()
	endif()
endmacro()

macro(find_dependency_component_files DEPENDENCY BUILD_TYPE COMPONENT)
	if(${DEPENDENCY}_PREBUILT)
		guess_dependency_component_files(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	elseif(DEFINED ${DEPENDENCY}_${${COMPONENT}_UPPER_CASE}_LIBRARY_${${BUILD_TYPE}_UPPER_CASE})
		get_dependency_component_files(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	elseif(DEFINED ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR)
		guess_dependency_component_files(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	endif()
endmacro()

macro(get_all_dependency_files DEPENDENCY BUILD_TYPE)
	file(GLOB ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR}/*.lib)
	file(GLOB ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR}/*.pdb)
	file(GLOB ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*.dll)
	file(
		GLOB
		${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS
		${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${DEPENDENCY}*.so
		${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${${DEPENDENCY}_LOWER_CASE}*.so
	)
	file(
		GLOB
		${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS
		${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${DEPENDENCY}*.a
		${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${${DEPENDENCY}_LOWER_CASE}*.a
	)
endmacro()

macro(get_dependency_files DEPENDENCY)
	if(DEFINED ${DEPENDENCY}_LIBRARIES)
		separate_dependency_libraries(${DEPENDENCY})
		set(${DEPENDENCY}_MINSIZEREL_LIBS ${${DEPENDENCY}_RELEASE_LIBS})
		set(${DEPENDENCY}_RELWITHDEBINFO_LIBS ${${DEPENDENCY}_RELEASE_LIBS})
		if(${DEPENDENCY}_LIBRARY_TYPE STREQUAL dynamic)
			string(REPLACE ".lib" ".dll;" ${DEPENDENCY}_DEBUG_DLLS "${${DEPENDENCY}_DEBUG_LIBS}")
			string(REPLACE ".lib" ".dll;" ${DEPENDENCY}_RELEASE_DLLS "${${DEPENDENCY}_RELEASE_LIBS}")
			set(${DEPENDENCY}_RELWITHDEBINFO_DLLS ${${DEPENDENCY}_DEBUG_DLLS})
			set(${DEPENDENCY}_MINSIZEREL_DLLS ${${DEPENDENCY}_DEBUG_DLLS})
		endif()
		set(${DEPENDENCY}_MINSIZEREL_SOS ${${DEPENDENCY}_RELEASE_SOS})
		set(${DEPENDENCY}_RELWITHDEBINFO_SOS ${${DEPENDENCY}_RELEASE_SOS})
		set(${DEPENDENCY}_MINSIZEREL_AS ${${DEPENDENCY}_RELEASE_AS})
		set(${DEPENDENCY}_RELWITHDEBINFO_AS ${${DEPENDENCY}_RELEASE_AS})
	endif()
endmacro()

macro(separate_dependency_libraries DEPENDENCY)
	unset(BUILD_TYPE)
	foreach(LIBRARY ${${DEPENDENCY}_LIBRARIES})
		if(${LIBRARY} STREQUAL optimized)
			set(BUILD_TYPE Release)
		elseif(${LIBRARY} STREQUAL debug)
			set(BUILD_TYPE Debug)
		else()
			get_filename_component(EXTENSION "${LIBRARY}" EXT)
			string(SUBSTRING "${EXTENSION}" 1 -1 EXTENSION)
			string(TOUPPER "${EXTENSION}" EXTENSION)
			if(DEFINED BUILD_TYPE)
				list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_${EXTENSION}S ${LIBRARY})
			else()
				list(APPEND ${DEPENDENCY}_DEBUG_${EXTENSION}S ${LIBRARY})
				list(APPEND ${DEPENDENCY}_RELEASE_${EXTENSION}S ${LIBRARY})
			endif()
		endif()
	endforeach()
endmacro()

macro(get_named_dependency_files DEPENDENCY BUILD_TYPE)
	find_named_dependency_libs(${DEPENDENCY} ${BUILD_TYPE})
	find_named_dependency_pdbs(${DEPENDENCY} ${BUILD_TYPE})
	find_named_dependency_dlls(${DEPENDENCY} ${BUILD_TYPE})
	find_named_dependency_sos(${DEPENDENCY} ${BUILD_TYPE})
	find_named_dependency_as(${DEPENDENCY} ${BUILD_TYPE})
endmacro()

macro(find_named_dependency_libs DEPENDENCY BUILD_TYPE)
	set(DEPENDENCY_BUILD_AND_LIBRARY_TYPE ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_${${${DEPENDENCY}_LIBRARY_TYPE}_UPPER_CASE})
	if(DEFINED ${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_LIBS)
		foreach(DEPENDENCY_LIB ${${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_LIBS})
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR}/${DEPENDENCY_LIB})
		endforeach()
	endif()
endmacro()

macro(find_named_dependency_pdbs DEPENDENCY BUILD_TYPE)
	set(DEPENDENCY_BUILD_AND_LIBRARY_TYPE ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_${${${DEPENDENCY}_LIBRARY_TYPE}_UPPER_CASE})
	if(DEFINED ${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_PDBS)
		foreach(DEPENDENCY_PDB ${${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_PDBS})
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR}/${DEPENDENCY_PDB})
		endforeach()
	endif()
endmacro()

macro(find_named_dependency_dlls DEPENDENCY BUILD_TYPE)
	set(DEPENDENCY_BUILD_AND_LIBRARY_TYPE ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_${${${DEPENDENCY}_LIBRARY_TYPE}_UPPER_CASE})
	if(DEFINED ${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_DLLS)
		foreach(DEPENDENCY_DLL ${${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_DLLS})
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/${DEPENDENCY_DLL})
		endforeach()
	endif()
endmacro()

macro(find_named_dependency_sos DEPENDENCY BUILD_TYPE)
	set(DEPENDENCY_BUILD_AND_LIBRARY_TYPE ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_${${${DEPENDENCY}_LIBRARY_TYPE}_UPPER_CASE})
	if(DEFINED ${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_SOS)
		foreach(DEPENDENCY_SO ${${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_SOS})
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/${DEPENDENCY_SO})
		endforeach()
	endif()
endmacro()

macro(find_named_dependency_as DEPENDENCY BUILD_TYPE)
	set(DEPENDENCY_BUILD_AND_LIBRARY_TYPE ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_${${${DEPENDENCY}_LIBRARY_TYPE}_UPPER_CASE})
	if(DEFINED ${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_AS)
		foreach(DEPENDENCY_A ${${DEPENDENCY_BUILD_AND_LIBRARY_TYPE}_AS})
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/${DEPENDENCY_A})
		endforeach()
	endif()
endmacro()

macro(get_dependency_component_files DEPENDENCY BUILD_TYPE COMPONENT)
	if(WIN32)
		list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS ${${DEPENDENCY}_${${COMPONENT}_UPPER_CASE}_LIBRARY_${${BUILD_TYPE}_UPPER_CASE}})
		if(BUILD_TYPE STREQUAL Debug)
			foreach(LIB ${${DEPENDENCY}_${${COMPONENT}_UPPER_CASE}_LIBRARY_${${BUILD_TYPE}_UPPER_CASE}})
				find_dependency_dll_from_lib(${DEPENDENCY} ${BUILD_TYPE} ${LIB})
				find_dependency_pdb_from_lib(${DEPENDENCY} ${BUILD_TYPE} ${LIB})
			endforeach()
		endif()
	else()
		if(${DEPENDENCY}_LIBRARY_TYPE STREQUAL "dynamic")
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS ${${DEPENDENCY}_${${COMPONENT}_UPPER_CASE}_LIBRARY_${${BUILD_TYPE}_UPPER_CASE}})
		else()
			list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS ${${DEPENDENCY}_${${COMPONENT}_UPPER_CASE}_LIBRARY_${${BUILD_TYPE}_UPPER_CASE}})
		endif()
	endif()
endmacro()

macro(find_dependency_dll_from_lib DEPENDENCY BUILD_TYPE LIB)
	string(REPLACE ".lib" ".dll" DLL "${LIB}")
	if(EXISTS ${DLL})
		list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS "${DLL}")
	endif()
endmacro()

macro(find_dependency_pdb_from_lib DEPENDENCY BUILD_TYPE LIB)
	string(REPLACE ".lib" ".pdb" PDB "${LIB}")
	if(EXISTS ${PDB})
		list(APPEND ${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS "${PDB}")
	endif()
endmacro()

macro(guess_dependency_component_files DEPENDENCY BUILD_TYPE COMPONENT)
	guess_dependency_component_libs(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	guess_dependency_component_pdbs(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	guess_dependency_component_dlls(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	guess_dependency_component_sos(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
	guess_dependency_component_as(${DEPENDENCY} ${BUILD_TYPE} ${COMPONENT})
endmacro()

macro(guess_dependency_component_libs DEPENDENCY BUILD_TYPE COMPONENT)
	file(GLOB COMPONENT_LIBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR}/*${COMPONENT}*.lib)
	set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIBS} ${COMPONENT_LIBS})
endmacro()

macro(guess_dependency_component_pdbs DEPENDENCY BUILD_TYPE COMPONENT)
	file(GLOB COMPONENT_PDBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_LIB_DIR}/*${COMPONENT}*.pdb)
	set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_PDBS} ${COMPONENT_PDBS})
endmacro()

macro(guess_dependency_component_dlls DEPENDENCY BUILD_TYPE COMPONENT)
	file(GLOB COMPONENT_DLLS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${COMPONENT}*.dll)
	set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_DLLS} ${COMPONENT_DLLS})
endmacro()

macro(guess_dependency_component_sos DEPENDENCY BUILD_TYPE COMPONENT)
	file(GLOB COMPONENT_SOS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${COMPONENT}*.so)
	set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_SOS} ${COMPONENT_SOS})
endmacro()

macro(guess_dependency_component_as DEPENDENCY BUILD_TYPE COMPONENT)
	file(GLOB COMPONENT_AS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_BIN_DIR}/*${COMPONENT}*.a)
	set(${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS ${${DEPENDENCY}_${${BUILD_TYPE}_UPPER_CASE}_AS} ${COMPONENT_AS})
endmacro()

macro(install_dependency_headers TARGET_NAME DEPENDENCY)
	if(("${${DEPENDENCY}_COMPONENTS}" STREQUAL "") AND ("${${DEPENDENCY}_HEADERS}" STREQUAL ""))
		#message("test1")
		#message("${${DEPENDENCY}_INCLUDE_DIR}")
		install(DIRECTORY ${${DEPENDENCY}_INCLUDE_DIR} DESTINATION ${${PROJECT_NAME}_INSTALL_PREFIX})
	else()
		#message("test2")
		#message(${${DEPENDENCY}_INCLUDE_DIR})
		install_general_headers(${DEPENDENCY})
		install_component_headers(${DEPENDENCY})
		if(${${DEPENDENCY}_PREBUILT} STREQUAL "FALSE")
			install_named_headers(${DEPENDENCY})
		endif()
	endif()
endmacro()

macro(install_general_headers DEPENDENCY)
	message("install_general_headers")
	message("${DEPENDENCY}")
	file(
		GLOB
		GENERAL_HEADERS
		${${DEPENDENCY}_INCLUDE_DIR}/*.h
		${${DEPENDENCY}_INCLUDE_DIR}/*.hpp
	)
	install(
		FILES ${GENERAL_HEADERS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}
	)
	file(
		GLOB
		ITEMS
		${${DEPENDENCY}_INCLUDE_DIR}/*
	)
	foreach(ITEM ${ITEMS})
		if(IS_DIRECTORY ${ITEM})
			#message("IS_DIRECTORY")
			#message(${ITEM})
			install(DIRECTORY ${ITEM} DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${ITEM_NAME})
			#file(
			#	GLOB
			#	GENERAL_HEADERS
			#	${ITEM}/*.h
			#	${ITEM}/*.hpp
			#)
			#get_filename_component(ITEM_NAME ${ITEM} NAME)
			#install(
			#	FILES ${GENERAL_HEADERS}
			#	DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${ITEM_NAME}
			#)
		endif()
	endforeach()
endmacro()

macro(install_component_headers DEPENDENCY)
	foreach(COMPONENT ${${DEPENDENCY}_COMPONENTS})
		install_component_header_dir(${DEPENDENCY} ${COMPONENT})
		install_component_header_files(${DEPENDENCY} ${COMPONENT})		
	endforeach()
endmacro()

macro(install_component_header_dir DEPENDENCY COMPONENT)
	if(EXISTS ${${DEPENDENCY}_INCLUDE_DIR}/${COMPONENT})
		install(DIRECTORY ${${DEPENDENCY}_INCLUDE_DIR}/${COMPONENT} DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR})
	endif()
	
	file(
		GLOB
		ITEMS
		${${DEPENDENCY}_INCLUDE_DIR}/*
	)
	foreach(ITEM ${ITEMS})
		if(IS_DIRECTORY ${ITEM})
			if(EXISTS ${ITEM}/${COMPONENT})
				get_filename_component(ITEM_NAME ${ITEM} NAME)
				install(DIRECTORY ${ITEM}/${COMPONENT} DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${ITEM_NAME})
			endif()
		endif()
	endforeach()
endmacro()

macro(install_component_header_files DEPENDENCY COMPONENT)
	file(GLOB COMPONENT_HEADERS ${${DEPENDENCY}_INCLUDE_DIR}/*${COMPONENT}*.*)
	install(
		FILES ${COMPONENT_HEADERS}
		DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}
	)
endmacro()

macro(install_named_headers DEPENDENCY)
	foreach(HEADER ${${DEPENDENCY}_HEADERS})
		get_header_path(${DEPENDENCY} ${HEADER})
		install(FILES ${${DEPENDENCY}_INCLUDE_DIR}/${HEADER} DESTINATION ${${PROJECT_NAME}_INSTALL_INCLUDEDIR}/${HEADER_PATH})
	endforeach()
endmacro()

macro(set_environment_paths)
	if (MSVC)
		file(
			WRITE ${PROJECT_SOURCE_DIR}/build/${PROJECT_NAME}.vcxproj.user
			"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
			"<Project ToolsVersion=\"${COMPILER_VERSION}\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n"
			#"	<PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Debug|${${PROJECT_NAME}_ARCHITECTURE}'\">\n"
			"	<PropertyGroup>\n"
			"		<LocalDebuggerEnvironment>PATH=${${PROJECT_NAME}_ENVIRONMENT_PATHS};%PATH% $(LocalDebuggerEnvironment)</LocalDebuggerEnvironment>\n"
			"		<DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>\n"
			"	</PropertyGroup>\n"
			"</Project>"
		)
	endif()
endmacro()

macro(create_config_file)
	file(READ ${PROJECT_NAME}Config.cmake CONFIG_FILE)
	set_prebuilt_option(${CONFIG_FILE})
	file(WRITE ${${PROJECT_NAME}_INSTALL_PREFIX}/${PROJECT_NAME}Config.cmake "${PREBUILT_CONFIG_FILE}")
endmacro()

macro(set_prebuilt_option CONFIG_FILE)
	string(REGEX MATCH "configure\\([\\\n\\\t ]*${PROJECT_NAME}" REGEX_MATCH "${CONFIG_FILE}")
	string(REPLACE ${REGEX_MATCH} "${REGEX_MATCH}\n\tPREBUILT" PREBUILT_CONFIG_FILE "${CONFIG_FILE}")
endmacro()
	
macro(create_config_version_file)
	file(INSTALL ${PROJECT_NAME}ConfigVersion.cmake DESTINATION ${${PROJECT_NAME}_INSTALL_PREFIX})
endmacro()

macro(create_installer TARGET_NAME)	
	include(InstallRequiredSystemLibraries)
	set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/License.txt")
	set(CPACK_PACKAGE_VERSION_MAJOR "${${TARGET_NAME}_MAJOR_VERSION}")
	set(CPACK_PACKAGE_VERSION_MINOR "${${TARGET_NAME}_MINOR_VERSION}")
	include(CPack)
endmacro()

macro(set_version TARGET_NAME)
	set_project_name_and_dir(${TARGET_NAME} "${DIRECTORY}")

	set(options "")
	set(oneValueArgs MAJOR_VERSION MINOR_VERSION PATCH_VERSION)
	set(multiValueArgs "")
	cmake_parse_arguments(${TARGET_NAME} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
	set(${TARGET_NAME}_VERSION ${${TARGET_NAME}_MAJOR_VERSION}.${${TARGET_NAME}_MINOR_VERSION}.${${TARGET_NAME}_PATCH_VERSION})
	message(STATUS "${TARGET_NAME} version: ${${TARGET_NAME}_VERSION}")
	check_compatability(${${TARGET_NAME}_VERSION})
endmacro()

macro(set_project_name_and_dir TARGET_NAME DIRECTORY)
	if("${PROJECT_NAME}" STREQUAL "")
		project(${TARGET_NAME})
		set(${TARGET_NAME}_DIR ${PROJECT_SOURCE_DIR})
	elseif(NOT DEFINED ${TARGET_NAME}_DIR)
		set(${TARGET_NAME}_DIR ${TARGET_NAME_DIR})
	endif()
endmacro()

macro(check_compatability TARGET_VERSION)
	set(PACKAGE_VERSION ${TARGET_VERSION})
	if("${TARGET_VERSION}" VERSION_LESS "${PACKAGE_FIND_VERSION}")
	  set(PACKAGE_VERSION_COMPATIBLE FALSE)
	else()
	  set(PACKAGE_VERSION_COMPATIBLE TRUE)
	  if ("${TARGET_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
		set(PACKAGE_VERSION_EXACT TRUE)
	  endif()
	endif()
endmacro()

macro(configure TARGET_NAME)
	set(options MULTI_THREADED WINDOWS_EXPORT_ALL_SYMBOLS PREBUILT)
	set(oneValueArgs TYPE CHARACTER_SET)
	set(multiValueArgs ARCHITECTURES BUILD_TYPES LIBRARY_TYPES DEPENDENCIES RUNTIMES HEADERS SOURCES)
	cmake_parse_arguments(CONFIGURATION	"${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
	set_type(${TARGET_NAME} ${CONFIGURATION_TYPE})
	set_architecture(${TARGET_NAME} ${CONFIGURATION_ARCHITECTURES})
	set_build_type(${TARGET_NAME} "${CONFIGURATION_BUILD_TYPES}")
	set_library_type(${TARGET_NAME} ${CONFIGURATION_LIBRARY_TYPES})
	set_runtime(${TARGET_NAME} ${CONFIGURATION_RUNTIMES})
	set_character_set(${TARGET_NAME} ${CONFIGURATION_CHARACTER_SET})
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ${CONFIGURATION_WINDOWS_EXPORT_ALL_SYMBOLS})
	set_threads(${TARGET_NAME} ${CONFIGURATION_MULTI_THREADED})
	set_directories(${TARGET_NAME})
	set(${TARGET_NAME}_PREBUILT ${CONFIGURATION_PREBUILT})
	set(${TARGET_NAME}_HEADERS "${CONFIGURATION_HEADERS}")
	set(${TARGET_NAME}_SOURCES "${CONFIGURATION_SOURCES}")
	set_install_prefixes(${TARGET_NAME})
	if(${CONFIGURATION_PREBUILT} STREQUAL "FALSE")
		create_project(${TARGET_NAME} "${${TARGET_NAME}_DIR}")
	endif()
	if((${CONFIGURATION_PREBUILT} STREQUAL "FALSE") OR GENERAL_RECURSIVELY_ADD_PREBUILT_DEPENDENCIES)
		set_dependencies(${TARGET_NAME} "${CONFIGURATION_DEPENDENCIES}")
	endif()
	if(EXISTS "${${TARGET_NAME}_DIR}")
		set(${TARGET_NAME}_INCLUDE_DIRS ${${TARGET_NAME}_INCLUDE_DIRS} ${${TARGET_NAME}_DIR} ${${TARGET_NAME}_DIR}/src ${${TARGET_NAME}_DIR}/include)
	endif()
	#message("CONFIGURE: ${TARGET_NAME}")
	#message("CONFIGURE_INCLUDE_DIR: ${${TARGET_NAME}_INCLUDE_DIRS}")
endmacro()

macro(set_type TARGET_NAME TYPE)
	set(${TARGET_NAME}_TYPE ${TYPE})
	message(STATUS "${TARGET_NAME} type: ${TYPE}")
endmacro()

macro(set_build_type TARGET_NAME TYPES)
	if(DEFINED CMAKE_BUILD_TYPE)
		set(${TARGET_NAME}_BUILD_TYPES ${CMAKE_BUILD_TYPE})
	else()
		set(${TARGET_NAME}_BUILD_TYPES ${TYPES})
	endif()
	message(STATUS "${TARGET_NAME} build types: ${${TARGET_NAME}_BUILD_TYPES}")
endmacro()

macro(set_library_type TARGET_NAME)
	if(${TARGET_NAME}_TYPE STREQUAL Library)
		if(DEFINED ${TARGET_NAME}_LIBRARY_TYPES)
			set(LIBRARY_TYPES ${${TARGET_NAME}_LIBRARY_TYPES})
		elseif(${ARGC} LESS 2)
			set(LIBRARY_TYPES dynamic static)
		else()
			set(LIBRARY_TYPES ${ARGN})
		endif() 
		list(GET LIBRARY_TYPES 0 DEFAULT_LIBRARY_TYPE)
		set(${TARGET_NAME}_LIBRARY_TYPE ${DEFAULT_LIBRARY_TYPE} CACHE STRING "Link type")
		set_property(CACHE ${TARGET_NAME}_LIBRARY_TYPE PROPERTY STRINGS "${LIBRARY_TYPES}")
		message(STATUS "${TARGET_NAME} library type: ${${TARGET_NAME}_LIBRARY_TYPE}")
	endif()
endmacro()

macro(set_runtime TARGET_NAME)
	if(DEFINED ${TARGET_NAME}_RUNTIMES)
		set(RUNTIMES ${${TARGET_NAME}_RUNTIMES})
	elseif(${ARGC} LESS 2)
		set(RUNTIMES dynamic static)
	else()
		set(RUNTIMES ${ARGN})
	endif() 
	list(GET RUNTIMES 0 DEFAULT_RUNTIME)
	set(${TARGET_NAME}_RUNTIME ${DEFAULT_RUNTIME} CACHE STRING "Runtime")
	set_property(CACHE ${TARGET_NAME}_RUNTIME PROPERTY STRINGS "${RUNTIMES}")
	message(STATUS "${TARGET_NAME} runtime: ${${TARGET_NAME}_RUNTIME}")
endmacro()

macro(set_character_set TARGET_NAME)
	if(${ARGC} LESS 2)
		set(${TARGET_NAME}_CHARACTER_SET Multi-Byte)
	else()
		set(${TARGET_NAME}_CHARACTER_SET ${ARGV1})
	endif()
endmacro()

macro(set_threads TARGET_NAME MULTI_THREADED)
	if(${MULTI_THREADED} STREQUAL "TRUE")
		set(${TARGET_NAME}_THREADS "multi-threaded")
	else()
		set(${TARGET_NAME}_THREADS "single-threaded")
	endif()
	message(STATUS "${TARGET_NAME} multi-threaded: ${MULTI_THREADED}")
endmacro()

macro(set_directories TARGET_NAME)
	if(${TARGET_NAME} STREQUAL ${PROJECT_NAME})
		include(GNUInstallDirs)
		set_include_directories()
		set_build_directories()
		set_install_directory()
	endif()
endmacro()

macro(set_include_directories)
	include_directories(
		${PROJECT_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/include
	)
endmacro()

macro(set_build_directories)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endmacro()

macro(set_install_directory)
	if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
		set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "default install path" FORCE )
	endif()
endmacro()

macro(set_install_prefixes TARGET_NAME)
	string(TOLOWER ${TARGET_NAME} ${TARGET_NAME}_LOWER_CASE)
#	if(${TARGET_NAME}_TYPE STREQUAL Application)
#		set(${TARGET_NAME}_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${${TARGET_NAME}_LOWER_CASE}-${${TARGET_NAME}_VERSION}-${COMPILER}-${${PROJECT_NAME}_ARCHITECTURE}-${${TARGET_NAME}_LIBRARY_TYPE}-${${TARGET_NAME}_RUNTIME}_runtime)
#	elseif(${TARGET_NAME}_TYPE STREQUAL Library)
#		set(${TARGET_NAME}_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${${TARGET_NAME}_LOWER_CASE}-${${TARGET_NAME}_VERSION}-${COMPILER}-${${PROJECT_NAME}_ARCHITECTURE}-${${TARGET_NAME}_LIBRARY_TYPE}-${${TARGET_NAME}_RUNTIME}_runtime)
#	else()
#		set(${TARGET_NAME}_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${${TARGET_NAME}_LOWER_CASE}-${${TARGET_NAME}_VERSION}-${COMPILER}-${${PROJECT_NAME}_ARCHITECTURE})
#	endif()
	set(${TARGET_NAME}_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${${TARGET_NAME}_LOWER_CASE})
	set(${TARGET_NAME}_INSTALL_INCLUDEDIR ${${TARGET_NAME}_INSTALL_PREFIX}/include)
	set(${TARGET_NAME}_INSTALL_LIBDIR ${${TARGET_NAME}_INSTALL_PREFIX}/lib)
	set(${TARGET_NAME}_INSTALL_BINDIR ${${TARGET_NAME}_INSTALL_PREFIX}/bin)
endmacro()

macro(set_dependencies TARGET_NAME DEPENDENCY_CONFIGURATIONS)
	foreach(DEPENDENCY_CONFIGURATION ${DEPENDENCY_CONFIGURATIONS})
		if(NOT STRING STREQUAL "")
			get_arguments(${DEPENDENCY_CONFIGURATION})	
			get_name(${ARGUMENTS})		
			if(NOT NAME STREQUAL ${PROJECT_NAME})
				unset_argument_variables(${NAME})
				if(NOT ARGUMENTS STREQUAL "")
					get_version(${NAME} ${ARGUMENTS})
					foreach(ITEM ${ARGUMENTS})
						list(FIND DEPENDENCY_ARGUMENTS ${ITEM} INDEX)
						if((DEFINED ARGUMENT) AND (INDEX EQUAL -1))
							set(${NAME}_${ARGUMENT} ${${NAME}_${ARGUMENT}} ${ITEM})
						else()
							set(ARGUMENT ${ITEM})
							set(${NAME}_${ARGUMENT} "")
						endif()
					endforeach()
				endif()
			endif()
			add_dependency(${TARGET_NAME} ${NAME})
		endif()
	endforeach()
endmacro()

macro(get_arguments STRING)
	string(REPLACE "\n" " " ARGUMENTS "${STRING}")
	string(REPLACE "\t" " " ARGUMENTS "${ARGUMENTS}")
	string(REGEX REPLACE "[ ]+" " " ARGUMENTS "${ARGUMENTS}")
	separate_arguments(ARGUMENTS)
endmacro()

macro(get_name ARGUMENTS)
	list(GET ARGUMENTS 0 NAME)
	list(REMOVE_AT ARGUMENTS 0)
endmacro()

macro(unset_argument_variables NAME)
	unset(${NAME}_COMPONENTS)
	unset(${NAME}_HEADERS)
	unset(${NAME}_DEBUG_DYNAMIC_LIBS)
	unset(${NAME}_DEBUG_DYNAMIC_DLLS)
	unset(${NAME}_DEBUG_DYNAMIC_PDBS)
	unset(${NAME}_DEBUG_STATIC_LIBS)
	unset(${NAME}_DEBUG_STATIC_PDBS)
	unset(${NAME}_RELEASE_DYNAMIC_LIBS)
	unset(${NAME}_RELEASE_DYNAMIC_DLLS)
	unset(${NAME}_RELEASE_STATIC_LIBS)
endmacro()

macro(get_version NAME ARGUMENTS)
	list(GET ARGUMENTS 0 VERSION)
	if (VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
		set(${NAME}_VERSION ${VERSION})
		list(REMOVE_AT ARGUMENTS 0)
	endif()
endmacro()

macro(SUBDIRLIST result curdir)
	#message("SUBDIRLIST")
	
	#message("${curdir}")
	
	FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
	SET(dirlist "")
	
	#message("${children}")
	
	FOREACH(child ${children})
		IF(IS_DIRECTORY ${curdir}/${child})
			LIST(APPEND dirlist ${child})
		ENDIF()
	ENDFOREACH()
	SET(${result} ${dirlist})
endmacro()

main()