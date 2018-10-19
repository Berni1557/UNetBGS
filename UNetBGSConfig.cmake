#Add definitions and set flags here, e.g.

configure(
	UNetBGS
	TYPE Library
	ARCHITECTURES x64
	LIBRARY_TYPES dynamic static
	RUNTIMES dynamic static
	BUILD_TYPES Debug Release
	MULTI_THREADED
	WINDOWS_EXPORT_ALL_SYMBOLS
	HEADERS src/*.py
	SOURCES src/*.py
	DEPENDENCIES
)

#Add definitions and set flags here, e.g.
