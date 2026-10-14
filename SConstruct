#!/usr/bin/env python
import os
import sys

env = SConscript("../godot-cpp/SConstruct")

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags


lib_path = {
    'windows': [
        'opencv/build/x64/vc16/lib',
        'tesseract/lib'
    ],
    'macos': [
    
    ],
    'linux': [
    
    ]
}

lib_files = {
    'windows': [
        'opencv_videoio4150',
        'opencv_dnn4150',
        'opencv_cudaarithm4150',
        'opencv_cudaimgproc4150',
        'opencv_core4150',
        'opencv_imgproc4150',
        'opencv_imgcodecs4150',
        'opencv_objdetect4150',
        'opencv_video4150',
        'opencv_tracking4150',
        'opencv_text4150',
        'opencv_highgui4150',
        'google.tesseract.libtesseract-main',
    ],
    'macos': [
        'libopencv_core.dylib',
        'libopencv_imgcodecs.dylib',
        'libopencv_imgproc.dylib',
        'libopencv_videoio.dylib',
        'libopencv_objdetect.dylib',
        'libopencv_video.dylib',
        'libopencv_tracking.dylib'
    ],
    'linux': [
        'libopencv_core.so',
        'libopencv_imgcodecs.so',
        'libopencv_imgproc.so',
        'libopencv_videoio.so',
        'libopencv_objdetect.so',
        'libopencv_video.so',
        'libopencv_tracking.so'
    ]
}

env.Append(CPPPATH=["opencv/build/include", "tesseract/include"])
env.Append(LIBPATH=lib_path[env["platform"]])
env.Append(LIBS=lib_files[env["platform"]])

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

lib_filename = "{}LPRCamera{}{}".format(env.subst('$SHLIBPREFIX'), env["suffix"], env.subst('$SHLIBSUFFIX'))

# Creates a SCons target for the path with our sources.
library = env.SharedLibrary(
    "bin/{}".format(lib_filename),
    source=sources,
)

# Selects the shared library as the default target.
Default(library)

# scons platform=windows disable_exceptions=false target=template_release
# scons platform=windows disable_exceptions=false target=template_debug