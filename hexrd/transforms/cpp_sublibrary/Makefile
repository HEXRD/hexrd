CXX = c++
CXXFLAGS = -O3 -Wall -shared -fPIC -funroll-loops -Wno-maybe-uninitialized
INCLUDES = -I/usr/include/eigen3/ -I/usr/include/xsimd -I/usr/include/pybind11 -I/usr/include/python3.10/
SRCS_TRANSFORMS = src/transforms.cpp
SRCS_INVERSE_DISTORTION = src/inverse_distortion.cpp
TARGET_DIR = ../../extensions/
TARGET_NAME_TRANSFORMS = transforms
TARGET_NAME_INVERSE_DISTORTION = inverse_distortion
EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)

all: transforms inverse_distortion

transforms: $(TARGET_DIR)$(TARGET_NAME_TRANSFORMS)$(EXTENSION_SUFFIX)

inverse_distortion: $(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX)

$(TARGET_DIR)$(TARGET_NAME_TRANSFORMS)$(EXTENSION_SUFFIX): $(SRCS_TRANSFORMS)
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $< -o $@ $(INCLUDES)

$(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX): $(SRCS_INVERSE_DISTORTION)
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $< -o $@ $(INCLUDES)

.PHONY: clean transforms inverse_distortion
clean:
	rm -f $(TARGET_DIR)$(TARGET_NAME_TRANSFORMS)$(EXTENSION_SUFFIX) \
				$(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX)
