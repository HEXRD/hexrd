CXX = c++
CXXFLAGS = -O3 -Wall -shared -fPIC -flto -funroll-loops -Wno-maybe-uninitialized -Wno-attributes -ffast-math
LDFLAGS = -L${CONDA_PREFIX}/lib
INCLUDES = -I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/include/python3.11/ -I${CONDA_PREFIX}/include/eigen3 -I${CONDA_PREFIX}/envs/hexrd/include/ -I${CONDA_PREFIX}/lib/python3.11/site-packages/numpy/core/include
SRCS_INVERSE_DISTORTION = src/inverse_distortion.cpp
SRCS_TRANSFORMS = src/transforms.cpp
TARGET_DIR = ../../extensions/
TARGET_NAME_INVERSE_DISTORTION = inverse_distortion
TARGET_NAME_TRANSFORMS = transforms
EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)

all: inverse_distortion transforms

inverse_distortion: $(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX)

transforms: $(TARGET_DIR)$(TARGET_NAME_TRANSFORMS)$(EXTENSION_SUFFIX)

$(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX): $(SRCS_INVERSE_DISTORTION)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(TARGET_DIR)$(TARGET_NAME_TRANSFORMS)$(EXTENSION_SUFFIX): $(SRCS_TRANSFORMS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

.PHONY: clean inverse_distortion transforms
clean:
	rm -f $(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX)
	rm -f $(TARGET_DIR)$(TARGET_NAME_TRANSFORMS)$(EXTENSION_SUFFIX)
