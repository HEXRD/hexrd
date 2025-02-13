CXX = c++
CXXFLAGS = -O3 -Wall -shared -fPIC -funroll-loops -Wno-maybe-uninitialized -Wno-attributes -fopenmp
INCLUDES = -I/${CONDA_PREFIX}/include -I/${CONDA_PREFIX}/include/python3.11/ -I/${CONDA_PREFIX}/lib/python3.11/site-packages/numpy/core/include
SRCS_INVERSE_DISTORTION = src/inverse_distortion.cpp
TARGET_DIR = ../../extensions/
TARGET_NAME_INVERSE_DISTORTION = inverse_distortion
EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)

all: inverse_distortion

inverse_distortion: $(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX)

$(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX): $(SRCS_INVERSE_DISTORTION)
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $< -o $@ $(INCLUDES)

.PHONY: clean inverse_distortion
clean:
	rm -f $(TARGET_DIR)$(TARGET_NAME_INVERSE_DISTORTION)$(EXTENSION_SUFFIX)