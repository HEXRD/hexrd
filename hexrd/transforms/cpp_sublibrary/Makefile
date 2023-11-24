CXX = c++
CXXFLAGS = -O3 -Wall -shared -fPIC -funroll-loops -Wno-maybe-uninitialized -fopenmp
INCLUDES = -I/${CONDA_PREFIX}/include/eigen3/ -I/${CONDA_PREFIX}/include/xsimd -I/${CONDA_PREFIX}/include -I/${CONDA_PREFIX}/include/python3.11/
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