#============================================================================
# sources

SRCS = transforms_CAPI.c transforms_CFUNC.c
OBJS = $(SRCS:.c=.o)

#============================================================================
# library name

LIBBASENAME = _transforms_CAPI

LIBNAME = $(LIBBASENAME).so

#============================================================================
# compiler options

CC       = gcc
CFLAGS   = -O3 -fPIC
CDEFINES = 

#============================================================================
# header location

PYTHON_INCLUDE_DIR = ${HOME}/opt/include/python2.7
NUMPY_INCLUDE_DIR  = ${HOME}/opt/lib/python2.7/site-packages/numpy/core/include/numpy

INCPATH = -I$(PYTHON_INCLUDE_DIR) -I$(NUMPY_INCLUDE_DIR)

#============================================================================
# targets

default:	lib

lib:	$(LIBNAME)

$(LIBBASENAME).so: $(OBJS)
	$(CC) -shared $(CFLAGS) -flat_namespace -o $(LIBBASENAME).so $(OBJS) -lm

clean:
	$(RM) -rf *.o $(LIBBASENAME).so

#============================================================================
# suffix rules

.SUFFIXES: .c
.c.o:
	$(CC) $(CFLAGS) $(CDEFINES) $(INCPATH) -c $<
