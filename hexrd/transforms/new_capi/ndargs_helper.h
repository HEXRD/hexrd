#ifndef NDARRAY_ARGS_HELPER_H
#define NDARRAY_ARGS_HELPER_H

#define NAH_DIMS_MAX 13
#define NAH_DIM_ANY -1
#define NAH_DIM_OPT -2
#define NAH_DIM_NONE 0
/*
 * nah_array holds a description of the array to be
 * converted, as well as holding some results from the
 * inspection performed by this conversion.
 */
enum
{
    NAH_TYPE_NONE  = 1<<0, // array may be None
    NAH_TYPE_SP_FP = 1<<1, // array of single precision fp
    NAH_TYPE_DP_FP = 1<<2, // array of double precision fp
};

typedef struct nah_array_struct
{
    PyArrayObject *pyarray;
    const char *name; /* arg name for error reporting */
    int type; /* at entry, contains allowable types, at exit the actual type */
    npy_intp dims[NAH_DIMS_MAX]; /* from inner to outer. */
} nah_array;

/* converter function for PyArg_ParseTupleAndKeywords.
 * op is the PyObject pointer to convert.
 * result points to a nah_array struct describing how the argument
 * should look like.
 *
 * returns 1 on success, 0 otherwise. When returning 0, a python
 * exception has been setup.
 */
XRD_CFUNCTION int nah_array_converter(PyObject *op, void *result);

#endif /* NDARRAY_ARGS_HELPER_H */
