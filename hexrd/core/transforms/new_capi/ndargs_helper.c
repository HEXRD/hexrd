#include "ndargs_helper.h"

static int nah_array_converter(PyObject *op, void *result)
{
    nah_array *na = (nah_array*)result;

    if (op == Py_None && (na->type & NAH_TYPE_NONE))
    {
        /* arg is None and None is allowed */
        na->type = NAH_TYPE_NONE;
        na->pyarray = NULL;
        return 1;
    }

    if (!PyArray_Check(op))
        goto fail;

    {
        PyArrayObject *array = (PyArrayObject *)op;
        npy_intp *array_shape = PyArray_SHAPE(array);
        npy_intp *na_shape = &na->dims[0];
        int array_ndim = PyArray_NDIM(array);
        int na_dim, array_dim;
        if (!PyArray_ISBEHAVED_RO(array))
            goto fail;

        switch (PyArray_TYPE(array))
        {
        case NPY_FLOAT32:
            na->type &= NAH_TYPE_SP_FP;
            if (!na->type)
                goto fail;
            break;
        case NPY_FLOAT64:
            na->type &= NAH_TYPE_DP_FP;
            if (!na->type)
                goto fail;
            break;
        default:
            goto fail;
        }
        /* note that in this spec, shape is specified with the inner-most dimensions first.
         * makes it easier to use C initializers for the dimensions and get the non-specified
         * ones be 0. However, this also means that checking has to be made with different
         * directions.
         */
        if (array_ndim > NAH_DIMS_MAX)
            goto fail;

        for (na_dim = 0, array_dim = array_ndim - 1;
             array_dim >= 0; /* must be signed! */
             na_dim++, array_dim--)
        {
            if (array_shape[array_dim] == na_shape[na_dim])
                continue; /* perfect match */

            /* 
               any explicit dimension that does not match or no dimension allowed
             */
            if (na_shape[na_dim] >= 0)
                goto fail;

            if (NAH_DIM_ANY == na_shape[na_dim] ||
                NAH_DIM_OPT == na_shape[na_dim])
                /* a dimension with size inherited by the argument, set the actual dimension */
                na_shape[na_dim] = array_shape[array_dim];
            else
                goto internal_error;
        }

        /*
         * at this point, only optional dimensions should remain in the shape
         * spec, and they are not present.
         */
        while (na_dim <  NAH_DIMS_MAX)
        {
            if (na_shape[na_dim] == NAH_DIM_NONE)
                break; /* done */

            if (na_shape[na_dim] == NAH_DIM_OPT)
            {
                na_shape[na_dim] = 0;
                continue;
            }
            else
                goto fail; /* there are still non-optional arguments: fail */
            na_dim++;
        }
        /* if we got here, the array seems ok as argument */
        na->pyarray = array;
    }
    
    return 1;
    
 fail:
    PyErr_Format(PyExc_ValueError, "'%s' argument does not match expected dimensions/types, is aligned and is writeable if an output argument.",
                 na->name);
    return 0;
    
 internal_error:
    /* This means an error in the code, not in the Value. This will happen when
       an special value in the dimension (a negative int) is used that is not
       understood.
    */
    PyErr_Format(PyExc_RuntimeError, "Internal error converting arg '%s' (%s::%d)",
                 na->name, __FILE__, __LINE__);
    return 0;
}

