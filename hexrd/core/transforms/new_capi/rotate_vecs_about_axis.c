static void
rotate_vecs_about_axis(size_t na, double *angles,
                       size_t nax, double *axes,
                       size_t nv, double *vecs,
                       double *rVecs)
{
    size_t i, j, sa, sax;
    double c, s, nrm, proj, aCrossV[3];

    if ( na == 1 ) sa = 0;
    else sa = 1;
    if ( nax == 1 ) sax = 0;
    else sax = 3;

    for (i=0; i<nv; i++)
    {
        /* Rotate using the Rodrigues' Rotation Formula */
        c = cos(angles[sa*i]);
        s = sin(angles[sa*i]);

        /* Compute norm of axis */
        if ( nax > 1 || i == 0 ) {
            nrm = 0.0;
            for (j=0; j<3; j++)
                nrm += axes[sax*i+j]*axes[sax*i+j];
            nrm = sqrt(nrm);
        }

        /* Compute projection of vec along axis */
        proj = 0.0;
        for (j=0; j<3; j++)
        {
            proj += axes[sax*i+j]*vecs[3*i+j];
        }

        /* Compute the cross product of the axis with vec */
        for (j=0; j<3; j++)
        {  
            aCrossV[j] = axes[sax*i+(j+1)%3]*vecs[3*i+(j+2)%3]-axes[sax*i+(j+2)%3]*vecs[3*i+(j+1)%3];
        }
        
        /* Combine the three terms to compute the rotated vector */
        for (j=0; j<3; j++)
        {
            rVecs[3*i+j] = c*vecs[3*i+j]+(s/nrm)*aCrossV[j]+(1.0-c)*proj*axes[sax*i+j]/(nrm*nrm);
        }
    }
}

static const char *docstring_rotate_vecs_about_axis =
    "c module implementation of rotate_vecs_about_axis.\n"
    "Please use the Python wrapper.\n";

static PyObject *
python_rotate_vecs_about_axis(PyObject *self, PyObject *args)
{
    /* API interface in Python is:
       angles, axis, vec

       return a vector with the rotated vectors
    */
    nah_array angles = { NULL, "angles", NAH_TYPE_DP_FP, { NAH_DIM_OPT }};
    nah_array axis = { NULL, "axis", NAH_TYPE_DP_FP, { 3, NAH_DIM_OPT }};
    nah_array vecs = { NULL, "vecs", NAH_TYPE_DP_FP, { 3, NAH_DIM_ANY }};
    PyArrayObject *result = NULL;
    size_t nangs, naxis, nvecs;
    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"O&O&O&",
                          nah_array_converter, &angles,
                          nah_array_converter, &axis,
                          nah_array_converter, &vecs))
        return NULL;

    /* check that (n) in angles and axis is the same as in vecs, if present */
    nangs = angles.dims[0];
    naxis = axis.dims[1];
    nvecs = vecs.dims[1];
    if ((nangs && nangs != nvecs) ||
        (naxis && naxis != nvecs))
        goto fail_dimensions;

    nangs = nangs?nangs:1;
    naxis = naxis?naxis:1;
    /*
      Allocate array for the result vectors. Its shape should be the same as
      vecs shape.
    */
    result = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(vecs.pyarray),
                                           PyArray_SHAPE(vecs.pyarray),
                                           NPY_DOUBLE, 0); 
    if (NULL == result)
        goto fail_alloc;

    /* Call the actual function */
    
    rotate_vecs_about_axis(nangs, (double *)PyArray_DATA(angles.pyarray),
                           naxis, (double *)PyArray_DATA(axis.pyarray),
                           nvecs, (double *)PyArray_DATA(vecs.pyarray),
                           (double *)PyArray_DATA(result));

    
    return (PyObject*)result;

 fail_dimensions:
    PyErr_Format(PyExc_ValueError,
                 "'%s', '%s', '%s' have mismatching dimensions",
                 angles.name, axis.name, vecs.name);
    goto fail;

 fail_alloc:
    PyErr_NoMemory();
    goto fail;

 fail:
    /* 
       Failure clean up. At this point the proper error should be already raised
       so just release any allocated resource and return 0 so that the exception
       is handled
    */
    Py_XDECREF(result);

    return 0;
}
