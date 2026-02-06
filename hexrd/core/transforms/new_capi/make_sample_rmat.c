
static void
make_sample_rmat(double chi, double ome, double *result_rmat)
{
    double cchi, schi, come, some;
    double * restrict rmat = result_rmat;

    cchi = cos(chi);
    schi = sin(chi);
    come = cos(ome);
    some = sin(ome);

    rmat[0] =  come;
    rmat[1] =  0.0;
    rmat[2] =  some;
    rmat[3] =  schi*some;
    rmat[4] =  cchi;
    rmat[5] = -schi*come;
    rmat[6] = -cchi*some;
    rmat[7] =  schi;
    rmat[8] =  cchi*come;

}

static void
make_sample_rmat_array(double chi, const double *ome_ptr, size_t ome_count, double *result_ptr)
{
    double cchi,schi;
    double * restrict rmat = result_ptr;

    cchi = cos(chi);
    schi = sin(chi);

    for (size_t i = 0; i < ome_count; i++)
    {
        double ome = ome_ptr[i];
        double come = cos(ome);
        double some = sin(ome);

        rmat[0] =  come;
        rmat[1] =  0.0;
        rmat[2] =  some;
        rmat[3] =  schi*some;
        rmat[4] =  cchi;
        rmat[5] = -schi*come;
        rmat[6] = -cchi*some;
        rmat[7] =  schi;
        rmat[8] =  cchi*come;

        rmat += 9;
    }
}

static const char *docstring_makeOscillRotMat =
    "c module implementation of make_sample_rotmat.\n"
    "Please use the Python wrapper.\n";


static PyObject *
python_makeOscillRotMat(PyObject * self, PyObject * args)
{
    nah_array ome = { NULL, "ome", NAH_TYPE_DP_FP, { NAH_DIM_ANY }};
    double chi;
    npy_intp dims[3] = { 0, 3, 3 };
    PyArrayObject *result = NULL;
    
    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"dO&",
                          &chi,
                          nah_array_converter, &ome))
        return NULL;

    dims[0] = PyArray_DIM(ome.pyarray, 0);
    result = (PyArrayObject*)PyArray_EMPTY(3, dims, NPY_DOUBLE, 0);
    if (NULL == result)
        goto fail_alloc;
    
    /* Call the actual function */
    make_sample_rmat_array(chi,
                           (double *)PyArray_DATA(ome.pyarray),
                           PyArray_DIM(ome.pyarray, 0),
                           (double *)PyArray_DATA(result));

    return (PyObject*)result;

 fail_alloc:
    Py_XDECREF(result);
    
    return PyErr_NoMemory();
}
