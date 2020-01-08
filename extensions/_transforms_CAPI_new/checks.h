
/* returns non-zero if the object is a numpy array that with dims dimensions
   dtype is double and is in C order.
*/
static inline int
is_valid_array(PyObject* op, int dims)
{
    PyArrayObject *aop = (PyArrayObject *)op;
    return PyArray_Check(op) &&
        dims == PyArray_NDIM(aop) &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}


/* returns non-zero if the object is a numpy array that with shape (elements,),
   dtype is double and is in C order.
*/
static inline int
is_valid_vector(PyObject* op, npy_intp elements)
{
    PyArrayObject *aop = (PyArrayObject *)op;
    return PyArray_Check(op) &&
        1 == PyArray_NDIM(aop) &&
        elements == PyArray_SHAPE(aop)[0] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}

/* Returns non-zero if the object is a numpy array with shape (X, elements),
   dtype is double and is in C order. X can be any positive integer.
*/
static inline int
is_valid_vector_array(PyObject *op, npy_intp elements)
{
    PyArrayObject *aop = (PyArrayObject *)op;    
    return PyArray_Check(op) &&
        2 == PyArray_NDIM(aop) &&
        0 < PyArray_SHAPE(aop)[0] &&
        elements == PyArray_SHAPE(aop)[1] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}


/* returns non-zero if the object is a numpy array with shape (outerd, innerd),
   dtype is double and is in C order.
*/
static inline int
is_valid_matrix(PyObject *op, npy_intp outerd, npy_intp innerd)
{
    PyArrayObject *aop = (PyArrayObject *)op;    
    return PyArray_Check(op) &&
        2 == PyArray_NDIM(aop) &&
        outerd == PyArray_SHAPE(aop)[0] &&
        innerd == PyArray_SHAPE(aop)[1] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}

/* returns non-zero if the object is a numpy array with shape (X, outerd,
   innerd), dtype is double and is in C order. X can be any positive integer.
*/
static inline int
is_valid_matrix_array(PyObject *op, npy_intp outerd, npy_intp innerd)
{
    PyArrayObject *aop = (PyArrayObject *)op;    
    return PyArray_Check(op) &&
        3 == PyArray_NDIM(aop) &&
        0 < PyArray_SHAPE(aop)[0] &&
        outerd == PyArray_SHAPE(aop)[1] &&
        innerd == PyArray_SHAPE(aop)[2] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}

static inline void
raise_value_error(const char *why)
{
    PyErr_SetString(PyExc_ValueError, why);
}

static inline void
raise_runtime_error(const char *why)
{
    PyErr_SetString(PyExc_RuntimeError, why);
}

/*
 * These are used so that actual checks can be done by the ParseTuple itself.
 *
 * Using some structs for the result allow piggybacking some extra information
 * to the converter and back (like the argument name to improve error reporting).
 */

typedef struct {
    const char *name;
    double *data;
    size_t count;
} named_array_1d;

typedef struct {
    const char *name;
    double *data;
    size_t inner_count;
    size_t outer_count;
} named_array_2d;

typedef struct {
    const char *name;
    double *data;
} named_vector3;

typedef struct {
    const char *name;
    double *data;
    size_t count; /* number of elements in array */
} named_vector3_array;

typedef struct {
    const char *name;
    double *data;
} named_matrix33;

typedef struct {
    const char *name;
    double *data;
    size_t count;
} named_matrix33_array;


static int
array_1d_converter(PyObject *op, void *result)
{
    named_array_1d *res = (named_array_1d *) result;

    if (is_valid_array(op, 1))
    {
        res->data = PyArray_DATA(op);
        res->count = (size_t)PyArray_SHAPE(op)[0];
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a 1d float64 C-ordered array",
                     res->name);
        return 0;
    }
}


static int
array_2d_converter(PyObject *op, void *result)
{
    named_array_2d *res = (named_array_2d *) result;

    if (is_valid_array(op, 2))
    {
        res->data = PyArray_DATA(op);
        res->inner_count = (size_t)PyArray_SHAPE(op)[1];
        res->outer_count = (size_t)PyArray_SHAPE(op)[0];
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a 2d float64 C-ordered array",
                     res->name);
        return 0;
    }
}


static int
vector3_converter(PyObject *op, void *result)
{
    named_vector3 *res = (named_vector3 *) result;

    if (is_valid_vector(op, 3))
    {
        res->data = PyArray_DATA(op);
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a (3,) float64 C-ordered array",
                     res->name);
        return 0;
    }
}

static int
vector3_array_converter(PyObject *op, void *result)
{
    named_vector3_array *res = (named_vector3_array *) result;

    if (is_valid_vector_array(op, 3))
    {
        res->data = PyArray_DATA(op);
        res->count = (size_t)PyArray_SHAPE(op)[0];
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a (n, 3) float64 C-ordered array",
                     res->name);
        return 0;
    }
}

static int
matrix33_converter(PyObject *op, void *result)
{
    named_matrix33 *res = (named_matrix33 *) result;

    if (is_valid_matrix(op, 3, 3))
    {
        res->data = PyArray_DATA(op);
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a (3, 3) float64 C-ordered array",
                     res->name);
        return 0;
    }
}

static int
matrix33_array_converter(PyObject *op, void *result)
{
    named_matrix33_array *res = (named_matrix33_array *) result;

    if (is_valid_matrix(op, 3, 3))
    {
        res->data = PyArray_DATA(op);
        res->count = (size_t)PyArray_SHAPE(op)[0];
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a (n, 3, 3) float64 C-ordered array",
                     res->name);
        return 0;
    }
}
