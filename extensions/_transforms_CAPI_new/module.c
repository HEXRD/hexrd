/*
 * %BEGIN LICENSE HEADER
 * %END LICENSE HEADER
 */


/* =============================================================================
 * THIS_MODULE_NAME is used in several places in macros to generate the module
 * entry point for Python 2 and Python 3, as well as for the internal module
 * name exported to Python.
 *
 * Having this as a define makes it easy to change as needed (only a single
 * place to modify
 *
 * Note the supporting CONCAT and STR macros...
 * =============================================================================
 */
#define THIS_MODULE_NAME transforms_CAPI_new

#define _CONCAT(a,b) a ## b
#define CONCAT(a,b) _CONCAT(a,b)
#define _STR(a) # a
#define STR(a) _STR(a)

/* ************************************************************************** */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* =============================================================================
 * Note: As we really want to only the entry point of the module to be visible
 * in the exported library, but there is a clear convenience on having different
 * functions in their own source file, the approach will be to include use this
 * file as the compile unit and import the source code of all the functions.
 *
 * This file will also contain all the module scaffolding needed (like the init
 * function as well as the exported method list declaration.
 * =============================================================================
 */



/* =============================================================================
 * This macros configure the way the module is built.
 * All implementation files included in a single compile unit.
 * Python wrappers are to be included.
 * Visibility for all functions are wrappers is removed by turning them into
 * statics.
 * =============================================================================
 */
#define XRD_SINGLE_COMPILE_UNIT 1
#define XRD_INCLUDE_PYTHON_WRAPPERS 1
#define XRD_CFUNCTION static
#define XRD_PYTHON_WRAPPER static

#include "transforms_types.h"
#include "transforms_utils.h"
#include "transforms_prototypes.h"
#include "checks.h"

#include "angles_to_gvec.c"
#include "angles_to_dvec.c"
#include "gvec_to_xy.c"
#include "xy_to_gvec.c"
#include "oscill_angles_of_HKLs.c"
#include "unit_row_vector.c"
/* #include "make_detector_rmat.c" */
#include "make_sample_rmat.c"
#include "make_rmat_of_expmap.c"
#include "make_binary_rmat.c"
#include "make_beam_rmat.c"
#include "validate_angle_ranges.c"
#include "rotate_vecs_about_axis.c"
#include "quat_distance.c"



/* =============================================================================
 * Module initialization
 * =============================================================================
 */

/*#define EXPORT_METHOD(name)                                           \
    { STR(name), CONCAT(python_, name), METH_VARARGS, CONCAT(docstring_, name) }
*/
#define EXPORT_METHOD(name) \
    { STR(name), CONCAT(python_, name), METH_VARARGS, "" }

static PyMethodDef _module_methods[] = {
    EXPORT_METHOD(anglesToGVec), /* angles_to_gvec */
    EXPORT_METHOD(anglesToDVec), /* angles_to_dvec */
    EXPORT_METHOD(gvecToDetectorXY),  /* gvec_to_xy */
    EXPORT_METHOD(gvecToDetectorXYArray), /* gvec_to_xy */
    EXPORT_METHOD(detectorXYToGvec), /* xy_to_gvec */
    EXPORT_METHOD(oscillAnglesOfHKLs), /* solve_omega */
    EXPORT_METHOD(unitRowVector), /* unit_vector */
    EXPORT_METHOD(unitRowVectors), /* unit_vector */
    EXPORT_METHOD(makeOscillRotMat), /* make_sample_rmat */
    EXPORT_METHOD(makeOscillRotMatArray), /* make_sample_rmat */
    EXPORT_METHOD(makeRotMatOfExpMap), /* make_rmat_of_expmap */
    EXPORT_METHOD(makeBinaryRotMat), /* make_binary_rmat */
    EXPORT_METHOD(makeEtaFrameRotMat), /* make_beam_rmat */
    EXPORT_METHOD(validateAngleRanges), /* validate_angle_ranges */
    EXPORT_METHOD(rotate_vecs_about_axis), /* rotate_vecs_about_axis */
    EXPORT_METHOD(quat_distance), /* quat_distance */
    /* adapted */
    /*  {"anglesToGVec",anglesToGVec,METH_VARARGS,"take angle tuples to G-vectors"},*/
    /*  {"anglesToDVec",anglesToDVec,METH_VARARGS,"take angle tuples to unit diffraction vectors"},*/
    /*  {"gvecToDetectorXY",gvecToDetectorXY,METH_VARARGS,""},*/
    /*  {"gvecToDetectorXYArray",gvecToDetectorXYArray,METH_VARARGS,""},*/
    /*  {"detectorXYToGvec",detectorXYToGvec,METH_VARARGS,"take cartesian coordinates to G-vectors"},*/
    /*  {"oscillAnglesOfHKLs",oscillAnglesOfHKLs,METH_VARARGS,"solve angle specs for G-vectors"},*/
    /*  {"unitRowVector",unitRowVector,METH_VARARGS,"Normalize a single row vector"},*/
    /*  {"unitRowVectors",unitRowVectors,METH_VARARGS,"Normalize a collection of row vectors"},*/
    /*  {"makeOscillRotMat",makeOscillRotMat,METH_VARARGS,""},*/
    /*  {"makeOscillRotMatArray",makeOscillRotMatArray,METH_VARARGS,""},*/
    /*  {"makeRotMatOfExpMap",makeRotMatOfExpMap,METH_VARARGS,""},*/
    /*  {"makeBinaryRotMat",makeBinaryRotMat,METH_VARARGS,""},*/
    /*  {"makeEtaFrameRotMat",makeEtaFrameRotMat,METH_VARARGS,"Make eta basis COB matrix"},*/
    /*  {"validateAngleRanges",validateAngleRanges,METH_VARARGS,""}, */
    /*  {"rotate_vecs_about_axis",rotate_vecs_about_axis,METH_VARARGS,"Rotate vectors about an axis"},*/
    /*  {"quat_distance",quat_distance,METH_VARARGS,"Compute distance between two unit quaternions"},*/

    /* adapted but not part of the API, so not exported */
    /*  {"makeDetectorRotMat",makeDetectorRotMat,METH_VARARGS,""},*/
    
    /* removed... */
    /*  {"makeGVector",makeGVector,METH_VARARGS,"Make G-vectors from hkls and B-matrix"},*/
    /*  {"arccosSafe",arccosSafe,METH_VARARGS,""},*/
    /*  {"angularDifference",angularDifference,METH_VARARGS,"difference for cyclical angles"},*/
    /*  {"mapAngle",mapAngle,METH_VARARGS,"map cyclical angles to specified range"},*/
    /*  {"columnNorm",columnNorm,METH_VARARGS,""},*/
    /*  {"rowNorm",rowNorm,METH_VARARGS,""},*/
    /*  {"makeRotMatOfQuat",makeRotMatOfQuat,METH_VARARGS,""},*/
    /*  {"homochoricOfQuat",homochoricOfQuat,METH_VARARGS,"Compute homochoric parameterization of list of unit quaternions"},*/
    {NULL,NULL,0,NULL} /* sentinel */
};

/*
 * In Python 3 the entry point for a C module changes slightly, but both can
 * be supported with little effort with some conditionals and macro magic
 */

#if PY_VERSION_HEX >= 0x03000000
#  define IS_PY3K
#endif

#if defined(IS_PY3K)
/* a module definition structure is required in Python 3 */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        STR(THIS_MODULE_NAME),
        NULL,
        -1,
        _module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(IS_PY3K)
#  define MODULE_INIT_FUNC_NAME CONCAT(PyInit_, THIS_MODULE_NAME)
#  define MODULE_RETURN(module) return module
#else
#  define MODULE_INIT_FUNC_NAME CONCAT(init, THIS_MODULE_NAME)
#  define MODULE_RETURN(module) return
#endif
PyMODINIT_FUNC
MODULE_INIT_FUNC_NAME(void)
{
    PyObject *module;

#if defined(IS_PY3K)
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule(STR(THIS_MODULE_NAME),_module_methods);
#endif
    if (NULL == module)
        MODULE_RETURN(module);

    import_array();
    
    MODULE_RETURN(module);
}

