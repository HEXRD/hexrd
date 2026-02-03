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
#define THIS_MODULE_NAME transforms_c_api

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
#include "ndargs_helper.h"

#include "ndargs_helper.c"
#include "angles_to_gvec.c"
#include "angles_to_dvec.c"
#include "gvec_to_xy.c"
#include "xy_to_gvec.c"
#include "oscill_angles_of_HKLs.c"
#include "unit_row_vector.c"
#include "make_detector_rmat.c"
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

/*
 * Module initialization (Python 3 only)
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
    EXPORT_METHOD(makeRotMatOfExpMap), /* make_rmat_of_expmap */
    EXPORT_METHOD(makeDetectorRotMat), /* make_detector_rmat */
    EXPORT_METHOD(makeBinaryRotMat), /* make_binary_rmat */
    EXPORT_METHOD(makeEtaFrameRotMat), /* make_beam_rmat */
    EXPORT_METHOD(validateAngleRanges), /* validate_angle_ranges */
    EXPORT_METHOD(rotate_vecs_about_axis), /* rotate_vecs_about_axis */
    EXPORT_METHOD(quat_distance), /* quat_distance */
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    STR(THIS_MODULE_NAME),
    NULL,          /* m_doc */
    -1,            /* m_size */
    _module_methods,
    NULL,          /* m_slots */
    NULL,          /* m_traverse */
    NULL,          /* m_clear */
    NULL           /* m_free */
};

PyMODINIT_FUNC CONCAT(PyInit_, THIS_MODULE_NAME)(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;
    }

    import_array();

    return module;
}

