#include <stdio.h>



static void debug_dump_val(const char *name, double val)
{
    printf("%s: %10.6f\n", name, val);
}

static void debug_dump_m33(const char *name, double *array)
{
    printf("%s:\n", name);
    printf("\t%10.6f %10.6f %10.6f\n", array[0], array[1], array[2]);
    printf("\t%10.6f %10.6f %10.6f\n", array[3], array[4], array[5]);
    printf("\t%10.6f %10.6f %10.6f\n", array[6], array[7], array[8]);
}

static void debug_dump_v3(const char *name, double *vec)
{
    printf("%s: %10.6f %10.6f %10.6f\n", name, vec[0], vec[1], vec[2]);
}

/* ============================================================================
 * These can be used to initialize and check buffers if we suspect the
 * code may leave it uninitialized
 * ============================================================================
 */

#define SNAN_HI 0x7ff700a0
#define SNAN_LO 0xbad0feed
void fill_signaling_nans(double *arr, int count)
{
    int i;
    npy_uint32 *arr_32 = (npy_uint32 *)arr;
    /* Fills an array with signaling nans to detect errors
     * Use the 0x7ff700a0bad0feed as the pattern
     */
    for (i = 0; i < count; ++i)
    {
        arr_32[2*i+0] = SNAN_LO;
        arr_32[2*i+1] = SNAN_HI;
    }
}

int detect_signaling_nans(double *arr, int count)
{
    int i;
    npy_uint32 *arr_32 = (npy_uint32 *) arr;
    for (i = 0; i < count; ++i)
    {
        if (arr_32[2*i+0] == SNAN_LO &&
            arr_32[2*i+1] == SNAN_HI)
        {
            return 1;
        }
    }

    return 0;
}
#undef SNAN_HI
#undef SNAN_LO
