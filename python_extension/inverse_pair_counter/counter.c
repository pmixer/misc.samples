#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

int merge_and_count(long long *arr, int lo, int mi, int hi, long long *buff)
{
    int res = 0;
    for (int i = lo; i < mi; i++) buff[i] = arr[i];
    
    int i = lo, j = mi, curr = lo;
    
    while (i < mi && j < hi) {
        if (buff[i] <= arr[j])
        {
            arr[curr] = buff[i];
            i++;
        } else {
            arr[curr] = arr[j];
            res += (mi - i);
            j++;
        }
        curr++;
    }
    
    while (i < mi) {
        arr[curr++] = buff[i++];
    }
    
    while (j < hi) {
        res += (mi - i);
        j++;
    }

    return res;
    
}

int count_inv_pair(long long *arr, int lo, int hi, long long *buff)
{
    if (hi - lo < 2) return 0;
    int mi = (lo + hi) >> 1;
    int res1 = count_inv_pair(arr, lo, mi, buff);
    int res2 = count_inv_pair(arr, mi, hi, buff);
    int res3 = merge_and_count(arr, lo, mi, hi, buff);
    return res1+res2+res3;
}

static PyObject* inversion_count_c(PyObject *self, PyObject *args)
{
    PyObject *rank_list_obj = NULL;
    PyObject *buff_for_mergesort_obj = NULL;

    long long *rank_list, *buff_for_mergesort;

    if (!PyArg_ParseTuple(args, "OO", &rank_list_obj, &buff_for_mergesort_obj))
        return NULL;

    // no check and error report if buff is smaller than ranklist currently
    rank_list = (long long *)PyArray_DATA((PyArrayObject *)rank_list_obj);
    buff_for_mergesort = (long long *)PyArray_DATA((PyArrayObject *)buff_for_mergesort_obj);
    
    npy_intp *shape = PyArray_DIMS((PyArrayObject *)rank_list_obj);
    
    int inversion_num = count_inv_pair(rank_list, 0, shape[0], buff_for_mergesort);

    return PyInt_FromLong(inversion_num);
}


static PyMethodDef counter_methods[] = {
        {
                "inversion_count", inversion_count_c, METH_VARARGS,
                "1D numpy inversion counter",
        },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef counter_definition = {
        PyModuleDef_HEAD_INIT,
        "counter",
        "A Python module implemented in C for counting.",
        -1,
        counter_methods
};


PyMODINIT_FUNC PyInit_counter(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&counter_definition);
}
