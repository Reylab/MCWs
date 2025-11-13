#include "mex.h"
#include <math.h>
#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }


/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 */



void Quickselect(double *arr, double *z, mwSize n) 
{
    mwSize low, high ;
    mwSize median;
    mwSize middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low){ /* One element only */
            *z = arr[median];
            return;
        }
        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            *z = arr[median];
            return;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
}
#undef ELEM_SWAP

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

void Quickselect_f(float *arr, float *z, mwSize n) 
{
    mwSize  low, high ;
    mwSize  median;
    mwSize middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low){ /* One element only */
            *z = arr[median];
            return;
        }
        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            *z = arr[median];
            return;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
}




#undef ELEM_SWAP


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *inMatrix, *inMatrix_copy, *outValue;   /* 1xN input matrix, local copy and out */
    float *inMatrixf, *inMatrix_copyf, *outValuef;               /* 1xN copy of input matrix */
    size_t nrows, i;                   /* size of matrix */
    
    

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("FerToolbox:fastAbsMedian:nrhs","One inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("FerToolbox:fastAbsMedian:nlhs","One output required.");
    }

    
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[0]) &&
         !mxIsSingle(prhs[0])) {
        mexErrMsgIdAndTxt("FerToolbox:fastAbsMedian:notDoubleorSingle","Input matrix must be type double or single.");
    }
        
    /* check that number of rows in second input argument is 1 */
    if(mxGetN(prhs[0])>1) {
        mexErrMsgIdAndTxt("FerToolbox:fastAbsMedian:notRowVector","Input must be a column vector.");
    }
    /* get dimensions of the input matrix */
    nrows = mxGetM(prhs[0]);
    if (mxIsDouble(prhs[0])){
        /* create a pointer to the real data in the input matrix  */
        #if MX_HAS_INTERLEAVED_COMPLEX
        inMatrix = mxGetDoubles(prhs[0]);
        #else
        inMatrix = mxGetPr(prhs[0]);
        #endif
        /* create the output matrix */
        plhs[0] = mxCreateDoubleMatrix(1,(mwSize)1,mxREAL);
        /* get a pointer to the real data in the output matrix */
        #if MX_HAS_INTERLEAVED_COMPLEX
        outValue = mxGetDoubles(plhs[0]);
        #else
        outValue = mxGetPr(plhs[0]);
        #endif
        
        inMatrix_copy = mxMalloc(nrows * sizeof(*inMatrix_copy));
        for (i=0;i<nrows;i++){
            inMatrix_copy[i] =fabs(inMatrix[i]);
        }
    }
    else {
        /* create a pointer to the real data in the input matrix  */
        #if MX_HAS_INTERLEAVED_COMPLEX
        inMatrixf = mxGetSingles(prhs[0]);
        #else
        inMatrixf = mxGetPr(prhs[0]);
        #endif
        /* create the output matrix */
        plhs[0] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
        /* get a pointer to the real data in the output matrix */
        #if MX_HAS_INTERLEAVED_COMPLEX
        outValuef = mxGetSingles(plhs[0]);
        #else
        outValuef = mxGetPr(plhs[0]);
        #endif
        inMatrix_copyf = mxMalloc(nrows * sizeof(*inMatrix_copyf));
        for (i=0;i<nrows;i++){
            inMatrix_copyf[i] =fabsf(inMatrixf[i]);
        }
    }
        


    
    /* call the computational routine */
    if (mxIsDouble(prhs[0])){
        Quickselect(inMatrix_copy,outValue,(mwSize)nrows);
        mxFree(inMatrix_copy);
    }
    else{
        Quickselect_f(inMatrix_copyf,outValuef,(mwSize)nrows);
        mxFree(inMatrix_copyf);
    }
}
