/* 	RECURSIVE PARALLEL MERGE SORT for sorting 32-bit floating point values
	JESSE PERLSTEIN

	REFERENCEs:	
				Parallel Merge: 	http://www.drdobbs.com/parallel/parallel-merge/229204454
				Parallel Mergesort:	http://www.drdobbs.com/parallel/parallel-merge-sort/229400239
*/
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double timer() 
{
    
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}



//DATA SWAP MACRO
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)


// BINARY SEARCH HELPER FUNCTION
int binarySearch(float val, const float A[], int left, int right )
{
    float low  = left;
    float high = fmax( left, right + 1 );

    while(low < high)
    {
        int mid = (low + high) / 2;
        if (val <= A[mid]) 
        	high = mid;
        else 
        	low  = mid + 1; 
    }
    return (int) high;
}



//SERIAL MERGE FUNCTION
inline void merge_serial(float Input[], float Output[], int p1, int r1, int p2, int r2, int p3)
{
//	Input: [... p1 ...mid1 ... r1 ... p2 ... mid2 ... r2 ...] 
//  Output: [... p3 ... mid3 ... q3 ...]
	
	int len1 = r1 - p1 + 1;
	int len2 = r2 - p2 + 1;

	if (len1 < len2)
	{
		SWAP(r1, r2, int);
		SWAP(p1, p2, int);
		SWAP(len1, len2, int);
	}

	//Base Case
	if (len1 == 0)
		return; 

	int mid1 = (p1 + r1) / 2;
	int mid2 = binarySearch(Input[mid1], Input, p2, r2);
	int mid3 = p3 + (mid1 - p1) + (mid2 - p2);

	Output[mid3] = Input[mid1];

	//Serial Recursion
	merge_serial(Input, Output, p1, mid1-1, p2, mid2-1, p3);
	merge_serial(Input, Output, mid1+1, r1, mid2, r2, mid3 +1);
}



//SERIAL MERGESORT FUNCTION
inline void mergeSort_serial_recursion(float Input[], float Output[], int l, int r, bool direction)
{
	//Base Case
	if (r == l)
	{
		if (direction) Output[l] = Input[l];
		return;
	}
	
	int mid = (r+l) / 2;

	//Serial Recursion
	mergeSort_serial_recursion(Input, Output, l, mid, !direction);
	mergeSort_serial_recursion(Input, Output, mid+1, r, !direction);

	if (direction)
		merge_serial(Input, Output, l, mid, mid+1, r, l);
    else          
    	merge_serial(Output, Input, l, mid, mid+1, r, l);
}



//PARALLEL MERGE FUNCTION
inline void merge(float Input[], float Output[], int p1, int r1, int p2, int r2, int p3, int threads)
{
//	Input: [... p1 ...mid1 ... r1 ... p2 ... mid2 ... r2 ...] 
//  Output: [... p3 ... mid3 ... q3 ...]
	
	int len1 = r1 - p1 + 1;
	int len2 = r2 - p2 + 1;

	if (len1 < len2)
	{
		SWAP(r1, r2, int);
		SWAP(p1, p2, int);
		SWAP(len1, len2, int);
	}

	//Base Case
	if (len1 == 0)
		return; 

	int mid1 = (p1 + r1) / 2;
	int mid2 = binarySearch(Input[mid1], Input, p2, r2);
	int mid3 = p3 + (mid1 - p1) + (mid2 - p2);

	Output[mid3] = Input[mid1];

	if (threads == 1)
	{
		merge_serial(Input, Output, p1, mid1-1, p2, mid2-1, p3);
		merge_serial(Input, Output, mid1+1, r1, mid2, r2, mid3 +1);
	}
	else if (threads > 1)
	{
		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
				{merge(Input, Output, p1, mid1-1, p2, mid2-1, p3, threads/2);}
			#pragma omp section
				{merge(Input, Output, mid1+1, r1, mid2, r2, mid3 +1, threads/2);}
		}
	}
}



//MAIN PARALLEL RECURSIVE MERGE SORT
inline void mergeSort_parallel_recursion(float Input[], float Output[], int l, int r, bool direction, int threads)
{
	//Base Case
	if (r == l)
	{
		if (direction) Output[l] = Input[l];
		return;
	}
	
	int mid = (r+l) / 2;

	if (threads == 1)
	{
		mergeSort_serial_recursion(Input, Output, l, mid, !direction);
		mergeSort_serial_recursion(Input, Output, mid+1, r, !direction);
	}
	else if(threads > 1)
	{
		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
				{mergeSort_parallel_recursion(Input, Output, l, mid, !direction, threads/2);}
			#pragma omp section
				{mergeSort_parallel_recursion(Input, Output, mid+1, r, !direction, threads/2);}
		}
	}

	if (direction)
		merge(Input, Output, l, mid, mid+1, r, l, threads);
    else          
    	merge(Output, Input, l, mid, mid+1, r, l, threads);
}


static int mergesort(const float *A, const int n, const int num_iterations) 
{

    fprintf(stderr, "N %d\n", n);
    fprintf(stderr, "Using Mergesort\n");
    fprintf(stderr, "Execution times (ms) for %d iterations:\n", num_iterations);

    int iter;
    double avg_elt;

    float *B;
    B = (float *) malloc(n * sizeof(float));
    assert(B != NULL);

    float *Output;
    Output = (float *) malloc(n * sizeof(float));
    assert(Output != NULL);


    avg_elt = 0.0;

    for (iter = 0; iter < num_iterations; iter++) {
        
        int i;

        for (i=0; i<n; i++) {
            B[i] = A[i];
        }

        double elt;
        elt = timer();


        omp_set_nested(1);
        omp_set_num_threads(omp_get_max_threads());

        mergeSort_parallel_recursion(B, Output, 0, n-1, true, omp_get_max_threads());
        

        elt = timer() - elt;
        avg_elt += elt;
        fprintf(stderr, "%9.3lf\n", elt*1e3);

        /* correctness check */
        for (i=1; i<n; i++) {
            assert(Output[i] >= Output[i-1]);
        }

    }

    avg_elt = avg_elt/num_iterations;
    
    free(B);
    free(Output);

    fprintf(stderr, "Average time: %9.3lf ms.\n", avg_elt*1e3);
    fprintf(stderr, "Average sort rate: %6.3lf MB/s\n", 4.0*n/(avg_elt*1e6));
    return 0;

}


static int mergesort_serial(const float *A, const int n, const int num_iterations) 
{

    fprintf(stderr, "N %d\n", n);
    fprintf(stderr, "Using Mergesort\n");
    fprintf(stderr, "Execution times (ms) for %d iterations:\n", num_iterations);

    int iter;
    double avg_elt;

    float *B;
    B = (float *) malloc(n * sizeof(float));
    assert(B != NULL);

    float *Output;
    Output = (float *) malloc(n * sizeof(float));
    assert(Output != NULL);


    avg_elt = 0.0;

    for (iter = 0; iter < num_iterations; iter++) {
        
        int i;

        for (i=0; i<n; i++) {
            B[i] = A[i];
        }

        double elt;
        elt = timer();
           //Printing of all elements of array

        mergeSort_serial_recursion(B, Output, 0, n-1, true);

        elt = timer() - elt;
        avg_elt += elt;
        fprintf(stderr, "%9.3lf\n", elt*1e3);

        /* correctness check */
        for (i=1; i<n; i++) {
            assert(Output[i] >= Output[i-1]);
        }

    }

    avg_elt = avg_elt/num_iterations;
    
    free(B);
    free(Output);

    fprintf(stderr, "Average time: %9.3lf ms.\n", avg_elt*1e3);
    fprintf(stderr, "Average sort rate: %6.3lf MB/s\n", 4.0*n/(avg_elt*1e6));
    return 0;

}


/* generate different inputs for testing sort */
int gen_input(float *A, int n, int input_type) {

    int i;

    /* uniform random values */
    if (input_type == 0) {

        srand(123);
        for (i=0; i<n; i++) {
            A[i] = ((float) rand())/((float) RAND_MAX);
        }

    /* sorted values */    
    } else if (input_type == 1) {

        for (i=0; i<n; i++) {
            A[i] = (float) i;
        }

    /* almost sorted */    
    } else if (input_type == 2) {

        for (i=0; i<n; i++) {
            A[i] = (float) i;
        }

        /* do a few shuffles */
        int num_shuffles = (n/100) + 1;
        srand(1234);
        for (i=0; i<num_shuffles; i++) {
            int j = (rand() % n);
            int k = (rand() % n);

            /* swap A[j] and A[k] */
            float tmpval = A[j];
            A[j] = A[k];
            A[k] = tmpval;
        }

    /* array with single unique value */    
    } else if (input_type == 3) {

        for (i=0; i<n; i++) {
            A[i] = 1.0;
        }

    /* sorted in reverse */    
    } else {

        for (i=0; i<n; i++) {
            A[i] = (float) (n + 1.0 - i);
        }

    }

    return 0;

}


int main(int argc, char **argv) {

    if (argc != 4) {
        fprintf(stderr, "%s <n> <input_type> <alg_type>\n", argv[0]);
        fprintf(stderr, "input_type 0: uniform random\n");
        fprintf(stderr, "           1: already sorted\n");
        fprintf(stderr, "           2: almost sorted\n");
        fprintf(stderr, "           3: single unique value\n");
        fprintf(stderr, "           4: sorted in reverse\n");
        fprintf(stderr, "alg_type   0: use parallel mergesort\n");
        fprintf(stderr, "           1: use serial mergesort\n");
        exit(1);  
    }

    int n;

    n = atoi(argv[1]);

    assert(n > 0);
    assert(n <= 1000000000);

    float *A;
    A = (float *) malloc(n * sizeof(float));
    assert(A != 0);

    int input_type = atoi(argv[2]);
    assert(input_type >= 0);
    assert(input_type <= 4);

    gen_input(A, n, input_type);

    int alg_type = atoi(argv[3]);

    int num_iterations = 10;
    
    assert((alg_type == 0) || (alg_type == 1) || (alg_type == 2) || (alg_type == 3));

     if (alg_type == 0) {
        mergesort(A, n, num_iterations);
    }else if (alg_type == 1) {
        mergesort_serial(A, n, num_iterations);
    }

    free(A);

    return 0;
}
