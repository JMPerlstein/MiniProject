#include <tbb/parallel_invoke.h>
#include <tbb/tbb.h>
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

static double timer() {

    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

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

inline void merge_serial(float Input[], float Output[], int p1, int r1, int p2, int r2, int p3)
{

    int len1 = r1 - p1 + 1;
    int len2 = r2 - p2 + 1;

    if (len1 < len2)
    {
        SWAP(r1, r2, int);
        SWAP(p1, p2, int);
        SWAP(len1, len2, int);
    }

    if (len1 == 0)
        return;

    int mid1 = (p1 + r1) / 2;
    int mid2 = binarySearch(Input[mid1], Input, p2, r2);
    int mid3 = p3 + (mid1 - p1) + (mid2 - p2);

    Output[mid3] = Input[mid1];

    merge_serial(Input, Output, p1, mid1-1, p2, mid2-1, p3);
    merge_serial(Input, Output, mid1+1, r1, mid2, r2, mid3 +1);
}

inline void mergeSort_serial_recursion(float Input[], float Output[], int l, int r, bool direction)
{
    if (r == l)
    {
        if (direction) Output[l] = Input[l];
        return;
    }

    int mid = (r+l) / 2;

    mergeSort_serial_recursion(Input, Output, l, mid, !direction);
    mergeSort_serial_recursion(Input, Output, mid+1, r, !direction);

    if (direction)
        merge_serial(Input, Output, l, mid, mid+1, r, l);
    else
        merge_serial(Output, Input, l, mid, mid+1, r, l);
}


inline void merge(float Input[], float Output[], int p1, int r1, int p2, int r2, int p3, int threads)
{

    int len1 = r1 - p1 + 1;
    int len2 = r2 - p2 + 1;

    if (len1 < len2)
    {
        SWAP(r1, r2, int);
        SWAP(p1, p2, int);
        SWAP(len1, len2, int);
    }

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
        tbb::parallel_invoke(
               [=]{ merge(Input, Output, p1, mid1-1, p2, mid2-1, p3, threads/2); },
               [=]{ merge(Input, Output, mid1+1, r1, mid2, r2, mid3 +1, threads/2); } );
    }
}



inline void mergeSort_parallel_recursion(float Input[], float Output[], int l, int r, bool direction, int threads)
{

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
        tbb::parallel_invoke(
             [=]{ mergeSort_parallel_recursion(Input, Output, l, mid, !direction, threads/2); },
             [=]{ mergeSort_parallel_recursion(Input, Output, mid+1, r, !direction, threads/2); } );
    }

    if (direction)
          merge(Input, Output, l, mid, mid+1, r, l, threads);
    else
        merge(Output, Input, l, mid, mid+1, r, l, threads);
}

static int mergesortTBB(const float *A, const int n, const int num_iterations)
{

    fprintf(stderr, "N %d\n", n);
    fprintf(stderr, "Using TBB Mergesort\n");
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

        int nthread = 8;
        tbb::task_scheduler_init init(nthread);

        mergeSort_parallel_recursion(B, Output, 0, n-1, true, nthread);


        elt = timer() - elt;
        avg_elt += elt;
        fprintf(stderr, "%9.3lf\n", elt*1e3);

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

int gen_input(float *A, int n, int input_type)
{

    int i;
 
    if (input_type == 0) {

        srand(123);
        for (i=0; i<n; i++) {
            A[i] = ((float) rand())/((float) RAND_MAX);
        }

    } else if (input_type == 1) {

        for (i=0; i<n; i++) {
            A[i] = (float) i;
        }

    } else if (input_type == 2) {

        for (i=0; i<n; i++) {
            A[i] = (float) i;
        }

        int num_shuffles = (n/100) + 1;
        srand(1234);
        for (i=0; i<num_shuffles; i++) {
            int j = (rand() % n);
            int k = (rand() % n);

            float tmpval = A[j];
            A[j] = A[k];
            A[k] = tmpval;
        }

    } else if (input_type == 3) {

        for (i=0; i<n; i++) {
            A[i] = 1.0;
        }

    } else {

        for (i=0; i<n; i++) {
            A[i] = (float) (n + 1.0 - i);
        }
    }

    return 0;

}

int main(int argc, char **argv)
{

    if (argc != 3) {
        fprintf(stderr, "%s <n> <input_type> \n", argv[0]);
        fprintf(stderr, "input_type 0: uniform random\n");
        fprintf(stderr, "           1: already sorted\n");
        fprintf(stderr, "           2: almost sorted\n");
        fprintf(stderr, "           3: single unique value\n");
        fprintf(stderr, "           4: sorted in reverse\n");
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

    int num_iterations = 10;

    mergesortTBB(A, n, num_iterations);


    free(A);

    return 0;
}
                     
                     
