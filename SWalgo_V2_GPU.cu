/***********************************************************************
 * Smith–Waterman algorithm
 * Purpose:     Local alignment of nucleotide or protein sequences
 * Authors:     Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 ***********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <stdbool.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>

/*--------------------------------------------------------------------
 * Text Tweaks
 */
#define RESET   "\033[0m"
#define BOLDRED "\033[1m\033[31m"      /* Bold Red */
/* End of text tweaks */

/*--------------------------------------------------------------------
 * Constants
 */
#define PATH -1
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
#define block_size 100

#define OUTER_LOOP_ITERS 1000  /* outer loop iterations */
#define MAXSIZE 1000000 /* inner loop iterations, and row length of arrays */
#define FLOPs_per_Loop 60.0 /* MODIFY this and ALL other lines marked "MODIFY" */
#define Unique_Reads_per_Loop 9.0 /* MODIFY this and ALL other lines marked "MODIFY" */
#define Arithmetic_Intensity (FLOPs_per_Loop/Unique_Reads_per_Loop)   /* recommended range =  [1/8 , 2] */


//#define DEBUG
//#define pragmas
/* End of constants */

#include <cmath>
#include <omp.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/*--------------------------------------------------------------------
 * Functions Prototypes
 */
void similarityScore(long long int ind, long long int ind_u, long long int ind_d, long long int ind_l, long long int ii, long long int jj, int* H, int* P, long long int max_len, long long int* maxPos, long long int *maxPos_max_len);
int  matchMissmatchScore(long long int i, long long int j);
void backtrack(int* P, long long int maxPos, long long int maxPos_max_len);
void printMatrix(int* matrix);
void printPredecessorMatrix(int* matrix);
void generate(void);
/* End of prototypes */


/*--------------------------------------------------------------------
 * Global Variables
 */
//Defines size of strings to be compared
long long int m = 11; //Columns - Size of string a
long long int n = 7;  //Lines - Size of string b

//Defines scores
int matchScore = 5;
int missmatchScore = -3;
int gapScore = -4;

//Strings over the Alphabet Sigma
char *a, *b;

/* End of global variables */

/*--------------------------------------------------------------------
 * Function:    main
 */
//typedef struct  {
//    long long int i;
//    long long int max_len;
//    long long int ind;
//    long long int ind_u;
//    long long int ind_d;
//    long long int ind_l;
//    long long int m;
//    long long int j_start;
//} similarity_parameters;



__global__ void smilarity(long long int* i, long long int* m, long long int* n,
                          char* a_gpu, char* b_gpu,int* H, int* P,
                          long long int *maxPos, long long int *maxPos_max_len) {

    long long int ind, max_len, ind_d, ind_u, ind_l, offset, jj, j_end;
    long long int ii = *i, mm = *m, nn = *n;
    ii++;

    if(ii<nn){
        ind = (ii)*(ii+1)/2;
    }
    else if(ii >= mm){
        ind = nn * mm - (nn-(ii-mm))*(nn-(ii-mm)-1)/2;
    }
    else{
        ind  = (ii)*(ii+1)/2 + (ii-nn+1)* (nn);
    }

    if (ii < nn){
        max_len = ii + 1;
        j_end   = max_len-1;
        ind_u   = ind - max_len;
        ind_l   = ind - max_len + 1;
        ind_d   = ind - (max_len<<1) + 2;
    }
    else if (ii >= mm){
        max_len = mm + nn - 1 - ii;
        j_end   = max_len-1;
        ind_u   = ind - max_len - 1;
        ind_l   = ind - max_len;
        ind_d   = ind - (max_len<<1) - 2;
    }
    else{
        max_len   = nn;
        j_end   = max_len-1;
        ind_u     = ind - max_len - 1;
        ind_l     = ind - max_len;
        if(ii>nn)
            ind_d = ind - (max_len<<1) - 1;
        else
            ind_d = ind - (max_len<<1);
    }

    long long int j = blockIdx.x * blockDim.x +threadIdx.x;
    if(ii<mm)
        j++;
    if (j<j_end){

        ind   = ind + j;
        ind_d = ind_d + j;
        ind_u = ind_u + j;
        ind_l = ind_l + j;

        //Columns long long int ind
        if (ii<mm){
            ii = ii-j;
            jj = j;
        }
        else{
            ii = mm - 1- j;
            jj = ii - mm + j+1;
        }

        int up, left, diag;

        //Get element above
        up = H[ind_u] - 4;

        //Get element on the left
        left = H[ind_l] - 4;

        //Get element on the diagonal
        if (a_gpu[ii - 1] == b_gpu[jj - 1]) {
            diag = H[ind_d] + 5;
        } else {
            diag = H[ind_d] - 3;
        }

        //Calculates the maximum
        int max = 0;
        int pred = 0;
        /* === Matrix ===
         *      a[0] ... a[n]
         * b[0]
         * ...
         * b[n]
         *
         * generate 'a' from 'b', if '←' insert e '↑' remove
         * a=GAATTCA
         * b=GACTT-A
         *
         * generate 'b' from 'a', if '←' insert e '↑' remove
         * b=GACTT-A
         * a=GAATTCA
        */

        if (diag > max) { //same letter ↖
            max = diag;
            pred = 3;
        }

        if (up > max) { //remove letter ↑
            max = up;
            pred = 1;
        }

        if (left > max) { //insert letter ←
            max = left;
            pred = 2;
        }

        //Inserts the value in the similarity and predecessor matrixes
        H[ind] = max;
        P[ind] = pred;

        //Updates maximum score to be used as seed on backtrack
        if (max > H[*maxPos]) {
            *maxPos = ind;
            *maxPos_max_len = max_len;
        }

        __shared__ int syncValue;

        if(threadIdx.x == 0) {
            // set the sync value to a known value
            syncValue = 0;

            // ensure all blocks have set the sync value
            __threadfence();
            atomicAdd(&syncValue, 1);

            // wait for all blocks to finish setting the sync value
            while(syncValue < gridDim.x) {}
        }
        if(threadIdx.x == 0 && blockIdx.x == 0)
            *i = ii;
    } //60
}

/* gtod_seconds() gets the current time in seconds using gettimeofday */
double gtod_seconds(void)
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}



int main(int argc, char* argv[]) {

    if(argc>1){
        m = strtoll(argv[1], NULL, 10);
        n = strtoll(argv[2], NULL, 10);
        long long int temp;
        if(m < n){
            temp = m;
            m = n;
            n = temp;
        }
    }
    else{
        m = 10;
        n = 8;
    }

#ifdef DEBUG
    printf("\nMatrix[%lld][%lld]\n", n, m);
#endif

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    //Allocates similarity matrix
    int *H_gpu;
    int *P_gpu;
    int *H;
    int *P;
    char *a_gpu;
    char *b_gpu;
    long long int maxPos = 0;
    long long int maxPos_max_len = 0;
    long long int *maxPos_gpu;
    long long int *maxPos_max_len_gpu;

    //Allocates a and b
    int allocSize_a = m * sizeof(char);
    int allocSize_b = n * sizeof(char);

    int size = sizeof(long long int);

    a = (char*)malloc(allocSize_a);
    b = (char*)malloc(allocSize_b);

    //Because now we have zeros
    m++;
    n++;

    int allocSize_matrix = m * n * sizeof(int);
    H = (int*)calloc(m * n, sizeof(int));
    P = (int*)calloc(m * n, sizeof(int));

    //Gen rand arrays a and b
    generate();
    // a[0] =   'C';
    // a[1] =   'G';
    // a[2] =   'T';
    // a[3] =   'G';
    // a[4] =   'A';
    // a[5] =   'A';
    // a[6] =   'T';
    // a[7] =   'T';
    // a[8] =   'C';
    // a[9] =   'A';
    // a[10] =  'T';

    // b[0] =   'G';
    // b[1] =   'A';
    // b[2] =   'C';
    // b[3] =   'T';
    // b[4] =   'T';
    // b[5] =   'A';
    // b[6] =   'C';

#ifdef pragmas
#pragma GCC ivdep
#endif

    long long int i;
#ifdef DEBUG
    printf("\n a string:\n");
    for(i=0; i<m-1; i++)
        printf("%c ",a[i]);
    printf("\n b string:\n");
    for(i=0; i<n-1; i++)
        printf("%c ",b[i]);
    printf("\n");
#endif
    //set gpu timer
    cudaEvent_t start, stop;
    float elapsed_gpu;
    double initialTime = omp_get_wtime();
    //Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
    //copy memory from the host to gpu

    CUDA_SAFE_CALL(cudaMalloc((void **)&H_gpu, allocSize_matrix));
    CUDA_SAFE_CALL(cudaMalloc((void **)&P_gpu, allocSize_matrix));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a_gpu, allocSize_a));
    CUDA_SAFE_CALL(cudaMalloc((void **)&b_gpu, allocSize_b));
    CUDA_SAFE_CALL(cudaMalloc((void **)&maxPos_gpu, size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&maxPos_max_len_gpu, size));

    CUDA_SAFE_CALL(cudaMemcpy(maxPos_max_len_gpu, &maxPos_max_len, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(maxPos_gpu, &maxPos, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(a_gpu, a, allocSize_a, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(b_gpu, b, allocSize_b, cudaMemcpyHostToDevice));

#ifdef DEBUG
    printf("\n a string:\n");
    for(i=0; i<m-1; i++)
        printf("%c ",a[i]);
    printf("\n b string:\n");
    for(i=0; i<n-1; i++)
        printf("%c ",b[i]);
    printf("\n");
#endif
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    long long int j_start, j_end, ind = 3, max_len, *i_gpu, *m_gpu, *n_gpu;
    i=0;
    CUDA_SAFE_CALL(cudaMalloc((void **)&i_gpu, size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&m_gpu, size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&n_gpu, size));
    CUDA_SAFE_CALL(cudaMemcpy(m_gpu, &m, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(n_gpu, &n, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(i_gpu, &i, size, cudaMemcpyHostToDevice));

    double start_time;
    start_time = gtod_seconds();
    double total_time = 0; double time2;
    double total_flops, flops_per_second, gflops_per_second;
    float quasi_random = 0;
    float final_answer = 0;

    for (i = 2; i < m + n - 1; i++) { //Lines
        if (i < n){
            max_len = i + 1;
            j_start = 1;
            j_end   = max_len-1;
        }
        else if (i >= m){
            max_len = m + n - 1 - i;
            j_start = 0;
            j_end   = max_len;
        }
        else{
            max_len = n;
            j_start = 1;
            j_end   = max_len;
        }
        long long int row_len = j_end - j_start;
//        long long int offset = block_size * ((long long int) (row_len/block_size));

        dim3 block_Dim(block_size, 1, 1);
        dim3 grid_Dim((int) (row_len / block_size)+1, 1, 1); //7
        smilarity<<<grid_Dim, block_Dim, 0, stream1>>>(
                i_gpu,m_gpu, n_gpu,
                a_gpu, b_gpu, H_gpu, P_gpu,
                maxPos_gpu, maxPos_max_len_gpu);
    }
    //calculate AI
    total_time = gtod_seconds() - start_time;

    total_flops = ((double) FLOPs_per_Loop)
                  * ((double) OUTER_LOOP_ITERS)
                  * ((double) MAXSIZE);

    flops_per_second = total_flops / total_time;

    gflops_per_second = flops_per_second / 1.0e9;

    printf ("AI = %f    GFLOPs/s = %f    time = %f\n",
            Arithmetic_Intensity, gflops_per_second, total_time);


    cudaStreamSynchronize(stream1);

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    //copy the memory from gpu to host
    CUDA_SAFE_CALL(cudaMemcpy(H, H_gpu, allocSize_matrix, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(P, P_gpu, allocSize_matrix, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&maxPos, maxPos_gpu, size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&maxPos_max_len, maxPos_max_len_gpu, size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(a, a_gpu, allocSize_a, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(b, b_gpu, allocSize_b, cudaMemcpyDeviceToHost));

    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    backtrack(P, maxPos, maxPos_max_len);

    //Gets final time
    double finalTime = omp_get_wtime();
    printf("\nElapsed time: %f\n\n", finalTime - initialTime);

#ifdef DEBUG
    printf("\nSimilarity Matrix:\n");
    printMatrix(H);

    printf("\nPredecessor Matrix:\n");
    printPredecessorMatrix(P);
#endif

    //Frees similarity matrixes
    CUDA_SAFE_CALL(cudaFree(H_gpu));
    CUDA_SAFE_CALL(cudaFree(P_gpu));
    //Frees input arrays
    CUDA_SAFE_CALL(cudaFree(a_gpu));
    CUDA_SAFE_CALL(cudaFree(b_gpu));
    CUDA_SAFE_CALL(cudaFree(maxPos_max_len_gpu));
    CUDA_SAFE_CALL(cudaFree(maxPos_gpu));
    CUDA_SAFE_CALL(cudaFree(m_gpu));
    CUDA_SAFE_CALL(cudaFree(n_gpu));


    cudaStreamDestroy(stream1);
    free(a);
    free(b);
    free(H);
    free(P);

    return 0;
}  /* End of main */


/*--------------------------------------------------------------------
 * Function:    SimilarityScore
 * Purpose:     Calculate  the maximum Similarity-Score H(i,j)
 */
void similarityScore(long long int ind, long long int ind_u, long long int ind_d, long long int ind_l, long long int ii, long long int jj, int* H, int* P, long long int max_len, long long int* maxPos, long long int *maxPos_max_len) {

    int up, left, diag;

    //Get element above
    up = H[ind_u] + gapScore;

    //Get element on the left
    left = H[ind_l] + gapScore;

    //Get element on the diagonal
    diag = H[ind_d] + matchMissmatchScore(ii, jj);

    //Calculates the maximum
    int max = NONE;
    int pred = NONE;
    /* === Matrix ===
     *      a[0] ... a[n]
     * b[0]
     * ...
     * b[n]
     *
     * generate 'a' from 'b', if '←' insert e '↑' remove
     * a=GAATTCA
     * b=GACTT-A
     *
     * generate 'b' from 'a', if '←' insert e '↑' remove
     * b=GACTT-A
     * a=GAATTCA
    */

    if (diag > max) { //same letter ↖
        max = diag;
        pred = DIAGONAL;
    }

    if (up > max) { //remove letter ↑
        max = up;
        pred = UP;
    }

    if (left > max) { //insert letter ←
        max = left;
        pred = LEFT;
    }
    //Inserts the value in the similarity and predecessor matrixes
    H[ind] = max;
    P[ind] = pred;

    //Updates maximum score to be used as seed on backtrack
    if (max > H[*maxPos]) {
        *maxPos = ind;
        *maxPos_max_len = max_len;
    }

}  /* End of similarityScore */


/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */
int matchMissmatchScore(long long int i, long long int j) {
    if (a[i-1] == b[j-1])
        return matchScore;
    else
        return missmatchScore;
}  /* End of matchMissmatchScore */

/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
void backtrack(int* P, long long int maxPos, long long int maxPos_max_len) {
    //hold maxPos value
    long long int predPos;
    long long int predMaxLen;
#ifdef pragmas
#pragma GCC ivdep
#endif
    //backtrack from maxPos to startPos = 0
    long long int first_sec = (n*(n+1))/2;
    long long int last_sec  = n*m - (n*(n-1))/2;
    long long int ind_u, ind_d, ind_l;
    bool diagCompensate     = 0;
    do {
        if (maxPos<first_sec){
            if(diagCompensate){
                if(maxPos<first_sec-n)
                    maxPos_max_len --;
                diagCompensate = 0;
            }
            ind_u      = maxPos - maxPos_max_len;
            ind_l      = maxPos - maxPos_max_len + 1;
            ind_d      = maxPos - (maxPos_max_len<<1) + 2;
            predMaxLen = maxPos_max_len-1;
        }
        else if (maxPos>=last_sec){
            if(diagCompensate){
                maxPos_max_len ++;
                diagCompensate = 0;
            }
            ind_u      = maxPos - maxPos_max_len - 1;
            ind_l      = maxPos - maxPos_max_len;
            ind_d      = maxPos - (maxPos_max_len<<1) - 2;
            predMaxLen = maxPos_max_len+1;
        }
        else{
            if(diagCompensate){
                if(maxPos>=last_sec-n)
                    maxPos_max_len ++;
                diagCompensate = 0;
            }
            ind_u      = maxPos - n - 1;
            ind_l      = maxPos - n;
            predMaxLen = maxPos_max_len;
            if(maxPos>=first_sec+n)
                ind_d  = maxPos - (n<<1) - 1;
            else
                ind_d  = maxPos - (n<<1);
        }
        if(P[maxPos] == DIAGONAL){
            predPos        = ind_d;
            diagCompensate = 1;
        }
        else if(P[maxPos] == UP)
            predPos        = ind_u;
        else if(P[maxPos] == LEFT)
            predPos        = ind_l;
        P[maxPos]*=PATH;
        maxPos         = predPos;
        maxPos_max_len = predMaxLen;
    } while(P[maxPos] != NONE);
}  /* End of backtrack */

/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(int* matrix) {
    long long int i, j, ind;
    printf(" \t \t");
    for(i=0; i<m-1; i++)
        printf("%c\t",a[i]);
    printf("\n");
    for (i = 0; i < n; i++) { //Lines
        for (j = -1; j < m; j++) {
            if(i+j<n)
                ind = (i+j)*(i+j+1)/2 + i;
            else if(i+j<m)
                ind = (n+1)*(n)/2 + (i+j-n)*n + i;
            else
                ind = (i*j) + ((m-j)*(i+(i-(m-j-1))))/2 + ((n-i)*(j+(j-(n-i-1))))/2 + (m-j-1);
            if(i+j<0)
                printf(" \t");
            else if(j==-1 && i>0)
                printf("%c\t",b[i-1]);
            else
                printf("%d\t", matrix[ind]);
        }
        printf("\n");
    }
}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
void printPredecessorMatrix(int* matrix) {
    long long int i, j, ind;
    printf("    ");
    for(i=0; i<m-1; i++)
        printf("%c ",a[i]);
    printf("\n");
    for (i = 0; i < n; i++) { //Lines
        for (j = -1; j < m; j++) {
            if(i+j<n)
                ind = (i+j)*(i+j+1)/2 + i;
            else if(i+j<m)
                ind = (n+1)*(n)/2 + (i+j-n)*n + i;
            else
                ind = (i*j) + ((m-j)*(i+(i-(m-j-1))))/2 + ((n-i)*(j+(j-(n-i-1))))/2 + (m-j-1);
            if(i+j<0)
                printf("  ");
            else if(j==-1 && i>0)
                printf("%c ",b[i-1]);
            else{
                if(matrix[ind] < 0) {
                    printf(BOLDRED);
                    if (matrix[ind] == -UP)
                        printf("↑ ");
                    else if (matrix[ind] == -LEFT)
                        printf("← ");
                    else if (matrix[ind] == -DIAGONAL)
                        printf("↖ ");
                    else
                        printf("- ");
                    printf(RESET);
                } else {
                    if (matrix[ind] == UP)
                        printf("↑ ");
                    else if (matrix[ind] == LEFT)
                        printf("← ");
                    else if (matrix[ind] == DIAGONAL)
                        printf("↖ ");
                    else
                        printf("- ");
                }
            }
        }
        printf("\n");
    }

}  /* End of printPredecessorMatrix */


/*--------------------------------------------------------------------
 * Function:    generate
 * Purpose:     Generate arrays a and b
 */
void generate(){
    //Generates the values of a
    long long int i;
    for(i=0;i<m;i++){
        int aux=rand()%4;
        if(aux==0)
            a[i]='A';
        else if(aux==2)
            a[i]='C';
        else if(aux==3)
            a[i]='G';
        else
            a[i]='T';
    }

    //Generates the values of b
    for(i=0;i<n;i++){
        int aux=rand()%4;
        if(aux==0)
            b[i]='A';
        else if(aux==2)
            b[i]='C';
        else if(aux==3)
            b[i]='G';
        else
            b[i]='T';
    }
} /* End of generate */


/*--------------------------------------------------------------------
 * External References:
 * http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
 * http://pt.slideshare.net/avrilcoghlan/the-smith-waterman-algorithm
 * http://baba.sourceforge.net/
 */