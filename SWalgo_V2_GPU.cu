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
typedef struct  {
    long long int i;
    long long int max_len;
    long long int ind;
    long long int ind_u;
    long long int ind_d;
    long long int ind_l;
    long long int m;
    long long int j_start;
} similarity_parameters;


__global__ void smilarity(similarity_parameters *sp_gpu, char* a_gpu, char* b_gpu,
                          int* H, int* P, long long int *maxPos, long long int *maxPos_max_len) {
    long long int j_start = sp_gpu -> j_start;
    long long int j       = blockIdx.x * blockDim.x +threadIdx.x + j_start;
    long long int ind     = sp_gpu -> ind + j;
    long long int ind_l   = sp_gpu -> ind_l+j;
    long long int ind_d   = sp_gpu -> ind_d+j;
    long long int ind_u   = sp_gpu -> ind_u+j;
    long long int i       = sp_gpu -> i;
    long long int m       = sp_gpu -> m;
    long long int max_len = sp_gpu -> max_len;

    //Columns long long int ind
    long long int jj;
    long long int ii;
    if (i<m){
        ii = i-j;
        jj = j;
    }
    else{
        ii = m-1-j;
        jj = i-m+j+1;
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
}


int main(int argc, char* argv[]) {

    similarity_parameters *sp_gpu;
    similarity_parameters *sp_cpu;


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

    //allocate host
    sp_cpu= (similarity_parameters *)(malloc(sizeof(similarity_parameters)));
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


//
//    CUDA_SAFE_CALL(cudaMalloc((void **)&i_gpu, size));
//    CUDA_SAFE_CALL(cudaMalloc((void **)&j_gpu, size));
//    CUDA_SAFE_CALL(cudaMemcpy(i_gpu, &i, size, cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(j_gpu, &j, size, cudaMemcpyHostToDevice));

//    CUDA_SAFE_CALL(cudaMalloc((void **)&ind_gpu, size));
//    CUDA_SAFE_CALL(cudaMalloc((void **)&ind_d_gpu, size));
//    CUDA_SAFE_CALL(cudaMalloc((void **)&ind_u_gpu, size));
//    CUDA_SAFE_CALL(cudaMalloc((void **)&ind_l_gpu, size));
//    CUDA_SAFE_CALL(cudaMemcpy(ind_gpu, &ind, size, cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(ind_d_gpu, &ind_d, size, cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(ind_u_gpu, &ind_u, size, cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(ind_l_gpu, &ind_l, size, cudaMemcpyHostToDevice));
//
//    CUDA_SAFE_CALL(cudaMalloc((void **)&max_len_gpu, size));
//    CUDA_SAFE_CALL(cudaMemcpy(max_len_gpu, &max_len, size, cudaMemcpyHostToDevice));
//
//    CUDA_SAFE_CALL(cudaMalloc((void **)&m_gpu, size));
//    CUDA_SAFE_CALL(cudaMemcpy(m_gpu, &m, size, cudaMemcpyHostToDevice));
//
//    CUDA_SAFE_CALL(cudaMalloc((void **)&offset_gpu, size));
//    CUDA_SAFE_CALL(cudaMemcpy(offset_gpu, &offset, size, cudaMemcpyHostToDevice));
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

    long long int j_start, j_end;
    long long int ind = 3;
    long long int ind_u, ind_d, ind_l;
    CUDA_SAFE_CALL(cudaMalloc((void **)&sp_gpu, sizeof(similarity_parameters)));
    for (sp_cpu->i = 2; sp_cpu->i < m + n - 1; sp_cpu->i++) { //Lines
        long long int max_len;
        if (sp_cpu->i < n){
            max_len = sp_cpu->i + 1;
            j_start = 1;
            j_end   = max_len-1;
            ind_u   = ind - max_len;
            ind_l   = ind - max_len + 1;
            ind_d   = ind - (max_len<<1) + 2;
        }
        else if (sp_cpu->i >= sp_cpu->m){
            max_len = m + n - 1 - sp_cpu->i;
            j_start = 0;
            j_end   = max_len;
            ind_u   = ind - max_len - 1;
            ind_l   = ind - max_len;
            ind_d   = ind - (max_len<<1) - 2;
        }
        else{
            max_len   = n;
            j_start = 1;
            j_end   = max_len;
            ind_u     = ind - max_len - 1;
            ind_l     = ind - max_len;
            if(sp_cpu->i>n)
                ind_d = ind - (max_len<<1) - 1;
            else
                ind_d = ind - (max_len<<1);
        }

        long long int row_len = j_end - j_start;
        long long int offset = block_size * ((long long int) (row_len/block_size));

        sp_cpu -> j_start = j_start;
        sp_cpu -> max_len = max_len;
        sp_cpu -> ind = ind;
        sp_cpu -> ind_u = ind_u;
        sp_cpu -> ind_d = ind_d;
        sp_cpu -> ind_l = ind_l;
        sp_cpu -> m = m;


        if(row_len > block_size) {
            CUDA_SAFE_CALL(cudaMemcpy(sp_gpu, sp_cpu, sizeof(similarity_parameters), cudaMemcpyHostToDevice));
            dim3 block_Dim(block_size, 1, 1);
            dim3 grid_Dim((int) (row_len / block_size), 1, 1);
            smilarity<<<grid_Dim, block_Dim>>>(sp_gpu, a_gpu, b_gpu, H_gpu, P_gpu, maxPos_gpu, maxPos_max_len_gpu);
        }

        dim3 block_Dim_corner(row_len % block_size, 1, 1);
        dim3 grid_Dim_corner(1, 1, 1);
        sp_cpu->ind   = sp_cpu->ind   + offset;
        sp_cpu->ind_d = sp_cpu->ind_d + offset;
        sp_cpu->ind_l = sp_cpu->ind_l + offset;
        sp_cpu->ind_u = sp_cpu->ind_u + offset;
        CUDA_SAFE_CALL(cudaMemcpy(sp_gpu, sp_cpu, sizeof(similarity_parameters), cudaMemcpyHostToDevice));
        smilarity<<<grid_Dim_corner, block_Dim_corner>>>(sp_gpu, a_gpu, b_gpu, H_gpu, P_gpu, maxPos_gpu, maxPos_max_len_gpu);

        sp_cpu->ind += sp_cpu->max_len;
    }

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    //copy the memory from gpu to host
    CUDA_SAFE_CALL(cudaMemcpy(H, H_gpu, allocSize_matrix, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(P, P_gpu, allocSize_matrix, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&maxPos, maxPos_gpu, size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&maxPos_max_len, maxPos_max_len_gpu, size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(a, a_gpu, allocSize_a, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(b, b_gpu, allocSize_b, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(sp_cpu, sp_gpu, sizeof(similarity_parameters), cudaMemcpyDeviceToHost));


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
    CUDA_SAFE_CALL(cudaFree(sp_gpu));

    free(a);
    free(b);
    free(H);
    free(P);
    free(sp_cpu);

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