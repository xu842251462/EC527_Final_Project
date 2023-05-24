/***********************************************************************
 * Smith–Waterman algorithm
 * Purpose:     Local alignment of nucleotide or protein sequences
 * Authors:     Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 ***********************************************************************/
//This is the AVX512 version.


// gcc -mavx512f SWalgo_V3.c -lgomp -o SWalgo_V3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <stdbool.h>
#include <sys/stat.h>

//#include <zmmintrin.h>

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

#define Vsize 16
#define NumOfTest 1e2//1e4

//#define DEBUG
//#define pragmas
/* End of constants */


/*--------------------------------------------------------------------
 * Functions Prototypes
 */
void similarityScore(long long int ind, long long int ind_u, long long int ind_d, long long int ind_l, long long int ii, long long int jj, int* H, int* P, long long int max_len, long long int* maxPos, long long int *maxPos_max_len);
void similarityScoreIntrinsic(__m512i* H, __m512i* Hu, __m512i* Hd, __m512i* Hl, __m512i* P, __m512i ii, __m512i jj, int* H_main, long long int ind, long long int max_len, long long int* maxPos, long long int *maxPos_max_len);
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
int main(int argc, char* argv[]) {
    
    
    if(argc>1){
        m = strtoll(argv[1], NULL, 10);
        n = strtoll(argv[2], NULL, 10); 
        long long int temp;
        if( m<n){
            temp = m;
            m = n;
            n = temp;
        }
    }
    else{
        m = 22;
        n = 20;
    }


    #ifdef DEBUG
    printf("\nMatrix[%lld][%lld]\n", n, m);
    #endif

    //Allocates a and b
    a = malloc(m * sizeof(char));
    b = malloc(n * sizeof(char));
    //Because now we have zeros
    m++;
    n++;
    
    //Allocates similarity matrix H
    int *H;
    H = calloc(m * n, sizeof(int));

    //Allocates predecessor matrix P
    int *P;
    P = calloc(m * n, sizeof(int));


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



    //Start position for backtrack
    long long int maxPos         = 0;
    long long int maxPos_max_len = 0;

    //Calculates the similarity matrix
    long long int i, j;

    double initialTime = omp_get_wtime();
    #ifdef pragmas
    #pragma GCC ivdep
    #endif


    #ifdef DEBUG
    printf("\n a string:\n");
    for(i=0; i<m-1; i++)
        printf("%c ",a[i]);
    printf("\n b string:\n");
    for(i=0; i<n-1; i++)
        printf("%c ",b[i]);
    printf("\n");
    #endif

    int it;
    for(it=0; it<NumOfTest; it++){

    long long int ind   = 3;
    long long int indd  = 0;
    long long int indul = 1;
    long long int ind_u, ind_d, ind_l; 
    __m512i offset = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (i = 2; i < m+n-1; i++) { //Lines
        long long int max_len;
        long long int ii,jj;
        long long int j_start, j_end;
        if (i<n){
		    max_len = i+1;
            j_start = 1;
            j_end   = max_len-1;
            ind_u   = ind - max_len;
            ind_l   = ind - max_len + 1;
            ind_d   = ind - (max_len<<1) + 2;
        }
	    else if (i>=m){
		    max_len = m+n-1-i;
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
            if(i>n)
                ind_d = ind - (max_len<<1) - 1;
            else
                ind_d = ind - (max_len<<1);
        }  
        __m512i* Hu = (__m512i*) (H+ind_u+j_start);
        __m512i* Hl = (__m512i*) (H+ind_l+j_start);
        __m512i* Hd = (__m512i*) (H+ind_d+j_start);
        __m512i* HH = (__m512i*) (H+ind+j_start);
        __m512i* PP = (__m512i*) (P+ind+j_start);

        //int Vsize = 512/sizeof(typeof(H));
      //  #pragma gcc ivdep 
        for (j = j_start; j <j_end-Vsize+1; j+=Vsize) { //Columns   

            __m512i Joffset = _mm512_add_epi32(offset, _mm512_set1_epi32(j));
            __m512i Ioffset = _mm512_set1_epi32(i);
            __mmask16 mask  = _mm512_cmpgt_epi32_mask(Ioffset,_mm512_set1_epi32(m));
            __m512i I_J     = _mm512_sub_epi32(Ioffset, Joffset);
            __m512i M_1_J   = _mm512_sub_epi32(_mm512_set1_epi32(m-1),Joffset);
            __m512i II      = _mm512_mask_mov_epi32(I_J, mask, M_1_J);
            __m512i IJ      = _mm512_add_epi32(Joffset, Ioffset);
            __m512i I_MJ1   = _mm512_sub_epi32(IJ,_mm512_set1_epi32(m-1));
            __m512i JJ      = _mm512_mask_mov_epi32(Joffset, mask, I_MJ1);
            similarityScoreIntrinsic(HH, Hu, Hd, Hl, PP, II, JJ, H, ind, max_len, &maxPos, &maxPos_max_len);
            Hu++;
           Hl++;
           Hd++;
           HH++;
           PP++;
        }
        
         for(;j<j_end; j++){
            if (i<m){
                ii = i-j;
                jj = j;
            }
            else{
                ii = m-1-j;
                jj = i-m+j+1;
            }      
            similarityScore(ind+j, ind_u+j, ind_d+j, ind_l+j, ii, jj, H, P, max_len, &maxPos, &maxPos_max_len);
        }
        ind += max_len;
    }
    

    backtrack(P, maxPos, maxPos_max_len);

    }
    //Gets final time
    double finalTime = omp_get_wtime();
    printf("\nElapsed time: %f\n\n", (finalTime - initialTime)/NumOfTest);

    FILE *fp;
    fp = fopen("Results.txt", "a");
    fprintf(fp, "Elapsed time V3: %lf\n", (finalTime - initialTime)/NumOfTest);
    fclose(fp);

    #ifdef DEBUG
    printf("\nSimilarity Matrix:\n");
    printMatrix(H);

    printf("\nPredecessor Matrix:\n");
    printPredecessorMatrix(P);
    #endif

    //Frees similarity matrixes
    free(H);
    free(P);

    //Frees input arrays
    free(a);
    free(b);

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


void similarityScoreIntrinsic(__m512i* HH, __m512i* Hu, __m512i* Hd, __m512i* Hl, __m512i* PP, __m512i ii, __m512i jj, int* H, long long int ind, long long int max_len, long long int* maxPos, long long int *maxPos_max_len) {

    __m512i up, left, diag;

    __m512i HHu = _mm512_loadu_si512(Hu);
    __m512i HHd = _mm512_loadu_si512(Hd);
    __m512i HHl = _mm512_loadu_si512(Hl);

    //Get element above
    up                     = _mm512_add_epi32(HHu, _mm512_set1_epi32(gapScore));

    //Get element on the left
    left                   = _mm512_add_epi32(HHl, _mm512_set1_epi32(gapScore));

    //Get element on the diagonal
    __m512i A              = _mm512_i32gather_epi32(_mm512_sub_epi32(ii,_mm512_set1_epi32(1)), (void*)a, sizeof(char));
    __m512i B              = _mm512_i32gather_epi32(_mm512_sub_epi32(jj,_mm512_set1_epi32(1)), (void*)b, sizeof(char));
    A                      = _mm512_slli_epi32(A,24);
    B                      = _mm512_slli_epi32(B,24);
    __mmask16 mask         = _mm512_cmpeq_epi32_mask(A,B);
    __m512i MATCHSCORE     = _mm512_set1_epi32(matchScore);
    __m512i MISSMATCHSCORE = _mm512_set1_epi32(missmatchScore);
    __m512i MATCHMISS      = _mm512_mask_mov_epi32(MISSMATCHSCORE, mask, MATCHSCORE);
    diag                   = _mm512_add_epi32(HHd, MATCHMISS);


    //Calculates the maximum
    __m512i max  = _mm512_set1_epi32(NONE);
    __m512i pred = _mm512_set1_epi32(NONE);

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
   //same letter ↖
    mask = _mm512_cmpgt_epi32_mask(max, diag);
    max  = _mm512_mask_mov_epi32(diag, mask, max);
    pred = _mm512_mask_mov_epi32(_mm512_set1_epi32(DIAGONAL), mask, pred);

    //remove letter ↑ 
    mask = _mm512_cmpgt_epi32_mask(max, up);
    max  = _mm512_mask_mov_epi32(up, mask, max);
    pred = _mm512_mask_mov_epi32(_mm512_set1_epi32(UP), mask, pred);
   

    //insert letter ←
    mask = _mm512_cmpgt_epi32_mask(max, left);
    max  = _mm512_mask_mov_epi32(left, mask, max);
    pred = _mm512_mask_mov_epi32(_mm512_set1_epi32(LEFT), mask, pred);
    
    //Inserts the value in the similarity and predecessor matrixes

    _mm512_storeu_si512(HH, max);
    _mm512_storeu_si512(PP, pred);

    //Updates maximum score to be used as seed on backtrack 
    int maxx       = _mm512_reduce_max_epi32(max);
    mask           = _mm512_cmpeq_epi32_mask(max, _mm512_set1_epi32(maxx));
    __m512i offset = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    int maxx_ind   = _mm512_mask_reduce_max_epi32(mask, offset);
    if (maxx > H[*maxPos]) {
        *maxPos         = ind+maxx_ind;
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