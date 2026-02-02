/*
 * axpy: Y += a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <pthread.h>

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float
#define VECTOR_LENGTH 102400

void init(REAL A[], int N) {
    for (int i = 0; i < N; i++) A[i] = (double) drand48();
}

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    for (int i = 0; i < N; ++i) Y[i] += a * X[i];
}

typedef struct {
    int tid;
    int num_threads;
    int N;
    REAL *Y;
    REAL *X;
    REAL a;
} axpy_task_t;

static void* axpy_worker(void *arg) {
    axpy_task_t *t = (axpy_task_t*)arg;

    // loop chunking / static worksharing
    int chunk = (t->N + t->num_threads - 1) / t->num_threads;
    int start = t->tid * chunk;
    int end   = start + chunk;
    if (end > t->N) end = t->N;

    for (int i = start; i < end; ++i) {
        t->Y[i] += t->a * t->X[i];
    }
    return NULL;
}

/**
 * pthread version using loop chunking and worksharing
 */
void axpy_kernel_threading(int N, REAL *Y, REAL *X, REAL a, int num_threads) {
    if (num_threads < 1) num_threads = 1;
    if (num_threads > N) num_threads = N;

    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)num_threads);
    axpy_task_t *tasks = (axpy_task_t*)malloc(sizeof(axpy_task_t) * (size_t)num_threads);
    if (!threads || !tasks) {
        fprintf(stderr, "malloc failed in axpy_kernel_threading\n");
        free(threads); free(tasks);
        exit(1);
    }

    for (int t = 0; t < num_threads; ++t) {
        tasks[t] = (axpy_task_t){ .tid=t, .num_threads=num_threads, .N=N, .Y=Y, .X=X, .a=a };
        int rc = pthread_create(&threads[t], NULL, axpy_worker, &tasks[t]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed (rc=%d)\n", rc);
            exit(1);
        }
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], NULL);

    free(threads);
    free(tasks);
}

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_threads = 4;

    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> [<num_threads>] (default n=%d, default threads=%d)\n", N, num_threads);
    } else if (argc == 2) {
        N = atoi(argv[1]);
    } else {
        N = atoi(argv[1]);
        num_threads = atoi(argv[2]);
    }

    REAL *X = (REAL*)malloc(sizeof(REAL)*N);
    REAL *Y = (REAL*)malloc(sizeof(REAL)*N);

    srand48((1 << 12));
    init(X, N);
    init(Y, N);

    int num_runs = 10;
    REAL a = 0.1234;

    double elapsed = read_timer();
    for (int i=0; i<num_runs; i++) axpy_kernel(N, Y, X, a);
    elapsed = (read_timer() - elapsed)/num_runs;

    double elapsed2 = read_timer();
    for (int i=0; i<num_runs; i++) axpy_kernel_threading(N, Y, X, a, num_threads);
    elapsed2 = (read_timer() - elapsed2)/num_runs;

    printf("======================================================================================================\n");
    printf("\tAXPY %d numbers, serial and threading\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("AXPY-serial:\t\t%4f\t%4f\n", elapsed * 1.0e3, 2*N / (1.0e6 * elapsed));
    printf("AXPY-%d threads:\t\t%4f\t%4f\n", num_threads, elapsed2 * 1.0e3, 2*N / (1.0e6 * elapsed2));
    return 0;
}

