#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../src/parse.h"

#define TEST_FILE "test_embeddings.txt"
#define TEST_FILE_EMPTY "test_empty.txt"
#define TEST_FILE_MALFORMED "test_malformed.txt"
#define TEST_FILE_SINGLE "test_single.txt"
#define TEST_FILE_LARGE "test_large.txt"
#define EPSILON 1e-6

/* Helper function to create test files */
static void create_test_file(const char *filename, const char *content) {
    FILE *f = fopen(filename, "w");
    assert(f != NULL);
    fprintf(f, "%s", content);
    fclose(f);
}

static void cleanup_test_files(void) {
    remove(TEST_FILE);
    remove(TEST_FILE_EMPTY);
    remove(TEST_FILE_MALFORMED);
    remove(TEST_FILE_SINGLE);
    remove(TEST_FILE_LARGE);
}

/* Test 1: Normal parsing with valid embeddings */
static void test_normal_parsing(void) {
    printf("Test 1: Normal parsing with valid embeddings...\n");
    
    const char *content = 
        "0.1 0.2 0.3\n"
        "0.4 0.5 0.6\n"
        "0.7 0.8 0.9\n";
    
    create_test_file(TEST_FILE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE, &embeddings, &count, &dim);
    assert(result == 0);
    assert(count == 3);
    assert(dim == 3);
    assert(embeddings != NULL);
    
    /* Verify first embedding */
    assert(fabs(embeddings[0].values[0] - 0.1) < EPSILON);
    assert(fabs(embeddings[0].values[1] - 0.2) < EPSILON);
    assert(fabs(embeddings[0].values[2] - 0.3) < EPSILON);
    
    /* Verify second embedding */
    assert(fabs(embeddings[1].values[0] - 0.4) < EPSILON);
    assert(fabs(embeddings[1].values[1] - 0.5) < EPSILON);
    assert(fabs(embeddings[1].values[2] - 0.6) < EPSILON);
    
    /* Verify third embedding */
    assert(fabs(embeddings[2].values[0] - 0.7) < EPSILON);
    assert(fabs(embeddings[2].values[1] - 0.8) < EPSILON);
    assert(fabs(embeddings[2].values[2] - 0.9) < EPSILON);
    
    free_embeddings(embeddings, count);
    printf("PASSED\n");
}

/* Test 2: Empty file */
static void test_empty_file(void) {
    printf("Test 2: Empty file...\n");
    
    create_test_file(TEST_FILE_EMPTY, "");
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE_EMPTY, &embeddings, &count, &dim);
    assert(result == -1);  /* Should return error for empty file */
    assert(embeddings == NULL);
    assert(count == 0);
    assert(dim == 0);
    
    printf("PASSED\n");
}

/* Test 3: Malformed line (non-numeric values) */
static void test_malformed_line(void) {
    printf("Test 3: Malformed line with non-numeric values...\n");
    
    const char *content = 
        "0.1 0.2 abc\n"
        "0.4 0.5 0.6\n";
    
    create_test_file(TEST_FILE_MALFORMED, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE_MALFORMED, &embeddings, &count, &dim);
    assert(result == -1);  /* Should return error for malformed line */
    assert(embeddings == NULL);
    
    printf("PASSED\n");
}

/* Test 4: Single embedding file */
static void test_single_embedding(void) {
    printf("Test 4: Single embedding file...\n");
    
    const char *content = "1.0 2.0 3.0\n";
    
    create_test_file(TEST_FILE_SINGLE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE_SINGLE, &embeddings, &count, &dim);
    assert(result == 0);
    assert(count == 1);
    assert(dim == 3);
    assert(embeddings != NULL);
    
    assert(fabs(embeddings[0].values[0] - 1.0) < EPSILON);
    assert(fabs(embeddings[0].values[1] - 2.0) < EPSILON);
    assert(fabs(embeddings[0].values[2] - 3.0) < EPSILON);
    
    free_embeddings(embeddings, count);
    printf("PASSED\n");
}

/* Test 5: Inconsistent dimensions */
static void test_inconsistent_dimensions(void) {
    printf("Test 5: Inconsistent dimensions across lines...\n");
    
    const char *content = 
        "0.1 0.2 0.3\n"
        "0.4 0.5\n";  /* Missing one value */
    
    create_test_file(TEST_FILE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE, &embeddings, &count, &dim);
    assert(result == -1);  /* Should return error for inconsistent dimensions */
    assert(embeddings == NULL);
    
    printf("PASSED\n");
}

/* Test 6: Centroid calculation */
static void test_centroid_calculation(void) {
    printf("Test 6: Centroid calculation...\n");
    
    const char *content = 
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
        "7.0 8.0 9.0\n";
    
    create_test_file(TEST_FILE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE, &embeddings, &count, &dim);
    assert(result == 0);
    
    double *centroid = compute_centroid(embeddings, count, dim);
    assert(centroid != NULL);
    
    /* Expected centroid: (4.0, 5.0, 6.0) */
    assert(fabs(centroid[0] - 4.0) < EPSILON);
    assert(fabs(centroid[1] - 5.0) < EPSILON);
    assert(fabs(centroid[2] - 6.0) < EPSILON);
    
    free(centroid);
    free_embeddings(embeddings, count);
    printf("PASSED\n");
}

/* Test 7: Centroid with single embedding */
static void test_centroid_single(void) {
    printf("Test 7: Centroid with single embedding...\n");
    
    const char *content = "5.0 10.0 15.0\n";
    
    create_test_file(TEST_FILE_SINGLE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE_SINGLE, &embeddings, &count, &dim);
    assert(result == 0);
    
    double *centroid = compute_centroid(embeddings, count, dim);
    assert(centroid != NULL);
    
    /* Centroid should equal the single embedding */
    assert(fabs(centroid[0] - 5.0) < EPSILON);
    assert(fabs(centroid[1] - 10.0) < EPSILON);
    assert(fabs(centroid[2] - 15.0) < EPSILON);
    
    free(centroid);
    free_embeddings(embeddings, count);
    printf("PASSED\n");
}

/* Test 8: Large file with many embeddings */
static void test_large_file(void) {
    printf("Test 8: Large file with many embeddings...\n");
    
    FILE *f = fopen(TEST_FILE_LARGE, "w");
    assert(f != NULL);
    
    /* Create 1000 embeddings with 10 dimensions each */
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 10; j++) {
            fprintf(f, "%f ", (double)(i * j) / 1000.0);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE_LARGE, &embeddings, &count, &dim);
    assert(result == 0);
    assert(count == 1000);
    assert(dim == 10);
    assert(embeddings != NULL);
    
    /* Verify first and last embeddings */
    assert(fabs(embeddings[0].values[0] - 0.0) < EPSILON);
    assert(fabs(embeddings[999].values[9] - 8.991) < EPSILON);
    
    /* Test centroid calculation on large dataset */
    double *centroid = compute_centroid(embeddings, count, dim);
    assert(centroid != NULL);
    assert(centroid[0] >= 0.0);  /* Centroid should be non-negative */
    
    free(centroid);
    free_embeddings(embeddings, count);
    printf("PASSED\n");
}

/* Test 9: Invalid file path */
static void test_invalid_file(void) {
    printf("Test 9: Invalid file path...\n");
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings("/nonexistent/path/file.txt", &embeddings, &count, &dim);
    assert(result == -1);  /* Should return error for invalid path */
    assert(embeddings == NULL);
    
    printf("PASSED\n");
}

/* Test 10: File with only whitespace */
static void test_whitespace_only(void) {
    printf("Test 10: File with only whitespace...\n");
    
    const char *content = "   \n  \n  \n";
    
    create_test_file(TEST_FILE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE, &embeddings, &count, &dim);
    assert(result == -1);  /* Should return error for whitespace-only file */
    assert(embeddings == NULL);
    
    printf("PASSED\n");
}

/* Test 11: Negative values in embeddings */
static void test_negative_values(void) {
    printf("Test 11: Negative values in embeddings...\n");
    
    const char *content = 
        "-1.0 -2.0 -3.0\n"
        "-4.0 -5.0 -6.0\n";
    
    create_test_file(TEST_FILE, content);
    
    Embedding *embeddings = NULL;
    int count = 0;
    int dim = 0;
    
    int result = parse_embeddings(TEST_FILE, &embeddings, &count, &dim);
    assert(result == 0);
    assert(count == 2);
    assert(dim == 3);
    
    assert(fabs(embeddings[0].values[0] - (-1.0)) < EPSILON);
    assert(fabs(embeddings[1].values[2] - (-6.0)) < EPSILON);
    
    free_embeddings(embeddings, count);
    printf("PASSED\n");
}

/* Test 12: Free embeddings with NULL */
static void test_free_null(void) {
    printf("Test 12: Free embeddings with NULL...\n");
    
    /* Should not crash */
    free_embeddings(NULL, 0);
    free_embeddings(NULL, 10);
    
    printf("PASSED\n");
}

int main(void) {
    printf("Running parse.c test suite...\n\n");
    
    test_normal_parsing();
    test_empty_file();
    test_malformed_line();
    test_single_embedding();
    test_inconsistent_dimensions();
    test_centroid_calculation();
    test_centroid_single();
    test_large_file();
    test_invalid_file();
    test_whitespace_only();
    test_negative_values();
    test_free_null();
    
    cleanup_test_files();
    
    printf("\nAll tests passed!\n");
    return 0;
}