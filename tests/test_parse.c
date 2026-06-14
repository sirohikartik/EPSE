#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../parser/parse.c"

#define EPSILON 1e-5

static void test_sequence_init(void) {
    sequence s;

    sequence_init(&s);

    assert(s.data != NULL);
    assert(s.length == 0);
    assert(s.capacity >= 1024);
    assert(strcmp(s.data, "") == 0);

    sequence_free(&s);

    printf("test_sequence_init PASSED\n");
}

static void test_sequence_append(void) {
    sequence s;

    sequence_init(&s);

    sequence_append(&s, "ABC");
    sequence_append(&s, "DEF");

    assert(strcmp(s.data, "ABCDEF") == 0);
    assert(s.length == 6);

    sequence_free(&s);

    printf("test_sequence_append PASSED\n");
}

static void test_sequence_clear(void) {
    sequence s;

    sequence_init(&s);

    sequence_append(&s, "HELLO");

    sequence_clear(&s);

    assert(s.length == 0);
    assert(strcmp(s.data, "") == 0);

    sequence_free(&s);

    printf("test_sequence_clear PASSED\n");
}

static void test_sequence_growth(void) {
    sequence s;

    sequence_init(&s);

    for (int i = 0; i < 5000; i++) {
        sequence_append(&s, "A");
    }

    assert(s.length == 5000);

    sequence_free(&s);

    printf("test_sequence_growth PASSED\n");
}

static void test_cosine_identical(void) {
    Centroid c;

    c.count = 1;

    for (int i = 0; i < DIM; i++) {
        c.centroid[i] = 1.0f;
    }

    float embedding[DIM];

    for (int i = 0; i < DIM; i++) {
        embedding[i] = 1.0f;
    }

    float score = cosine(&c, embedding);

    assert(fabsf(score - 1.0f) < EPSILON);

    printf("test_cosine_identical PASSED\n");
}

static void test_cosine_zero_vector(void) {
    Centroid c;

    c.count = 1;

    for (int i = 0; i < DIM; i++) {
        c.centroid[i] = 0.0f;
    }

    float embedding[DIM];

    for (int i = 0; i < DIM; i++) {
        embedding[i] = 1.0f;
    }

    float score = cosine(&c, embedding);

    assert(score == 0.0f);

    printf("test_cosine_zero_vector PASSED\n");
}

static void test_update_centroid(void) {
    Centroid c;

    c.count = 1;

    for (int i = 0; i < DIM; i++) {
        c.centroid[i] = 2.0f;
    }

    float embedding[DIM];

    for (int i = 0; i < DIM; i++) {
        embedding[i] = 4.0f;
    }

    update_centroid(&c, embedding);

    assert(c.count == 2);

    for (int i = 0; i < DIM; i++) {
        assert(fabsf(c.centroid[i] - 3.0f) < EPSILON);
    }

    printf("test_update_centroid PASSED\n");
}

int main(void) {
    printf("Running EPSE test suite...\n\n");

    test_sequence_init();
    test_sequence_append();
    test_sequence_clear();
    test_sequence_growth();

    test_cosine_identical();
    test_cosine_zero_vector();

    test_update_centroid();

    printf("\nAll tests passed!\n");

    return 0;
}
