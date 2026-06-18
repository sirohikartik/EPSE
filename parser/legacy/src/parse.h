#ifndef PARSE_H
#define PARSE_H

#include <stdint.h>
#include <stddef.h>

#define DIM 480

typedef struct {
    char *data;
    size_t capacity;
    size_t length;
} sequence;

typedef struct {
    uint32_t id;
    char accession[32];
    char entry_name[32];
    char description[256];
    char organism[256];
    sequence *seq;
} Entry;

typedef struct {
    uint32_t centroid_id;
    uint32_t count;
    float centroid[DIM];
} Centroid;

typedef struct {
    uint32_t protein_id;
    float embedding[DIM];
} ProteinEmbedding;

typedef struct {
    uint32_t protein_id;
    uint32_t cluster_id;
} Assignment;

/* sequence helpers */
void sequence_init(sequence *s);
void sequence_append(sequence *s, const char *text);
void sequence_clear(sequence *s);
void sequence_free(sequence *s);

/* clustering helpers */
void update_centroid(Centroid *c, float *embedding);
float cosine(Centroid *c, float *embedding);

/* pipeline */
void parse_and_save(char *filepath, float threshold, char *data_dir);

#endif
