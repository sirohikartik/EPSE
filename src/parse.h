#ifndef PARSE_H
#define PARSE_H

typedef struct {
    double *values;
    int dim;
} Embedding;

/**
 * Parse embeddings from a file.
 * 
 * @param filename Path to the input file
 * @param embeddings Output pointer to array of embeddings
 * @param count Output number of embeddings parsed
 * @param dim Output dimension of each embedding
 * @return 0 on success, -1 on failure
 */
int parse_embeddings(const char *filename, Embedding **embeddings, int *count, int *dim);

/**
 * Compute the centroid of a set of embeddings.
 * 
 * @param embeddings Array of embeddings
 * @param count Number of embeddings
 * @param dim Dimension of each embedding
 * @return Pointer to centroid array (caller must free), NULL on failure
 */
double *compute_centroid(Embedding *embeddings, int count, int dim);

/**
 * Free an array of embeddings.
 * 
 * @param embeddings Array of embeddings to free
 * @param count Number of embeddings
 */
void free_embeddings(Embedding *embeddings, int count);

#endif /* PARSE_H */