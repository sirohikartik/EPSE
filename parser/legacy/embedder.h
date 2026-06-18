#ifndef EMBEDDER_H
#define EMBEDDER_H

#ifdef __cplusplus
extern "C" {
#endif

int load_model(const char *model_path);

void get_embedding(const char *sequence, float *embedding_out);

void close_model();

#ifdef __cplusplus
}
#endif

#endif
