#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "embedder.h"
#include <math.h>
#include <float.h>

#define SIMILARITY_THRESHOLD 0.90f // Adjust this threshold as needed
#define DIM 480

typedef struct ProteinEmbedding {
    uint32_t protein_id;
    float embedding[DIM];
} ProteinEmbedding;

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
    float centroid[480];
} Centroid;

typedef struct {
    uint32_t protein_id;
    uint32_t cluster_id;
} Assignment;

// Utilities for sequence dynamic string container
void sequence_init(sequence *s) {
    s->capacity = 1024;
    s->length = 0;
    s->data = malloc(s->capacity);
    if (s->data == NULL) {
        printf("Malloc failed\n");
        exit(1);
    }
    s->data[0] = '\0';
}

void sequence_append(sequence *s, const char *text) {
    size_t text_len = strlen(text);
    while (s->length + text_len + 1 > s->capacity) {
        s->capacity *= 2;
        char *tmp = realloc(s->data, s->capacity);
        if (tmp == NULL) {
            printf("Realloc failed\n");
            exit(1);
        }
        s->data = tmp;
    }
    memcpy(s->data + s->length, text, text_len);
    s->length += text_len;
    s->data[s->length] = '\0';
}

void sequence_clear(sequence *s) {
    s->length = 0;
    s->data[0] = '\0';
}

void sequence_free(sequence *s) {
    free(s->data);

    s->data = NULL;
    s->length = 0;
    s->capacity = 0;
}

// update centroid function - updates a centroid's average embedding 
void update_centroid(Centroid *c, float* embedding){
    c->count++;
    for(int i =0;i<480;i++){
            c->centroid[i] += (embedding[i] - c->centroid[i])/c->count;
    }
}

// cosing similarity score function
float cosine(Centroid *c, float* embedding) {
    float dot_product = 0.0f;
    float norm_c = 0.0f;
    float norm_e = 0.0f;

    for(int i = 0; i < DIM; i++) {
        dot_product += c->centroid[i] * embedding[i];
        norm_c += c->centroid[i] * c->centroid[i];
        norm_e += embedding[i] * embedding[i];
    }

    // Prevent division by zero if either vector is entirely zeros
    if (norm_c == 0.0f || norm_e == 0.0f) {
        return 0.0f; 
    }

    // Return the normalized cosine similarity
    return dot_product / (sqrtf(norm_c) * sqrtf(norm_e));
}
// main function starts here -



void parse_and_save(char *filepath) {
    if (filepath == NULL) {
        printf("\033[31;1;4mERROR: NULL filepath \033[0m\n");
        return;
    }
    
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL) {
        printf("\033[31;1;4mERROR: %s could not be opened \033[0m\n", filepath);
        return;
    }

    // Output files
    FILE *csv = fopen(".data/proteins.csv", "w");
    FILE *emb_fp = fopen(".data/embeddings.bin", "wb");
    FILE *assign_fp = fopen(".data/assignments.bin", "wb"); 
    FILE *cent_csv = fopen(".data/centroids.csv", "w"); // NEW: Centroids file

    if (!csv || !emb_fp || !assign_fp || !cent_csv) {
        printf("\033[31;1;4mERROR: Failed to open output files\033[0m\n");
        if(csv) fclose(csv);
        if(emb_fp) fclose(emb_fp);
        if(assign_fp) fclose(assign_fp);
        if(cent_csv) fclose(cent_csv);
        if(fp) fclose(fp);
        return;
    }

    // Write headers
    fprintf(csv, "id,accession,entry_name,description,organism,seq_len\n");
    
    fprintf(cent_csv, "centroid_id,count");
    for(int i = 0; i < DIM; i++) {
        fprintf(cent_csv, ",e%d", i);
    }
    fprintf(cent_csv, "\n");

    char BUFFER[4096];

    Entry *entry = malloc(sizeof(Entry));
    ProteinEmbedding *embedding = malloc(sizeof(ProteinEmbedding));
    sequence *s = malloc(sizeof(sequence));
    
    // Dynamic array for our existing centroids
    size_t centroid_capacity = 1000;
    size_t num_centroids = 0;
    Centroid *centroids = malloc(centroid_capacity * sizeof(Centroid));

    if (!entry || !embedding || !s || !centroids) {
        printf("\033[31;1;4mMalloc failed\033[0m\n");
        if(csv) fclose(csv);
        if(emb_fp) fclose(emb_fp);
        if(assign_fp) fclose(assign_fp);
        if(cent_csv) fclose(cent_csv);
        if(fp) fclose(fp);
        if(entry) free(entry);
        if(embedding) free(embedding);
        if(s) free(s);
        if(centroids) free(centroids);
        return;
    }

    uint32_t id = 0;
    int first_protein = 1;

    int loaded = load_model("model/esm2_t12_35M.pt");
    if(loaded == -1) {
            printf("Model could not be loaded");
            if(csv) fclose(csv);
        if(emb_fp) fclose(emb_fp);
        if(assign_fp) fclose(assign_fp);
        if(cent_csv) fclose(cent_csv);
        if(fp) fclose(fp);
        if(entry) free(entry);
        if(embedding) free(embedding);
        if(s) free(s);
        if(centroids) free(centroids);
        return;



    }
    sequence_init(s);

    while (fgets(BUFFER, sizeof(BUFFER), fp) != NULL) {
        if (BUFFER[0] == '>') {

            if (!first_protein) {
                entry->seq = s;

                // 1. Get Embedding
                get_embedding(s->data, embedding->embedding);
                embedding->protein_id = entry->id;
                
                // 2. Save to embeddings.bin
                fwrite(embedding, sizeof(ProteinEmbedding), 1, emb_fp);

                // 3. Compute similarity to existing centroids
                float best_score = -2.0f; // Cosine ranges from -1 to 1
                int best_centroid_idx = -1;

                for (size_t c = 0; c < num_centroids; c++) {
                    float score = cosine(&centroids[c], embedding->embedding);
                    if (score > best_score) {
                        best_score = score;
                        best_centroid_idx = c;
                    }
                }

                Assignment assign;
                assign.protein_id = entry->id;

                // 4 & 5. Find best centroid and check threshold
                if (best_centroid_idx != -1 && best_score >= SIMILARITY_THRESHOLD) {
                    // YES: Assign to existing centroid & update running average
                    assign.cluster_id = centroids[best_centroid_idx].centroid_id;
                    update_centroid(&centroids[best_centroid_idx], embedding->embedding);
                } else {
                    // NO: Create new centroid
                    if (num_centroids >= centroid_capacity) {
                        centroid_capacity *= 2;
                        centroids = realloc(centroids, centroid_capacity * sizeof(Centroid));
                    }
                    
                    Centroid new_c;
                    new_c.centroid_id = num_centroids; 
                    new_c.count = 1;
                    memcpy(new_c.centroid, embedding->embedding, DIM * sizeof(float));
                    
                    centroids[num_centroids] = new_c;
                    assign.cluster_id = new_c.centroid_id;
                    
                    num_centroids++;
                }

                // 6. Save assignment record
                fwrite(&assign, sizeof(Assignment), 1, assign_fp);

                // Save metadata to CSV
                fprintf(csv, "%u,\"%s\",\"%s\",\"%s\",\"%s\",%zu\n", entry->id, entry->accession,
                        entry->entry_name, entry->description, entry->organism, s->length);

                printf("Saved protein %u (%s) -> Cluster %u\n", entry->id, entry->accession, assign.cluster_id);

                sequence_clear(s);
            }

            first_protein = 0;
            id++;
            entry->id = id;

            // --- Header parsing logic ---
            int ptr = 4;
            char accession[32];
            int i = 0;
            while (BUFFER[ptr] != '|' && BUFFER[ptr] != '\0') {
                accession[i++] = BUFFER[ptr++];
            }
            accession[i] = '\0';

            ptr++; 

            char entry_name[32];
            i = 0;
            while (BUFFER[ptr] != ' ' && BUFFER[ptr] != '\0') {
                entry_name[i++] = BUFFER[ptr++];
            }
            entry_name[i] = '\0';

            ptr++; 

            char descp[256];
            i = 0;
            while (!(BUFFER[ptr] == 'O' && BUFFER[ptr + 1] == 'S' && BUFFER[ptr + 2] == '=') && BUFFER[ptr] != '\0') {
                descp[i++] = BUFFER[ptr++];
            }
            descp[i] = '\0';
            
            ptr += 3; 
            char org[256];
            i = 0;
            while (!(BUFFER[ptr] == 'O' && BUFFER[ptr + 1] == 'X' && BUFFER[ptr + 2] == '=') && BUFFER[ptr] != '\0' && BUFFER[ptr] != '\n') {
                org[i++] = BUFFER[ptr++];
            }
            org[i] = '\0';

            strcpy(entry->accession, accession);
            strcpy(entry->entry_name, entry_name);
            strcpy(entry->description, descp);
            strcpy(entry->organism, org);

        } else {
            BUFFER[strcspn(BUFFER, "\n")] = '\0';
            sequence_append(s, BUFFER);
        }
    }

    // Handle the final protein
    if (!first_protein) {
        get_embedding(s->data, embedding->embedding);
        embedding->protein_id = entry->id;
        fwrite(embedding, sizeof(ProteinEmbedding), 1, emb_fp);

        float best_score = -2.0f;
        int best_centroid_idx = -1;
        for (size_t c = 0; c < num_centroids; c++) {
            float score = cosine(&centroids[c], embedding->embedding);
            if (score > best_score) {
                best_score = score;
                best_centroid_idx = c;
            }
        }

        Assignment assign;
        assign.protein_id = entry->id;

        if (best_centroid_idx != -1 && best_score >= SIMILARITY_THRESHOLD) {
            assign.cluster_id = centroids[best_centroid_idx].centroid_id;
            update_centroid(&centroids[best_centroid_idx], embedding->embedding);
        } else {
            if (num_centroids >= centroid_capacity) {
                centroid_capacity *= 2;
                centroids = realloc(centroids, centroid_capacity * sizeof(Centroid));
            }
            Centroid new_c;
            new_c.centroid_id = num_centroids;
            new_c.count = 1;
            memcpy(new_c.centroid, embedding->embedding, DIM * sizeof(float));
            centroids[num_centroids] = new_c;
            assign.cluster_id = new_c.centroid_id;
            num_centroids++;
        }

        fwrite(&assign, sizeof(Assignment), 1, assign_fp);
        fprintf(csv, "%u,\"%s\",\"%s\",\"%s\",\"%s\",%zu\n", entry->id, entry->accession,
                entry->entry_name, entry->description, entry->organism, s->length);
        
        printf("Saved protein %u (%s) -> Cluster %u\n", entry->id, entry->accession, assign.cluster_id);
    }

    // NEW: Dump the final centroids array to centroids.csv
    for (size_t c = 0; c < num_centroids; c++) {
        fprintf(cent_csv, "%u,%u", centroids[c].centroid_id, centroids[c].count);
        for (int i = 0; i < DIM; i++) {
            fprintf(cent_csv, ",%f", centroids[c].centroid[i]);
        }
        fprintf(cent_csv, "\n");
    }

    // Cleanup
    sequence_free(s);
    free(s);
    free(entry);
    free(embedding);
    free(centroids); 

    fclose(csv);
    fclose(emb_fp);
    fclose(assign_fp);
    fclose(cent_csv);
    fclose(fp);
}

int main() {
    mkdir(".data", 0755);
    parse_and_save("uniprot_sprot.fasta");
}
