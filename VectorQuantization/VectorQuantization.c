//
//  main.c
//  Homework
//
//  Created by dikov on 24/12/15.
//  Copyright © 2015 dikov. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_INPUT_LENGHT 1001
#define true 1
#define false 0
typedef unsigned char bool;
#define FLOAT_MIN -0x1.fffffep+127f
#define FLOAT_MAX 0x1.fffffep+127f
#define DIMENSIONALITY 2

#define ASSERTIONS_ENABLED false
#define DATA_LOGGING false

#define SQR(x) ((x)*(x))

#define UNKNOWN_CLUSTER_COUNT true
#define MAX_CLUSTER_COUNT 100
#define LEARNING_RATE 0.001
#define SENSITIVITY_GAIN 0.001
#define SENSITIVITY_SIGNIFICANCE 0.5


//-----------------------------------------------//
//---------- CUSTOM TYPES -----------------------//
//-----------------------------------------------//

//----------- Vector structure ------------------//
typedef struct Vector{
    double* scalars;
    size_t length;
} Vector;

Vector* new_vector(size_t len){
    Vector* vec = malloc(sizeof(Vector));
    vec->length = len;
    vec->scalars = malloc(len*sizeof(double));
    for (int i = 0; i < len; i++) {
        vec->scalars[i] = 0.0;
    }
    return vec;
}

void print_vector(Vector* vec){
    printf("[");
    for (size_t i = 0; i < vec->length-1; i++) {
        printf("%lf, ", vec->scalars[i]);
    }
    printf("%lf]\n", vec->scalars[vec->length-1]);
}

void delete_vector(Vector* vec){
    if (vec->scalars != NULL) {
        free(vec->scalars);
    }
    free(vec);
}

// Data sample poitn structure
typedef struct Datapoint {
    Vector* position;
    int affiliation_id;
} Datapoint;

Datapoint* new_datapoint(size_t dimensionality){
    Datapoint* datapoint = malloc(sizeof(Datapoint));
    datapoint->position = new_vector(dimensionality);
    datapoint->affiliation_id = -1;
    return datapoint;
}

void print_datapoint(Datapoint* datapoint){
    printf("[");
    for (size_t i = 0; i < datapoint->position->length-1; i++) {
        printf("%lf, ", datapoint->position->scalars[i]);
    }
    printf("%lf], CentroidID: %d\n", datapoint->position->scalars[datapoint->position->length-1], datapoint->affiliation_id);
}

void delete_datapoint(Datapoint* datapoint){
    delete_vector(datapoint->position);
    free(datapoint);
}

//--------------- Cluster structure--------------//
typedef struct Centroid{
    Vector* center;
    bool is_unmoved;
    int id;
    double sensitivity;
} Centroid;

Centroid* new_centroid(int id, double scaling_factor){
    Centroid* centroid = malloc(sizeof(Centroid));
    centroid->id = id;
    centroid->sensitivity = 0.0;
    centroid->center = new_vector(DIMENSIONALITY);
    centroid->is_unmoved = true;
    for (size_t i = 0; i < centroid->center->length; i++) {
        double random_value = (double)rand() / ((double) RAND_MAX);
        centroid->center->scalars[i] = (rand()&1? 1.0 : -1.0) * random_value * scaling_factor;
    }
    return centroid;
}

void delete_centroid(Centroid* centroid){
    delete_vector(centroid->center);
    free(centroid);
}

void print_centroid(Centroid* centroid){
    printf("CentroidID: %d, Sensitivity: %lf\n\tCenter: ", centroid->id, centroid->sensitivity);
    print_vector(centroid->center);
}


//------------- Dataset structure ---------------//
typedef struct Dataset{
    Datapoint** points;
    size_t size;
    size_t dimensionality;
} Dataset;

Dataset* new_dataset(size_t dimensionality){
    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->size = 0;
    dataset->dimensionality = dimensionality;
    dataset->points = (Datapoint**)malloc(MAX_INPUT_LENGHT * sizeof(Datapoint*));
    for (size_t i = 0; i < MAX_INPUT_LENGHT; i++) {
        dataset->points[i] = new_datapoint(dimensionality);
    }
    return dataset;
}

void delete_dataset(Dataset* dataset){
    if (dataset->points != NULL) {
        for (size_t i = 0; i < MAX_INPUT_LENGHT; i++) {
            delete_datapoint(dataset->points[i]);
        }
        free(dataset->points);
    }
    free(dataset);
}

void add_datapoint_dataset(Dataset* dataset, Datapoint* datapoint){
    dataset->points[dataset->size]->position->length = datapoint->position->length;
    for (size_t i = 0; i < datapoint->position->length; i++) {
        dataset->points[dataset->size]->position->scalars[i] = datapoint->position->scalars[i];
    }
    dataset->size++;
}

void print_dataset(Dataset* dataset){
    printf("Dimensionality of data: %ld\nDataset size: %ld\n", dataset->dimensionality, dataset->size);
    for (size_t i = 0; i < dataset->size; i++) {
        print_datapoint(dataset->points[i]);
    }
}



//--------------------------------------------//
//---------- HELPER FUNCTIONS & CO. ----------//
//--------------------------------------------//
size_t read_input(Dataset* dataset);
void print_centers(Centroid** centroids);

void reinitialise_dataset(Dataset* dataset);
void scramble_dataset(Dataset* dataset);
void normalise_dataset(Dataset* dataset, double* minima, double* maxima);
void denormalise_dataset(Dataset* dataset, double* minima, double* maxima);
double normalise_value(double value, double min, double max);
double denormalise_value(double value, double min, double max);


static double MAX_VALUES[DIMENSIONALITY] = {FLOAT_MIN, FLOAT_MIN};
static double MIN_VALUES[DIMENSIONALITY] = {FLOAT_MAX, FLOAT_MAX};
static size_t CLUSTER_COUNT = 0;
#if DATA_LOGGING
static int fds = 6;
static FILE* fd[] = {NULL, NULL, NULL, NULL, NULL, NULL};
static FILE* ellbow = NULL;
#endif
static size_t ITERATION_STEP = 0;

//--------------------------------------------//
//------ VECTOR QUANTIZATION FUNCTIONS -------//
//--------------------------------------------//

void vector_quantisation_n_clusters();
void vector_quantisation_x_clusters();

double error_centroid(Dataset* dataset, Centroid* centroid);
double error_total(Dataset* dataset, Centroid** centroids);
double error_function(Vector* current_position, Vector* target_position);

void compute_cluster_centers(Dataset* dataset, Centroid** centroids);
double compute_distance(const Vector* point1, const Vector* point2);
double compute_sensitive_distance(double sensitivity, double distance);
Centroid* find_closest(Centroid** centroids, Datapoint* point);
void update_center(Dataset* dataset, Centroid* centroid);

void dump_centers(Centroid** centroids, Datapoint** centers);
void load_centers(Centroid** centroids, Datapoint** centers);
void reinitialise_centroids(Centroid** centroids);

size_t find_elbow(double* errors);

//--------------------------------------------//
//------------------ MAIN --------------------//
//--------------------------------------------//
int main(){
    srand((unsigned int)time(NULL));
#if DATA_LOGGING
    fd[0] = fopen("cluster1", "w");
    fd[1] = fopen("cluster2", "w");
    fd[2] = fopen("cluster3", "w");
    fd[3] = fopen("cluster4", "w");
    fd[4] = fopen("cluster5", "w");
    fd[5] = fopen("cluster6", "w");
#endif

#if !UNKNOWN_CLUSTER_COUNT
    vector_quantisation_n_clusters();
#else
    vector_quantisation_x_clusters();
#endif
    
#if DATA_LOGGING
    for (int i = 0; i < fds; i++) {
        fclose(fd[i]);
    };
#endif
    return EXIT_SUCCESS;
}


size_t read_input(Dataset* dataset){
    
    size_t cluster_count = 0;
    char* input_line = malloc(50*sizeof(char));
#if !UNKNOWN_CLUSTER_COUNT
    scanf("%ld\n", &cluster_count);
#endif
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        if (scanf("%s\n", input_line) == EOF){
            break;
        }
        sscanf(input_line, "%lf,%lf\n",
               &dataset->points[i]->position->scalars[0],
               &dataset->points[i]->position->scalars[1]);
        for (size_t d = 0; d < DIMENSIONALITY; d++) {
            if (dataset->points[i]->position->scalars[d] > MAX_VALUES[d]) {
                MAX_VALUES[d] = dataset->points[i]->position->scalars[d];
            }
            if (dataset->points[i]->position->scalars[d] < MIN_VALUES[d]) {
                MIN_VALUES[d] = dataset->points[i]->position->scalars[d];
            }
        }
        dataset->size++;
    }
    
    free(input_line);
    return cluster_count;
}

void print_centers(Centroid** centroids){
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        double denormalised_value_fst = denormalise_value(centroids[i]->center->scalars[0], MIN_VALUES[0], MAX_VALUES[0]);
        printf("%lf,", denormalised_value_fst);
        for (size_t d = 0; d < DIMENSIONALITY-2; d++) {
            double denormalised_value = denormalise_value(centroids[i]->center->scalars[d], MIN_VALUES[d], MAX_VALUES[d]);
            printf("%lf,", denormalised_value);
        }
        double denormalised_value_lst = denormalise_value(centroids[i]->center->scalars[DIMENSIONALITY-1],
                                                          MIN_VALUES[DIMENSIONALITY-1],
                                                          MAX_VALUES[DIMENSIONALITY-1]);
        printf("%lf\n", denormalised_value_lst);
    }
}


void vector_quantisation_n_clusters(){
    Dataset* dataset = new_dataset(DIMENSIONALITY);
    
    CLUSTER_COUNT = read_input(dataset);
    
    normalise_dataset(dataset, MIN_VALUES, MAX_VALUES);
    Centroid** centroids = malloc(CLUSTER_COUNT * sizeof(Centroid*));
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        centroids[i] = new_centroid((int)i, 1.0);
    }
    
    double total_error_previous_try = FLOAT_MAX;
    double total_error = FLOAT_MAX;
    double total_error_old = 0.0;
    double epsilon = 0.0001;
    
    Datapoint** best_centers = malloc(CLUSTER_COUNT * sizeof(Datapoint*));
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        best_centers[i] = new_datapoint(DIMENSIONALITY);
    }
    
    time_t time_at_beginning = time(0);
    while ((time(0) - time_at_beginning) < 20) {
        while ((time(0) - time_at_beginning) < 20) {
            if (fabs(total_error_old - total_error) <= epsilon && total_error != FLOAT_MAX) {
                break;
            }
            compute_cluster_centers(dataset, centroids);
            total_error_old = total_error;
            total_error = error_total(dataset, centroids);
        }
        if (total_error < total_error_previous_try) {
            dump_centers(centroids, best_centers);
            total_error_previous_try = total_error;
        }
        reinitialise_centroids(centroids);
        reinitialise_dataset(dataset);
        total_error = total_error_old = FLOAT_MAX;
    }
    
    load_centers(centroids, best_centers);
    print_centers(centroids);
    
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        delete_centroid(centroids[i]);
        delete_datapoint(best_centers[i]);
    }
}

void vector_quantisation_x_clusters(){
    Dataset* dataset = new_dataset(DIMENSIONALITY);
    
    read_input(dataset);
    if (dataset->size == 1) {
        printf("%lf,%lf\n", dataset->points[0]->position->scalars[0], dataset->points[0]->position->scalars[1]);
        return;
    }
    if (dataset->size == 2) {
        double x1 = dataset->points[0]->position->scalars[0];
        double x2 = dataset->points[1]->position->scalars[0];
        double y1 = dataset->points[0]->position->scalars[1];
        double y2 = dataset->points[1]->position->scalars[1];
        printf("%lf,%lf\n", (x1+x2)/2.0 , (y1+y2)/2.0);
        return;
    }
    
    normalise_dataset(dataset, MIN_VALUES, MAX_VALUES);
    
    double errors_diff_cluster_count[MAX_CLUSTER_COUNT];
    memset(errors_diff_cluster_count, 0.0, MAX_CLUSTER_COUNT*sizeof(double));
    
    Datapoint*** best_centers = malloc(MAX_CLUSTER_COUNT * sizeof(Datapoint**));
    
    size_t CLUSTER_COUNT_COMPUTED = 0;
    time_t time_at_beginning = time(0);
    while ((time(0) - time_at_beginning) < 9 && CLUSTER_COUNT_COMPUTED < MAX_CLUSTER_COUNT) {
        CLUSTER_COUNT = ++CLUSTER_COUNT_COMPUTED;
        best_centers[CLUSTER_COUNT_COMPUTED-1] = malloc(CLUSTER_COUNT_COMPUTED * sizeof(Datapoint*));
        for (size_t i = 0; i < CLUSTER_COUNT_COMPUTED; i++) {
            best_centers[CLUSTER_COUNT_COMPUTED-1][i] = new_datapoint(DIMENSIONALITY);
        }
        
        Centroid** centroids = malloc(CLUSTER_COUNT_COMPUTED * sizeof(Centroid*));
        for (size_t i = 0; i < CLUSTER_COUNT_COMPUTED; i++) {
            centroids[i] = new_centroid((int)i, 1.0);
        }
        
        size_t trial = 0;
        double total_error_previous_try = FLOAT_MAX;
        time_t three_sec_max = time(0);
        while (trial++ < 50 && (time(0) - three_sec_max) < 1) {
            double total_error = FLOAT_MAX;
            double total_error_old = FLOAT_MAX;
            double epsilon = 0.00001;
            
            reinitialise_centroids(centroids);
            reinitialise_dataset(dataset);
            // allow max 1 sec per cluseter_count iterations
            time_t one_sec_max = time(0);
            while ((time(0) - one_sec_max) < 1) {
                if (fabs(total_error_old - total_error) <= epsilon && total_error != FLOAT_MAX) {
                    break;
                }
                compute_cluster_centers(dataset, centroids);
                total_error_old = total_error;
                total_error = error_total(dataset, centroids);
            }
            if (total_error < total_error_previous_try) {
                dump_centers(centroids, best_centers[CLUSTER_COUNT_COMPUTED-1]);
                total_error_previous_try = total_error;
                errors_diff_cluster_count[CLUSTER_COUNT_COMPUTED-1] = total_error;
            }
        }
        //        printf("For cluster count: %ld succedded with %ld trials.\n", CLUSTER_COUNT_COMPUTED, trial);
        for (size_t i = 0; i < CLUSTER_COUNT_COMPUTED; i++) {
            delete_centroid(centroids[i]);
        }
        free(centroids);
    }
    CLUSTER_COUNT = CLUSTER_COUNT_COMPUTED;
    size_t optimal_cluster_count = find_elbow(errors_diff_cluster_count);
    //    printf("Optimal cluster count: %ld\n", optimal_cluster_count);
    CLUSTER_COUNT = optimal_cluster_count;
    Centroid** optimal_centroids = malloc(optimal_cluster_count * sizeof(Centroid*));
    for (size_t i = 0; i < optimal_cluster_count; i++) {
        optimal_centroids[i] = new_centroid((int)i, 1.0);
    }
    load_centers(optimal_centroids, best_centers[optimal_cluster_count-1]);
    print_centers(optimal_centroids);
    
    for (size_t i = 0; i < CLUSTER_COUNT_COMPUTED; i++) {
        for (int j = 0; j < i+1; j++) {
            delete_datapoint(best_centers[i][j]);
        }
        free(best_centers[i]);
    }
    free(best_centers);
    
    for (size_t i = 0; i < optimal_cluster_count; i++) {
        delete_centroid(optimal_centroids[i]);
    }
    free(optimal_centroids);
    
}



void reinitialise_dataset(Dataset* dataset){
    for (size_t i = 0; i < dataset->size; i++) {
        dataset->points[i]->affiliation_id = -1;
    }
}

void normalise_dataset(Dataset* dataset, double* minima, double* maxima){
    for (size_t d = 0; d < dataset->dimensionality; d++) {
        for (size_t i = 0; i < dataset->size; i++) {
            dataset->points[i]->position->scalars[d] = normalise_value(dataset->points[i]->position->scalars[d], MIN_VALUES[d], MAX_VALUES[d]);
        }
    }
}

void denormalise_dataset(Dataset* dataset, double* minima, double* maxima){
    for (size_t d = 0; d < dataset->dimensionality; d++) {
        for (size_t i = 0; i < dataset->size; i++) {
            dataset->points[i]->position->scalars[d] = denormalise_value(dataset->points[i]->position->scalars[d], MIN_VALUES[d], MAX_VALUES[d]);
        }
    }
}

double normalise_value(double value, double min, double max){
#if ASSERTIONS_ENABLED
    assert(max != min);
#endif
    return (2.0 * value - max - min) / (max - min);
}

double denormalise_value(double value, double min, double max){
    return (max*(value + 1.0) - min*(value - 1.0))/2.0;
}

void scramble_dataset(Dataset* dataset){
    for (size_t i = 0; i < dataset->size/2; i++) {
        size_t random_index = rand() % (dataset->size/2);
        Datapoint* temp_point_holder = dataset->points[random_index];
        dataset->points[random_index] = dataset->points[random_index+dataset->size/2];
        dataset->points[random_index+dataset->size/2] = temp_point_holder;
    }
}

double error_centroid(Dataset* dataset, Centroid* centroid){
    double error = 0.0;
    size_t neighbouring_points_count = 0;
    for (size_t i = 0; i < dataset->size; i++) {
        if (dataset->points[i]->affiliation_id == centroid->id) {
            error += SQR(compute_distance(dataset->points[i]->position, centroid->center));
            neighbouring_points_count++;
        }
    }
    if (neighbouring_points_count) {
        return error/(double)neighbouring_points_count;
    }
    return FLOAT_MAX;
}

double error_total(Dataset* dataset, Centroid** centroids){
    double total_error = 0.0;
    for (size_t centroid_id = 0; centroid_id < CLUSTER_COUNT; centroid_id++) {
        double err_one_centroid = error_centroid(dataset, centroids[centroid_id]);
        if (err_one_centroid == FLOAT_MAX) {
            return FLOAT_MAX;
        }
        total_error += err_one_centroid;
    }
    return total_error/(double)CLUSTER_COUNT;
}

void compute_cluster_centers(Dataset* dataset, Centroid** centroids){
//    Datapoint* random_point = dataset->points[rand() % dataset->size];
    for (size_t i = 0; i < dataset->size; i++) {
        Centroid* closest_centroid = find_closest(centroids, dataset->points[i]);
        update_center(dataset, closest_centroid);
    }
    scramble_dataset(dataset);
}

double compute_distance(const Vector* point1, const Vector* point2){
#if ASSERTIONS_ENABLED
    assert(point1->length == point1->length);
#endif
    double sqrd_components_sum = 0.0;
    for (size_t d = 0; d < point1->length; d++) {
        sqrd_components_sum += SQR(point1->scalars[d] - point2->scalars[d]);
    }
    return sqrt(sqrd_components_sum);
}

double compute_sensitive_distance(double sensitivity, double distance){
    return (1.0 - SENSITIVITY_SIGNIFICANCE) * distance - SENSITIVITY_SIGNIFICANCE * sensitivity;
}

Centroid* find_closest(Centroid** centroids, Datapoint* point){
    int id_winning_centroid = 0;
    double best_score = FLOAT_MAX;
    for (int centroid_id = 0; centroid_id < CLUSTER_COUNT; centroid_id++) {
        double distance = compute_distance(centroids[centroid_id]->center, point->position);
        double score = compute_sensitive_distance(centroids[centroid_id]->sensitivity, distance);
        if (score < best_score) {
            id_winning_centroid = centroid_id;
            best_score = score;
        }
    }
    point->affiliation_id = id_winning_centroid;
    for (size_t centroid_id = 0; centroid_id < CLUSTER_COUNT; centroid_id++) {
        if (centroids[centroid_id]->id == id_winning_centroid) {
            centroids[centroid_id]->sensitivity = 0.0;
        } else {
            centroids[centroid_id]->sensitivity += SENSITIVITY_GAIN;
        }
    }
    return centroids[id_winning_centroid];
}

void update_center(Dataset* dataset, Centroid* centroid){
    for (size_t d = 0; d < DIMENSIONALITY; d++) {
        double net_displacement = 0.0;
        int pulling_points_count = 0;
        for (size_t i = 0; i < dataset->size; i++) {
            if (dataset->points[i]->affiliation_id == centroid->id) {
                pulling_points_count++;
                net_displacement += dataset->points[i]->position->scalars[d] - centroid->center->scalars[d];
            }
        }
        // jump at most, as the furthest point is pulling
        centroid->center->scalars[d] += (centroid->is_unmoved? 1.0 : LEARNING_RATE) * (net_displacement/(double)pulling_points_count);
    }
    ITERATION_STEP++;
#if DATA_LOGGING
//    for (int i = 0; i < fds; i++) {
//        if (ITERATION_STEP > 100 && centroid->id == i) {
//            fprintf(fd[i], "%lf,%lf\n", denormalise_value(centroid->center->position->scalars[0], MIN_VALUES[0], MAX_VALUES[0])
//                    , denormalise_value(centroid->center->position->scalars[1], MIN_VALUES[1], MAX_VALUES[1]));
//        }
//    }
#endif
}


void dump_centers(Centroid** centroids, Datapoint** centers){
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        for (size_t d = 0; d < DIMENSIONALITY; d++) {
            centers[i]->position->scalars[d] = centroids[i]->center->scalars[d];
        }
        centers[i]->affiliation_id = centroids[i]->id;
    }
}
void load_centers(Centroid** centroids, Datapoint** centers){
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        for (size_t d = 0; d < DIMENSIONALITY; d++) {
            centroids[i]->center->scalars[d] = centers[i]->position->scalars[d];
        }
    }
}

void reinitialise_centroids(Centroid** centroids){
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        centroids[i]->sensitivity = 0.0;
        for (size_t j = 0; j < centroids[i]->center->length; j++) {
            double random_value = (double)rand() / ((double) RAND_MAX);
            centroids[i]->center->scalars[j] = (rand()&1? 1.0 : -1.0) * random_value;
        }
    }
}

size_t find_elbow(double* errors){
    for (size_t i = 0; i < CLUSTER_COUNT; i++) {
        if (errors[i] < (1.0 - 90.0/100.0)*errors[0]) {
            return i+1;
        }
    }
//    for (size_t i = 1; i < CLUSTER_COUNT-1; i++) {
//        if (7.0*(errors[i] - errors[i+1]) < errors[i-1] - errors[i]) {
//            return i+1;
//        }
//    }
    return 1;
}

