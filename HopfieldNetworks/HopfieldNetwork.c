//
//  A simple implementation of a Hopfield network for pattern recognition.
//  Input: 
//      1. a set of correct "images" consisting of . and * symbols.
//      2. distorted images
//  Output: a correct image, which is believed to be the closest one to the distorted pattern analysed.
//
//  Created by dikov on 24/12/15.
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define true 1
#define false 0
typedef unsigned char bool;
#define FLOAT_MIN -0x1.fffffep+127f
#define FLOAT_MAX 0x1.fffffep+127f

#define ASSERTIONS_ENABLED false
#define SQR(x) ((x)*(x))

#define IMAGE_SIZE_X 5
#define IMAGE_SIZE_Y 5
#define MAX_INPUT_IMG_COUNT 3
#define NEXT_IMG_DELIMITER "-\n"
#define END_IMG_INPUT "---\n"

#define NOISE_PERCENTAGE 4.0
//-----------------------------------------------//
//---------- CUSTOM TYPES -----------------------//
//-----------------------------------------------//

//----------- Vector structure ---------------//
typedef struct Vector{
    int* scalars;
    size_t length;
} Vector;

Vector* new_vector(size_t len){
    Vector* vec = malloc(sizeof(Vector));
    vec->length = len;
    vec->scalars = malloc(len*sizeof(int));
    for (size_t i = 0; i < len; i++) {
        vec->scalars[i] = 0;
    }
    return vec;
}

void print_vector(Vector* vec){
    printf("[");
    for (size_t i = 0; i < vec->length-1; i++) {
        printf("%d, ", vec->scalars[i]);
    }
    printf("%d]\n", vec->scalars[vec->length-1]);
}

void delete_vector(Vector* vec){
    if (vec->scalars != NULL) {
        free(vec->scalars);
    }
    free(vec);
}

//----------- Matrix structure ---------------//
typedef struct Matrix{
    Vector** columns;
    size_t col_count;
    size_t row_count;
} Matrix;

Matrix* new_matrix(size_t row_count, size_t col_count){
    Matrix* matrix = malloc(sizeof(Matrix));
    matrix->col_count = col_count;
    matrix->row_count = row_count;
    matrix->columns = malloc(col_count * sizeof(Vector*));
    for (size_t i = 0; i < col_count; i++) {
        matrix->columns[i] = new_vector(row_count);
    }
    return matrix;
}

void print_matrix(Matrix* matrix){
    printf("Matrix dimensionality: %ldx%ld\n    ", matrix->row_count, matrix->col_count);
    for (size_t i = 0; i < matrix->col_count; i++) {
        printf("%ld\t", i);
    }
    printf("\n");
    for (size_t j = 0; j < matrix->row_count; j++) {
        printf("%ld.| ", j);
        for (size_t i = 0; i < matrix->col_count-1; i++) {
            printf("%d\t", matrix->columns[i]->scalars[j]);
        }
        printf("%d |\n", matrix->columns[matrix->col_count-1]->scalars[j]);
    }
}

void delete_matrix(Matrix* matrix){
    for (size_t i = 0; i < matrix->col_count; i++) {
        delete_vector(matrix->columns[i]);
    }
    free(matrix->columns);
    free(matrix);
}

//--------- Neuron structure ----------------//
struct Neuron;
struct Network;
typedef int (*function)(struct Network* net, int current_neuron_id);

/* Define a  
*/
int neuron_function(struct Network* net, int current_neuron_id);

typedef struct Neuron{
    function switching_function;
    int id;
    int state;
} Neuron;

Neuron* new_neuron(int id){
    Neuron* neuron = malloc(sizeof(Neuron));
    neuron->id = id;
    neuron->state = 0;
    neuron->switching_function = neuron_function;
    return neuron;
}

void print_neuron(Neuron* neuron){
    printf("NeuronID: %d, State %d\n", neuron->id, neuron->state);
}

void delete_neuron(Neuron* neuron){
    free(neuron);
}

//---------- Network structure --------------//
typedef struct Network{
    Neuron** neurons;
    Matrix* weights;
    size_t size;
} Network;

/*
* A network consists of identical interconnected neurons.
* Each of them has a symmetrical (non-reflexive) connection to each other neuron.
* 
*/
Network* new_network(size_t size){
    Network* network = malloc(sizeof(Network));
    network->size = size;
    network->neurons = malloc(size * sizeof(Neuron*));
    for (size_t i = 0; i < size; i++) {
        network->neurons[i] = new_neuron((int)(i));
    }
    network->weights = new_matrix(size, size);
    return network;
}

void print_network(Network* network, size_t row_len){
    printf("Network size: %ld\n", network->size);
    for (size_t i = 1; i < network->size+1; i++) {
        printf("%c ", (network->neurons[i-1]->state == 0 ? '0' : (network->neurons[i-1]->state == -1 ? '*' : '.')));
        if (i % row_len == 0) {
            printf("\n");
        }
    }
}

void delete_network(Network* network){
    for (size_t i = 0; i < network->size; i++) {
        delete_neuron(network->neurons[i]);
    }
    delete_matrix(network->weights);
    free(network);
}

/*
* This is the function that changes the states of individual neurons
* in correspondence to the net influence of the neighbouring neurons.
*/
int neuron_function(Network* network, int neuron_id){
    int weighted_sum = 0;
    for (size_t i = 0; i < network->size; i++) {
        weighted_sum += network->weights->columns[neuron_id]->scalars[i] * network->neurons[i]->state;
    }
    if (weighted_sum >= 0) {
        return 1;
    }
    return -1;
}

//--------------------------------------------//
//---------- HELPER FUNCTIONS & CO. ----------//
//--------------------------------------------//
void read_input_patterns(char*** teaching_patterns, size_t* teaching_patterns_count, char*** distorted_paterns, size_t* distorted_patterns_count);
void print_output_patterns(Network* network);

void normalise_input_patterns(char** teaching_patterns, size_t teaching_patterns_count, char** distorted_paterns, size_t distorted_patterns_count);
char denormalise_character(int state);

//--------------------------------------------//
//-------- HOPFIELD NETWORK FUNCTIONS --------//
//--------------------------------------------//
void train_network(Network* network, char** teaching_patterns, size_t patterns_count);
void single_shot_learning(Network* network, char* teaching_pattern);
void initialise_states_network(Network* network, char* pattern);

void test_network(Network* network, char* distorted_pattern, char** teaching_patterns, size_t teaching_patterns_count);
void test_distorted_patterns(Network* network, char** distorted_patterns, size_t distorted_patterns_count, char** teaching_patterns, size_t teaching_patterns_count);

bool assert_correct_convergence(Network* network, char** teaching_patterns, size_t teaching_patterns_count);
void noisify_the_network_state(Network* network);

//--------------------------------------------//
//------------------ MAIN --------------------//
//--------------------------------------------//
int main(){
    srand((unsigned int)time(NULL));
    char** teaching_patterns = NULL;
    char** distorted_patterns = NULL;
    size_t TEACHING_PATTERNS_COUNT = 0;
    size_t DISTORTED_PATTERNS_COUNT = 0;
    read_input_patterns(&teaching_patterns, &TEACHING_PATTERNS_COUNT, &distorted_patterns, &DISTORTED_PATTERNS_COUNT);
    normalise_input_patterns(teaching_patterns, TEACHING_PATTERNS_COUNT, distorted_patterns, DISTORTED_PATTERNS_COUNT);
    
    Network* network = new_network(IMAGE_SIZE_X*IMAGE_SIZE_Y);
    //    print_network(network, IMAGE_SIZE_X);
    if ((double)TEACHING_PATTERNS_COUNT > 0.139 * network->size) {
        printf("WARRNING: Network may not be able to learn so many patterns!\n");
    }
    
    train_network(network, teaching_patterns, TEACHING_PATTERNS_COUNT);
    test_distorted_patterns(network, distorted_patterns, DISTORTED_PATTERNS_COUNT, teaching_patterns, TEACHING_PATTERNS_COUNT);
    
    delete_network(network);
    for (int i = 0; i < TEACHING_PATTERNS_COUNT; i++) {
        free(teaching_patterns[i]);
    }
    for (int i = 0; i < DISTORTED_PATTERNS_COUNT; i++) {
        free(distorted_patterns[i]);
    }
    free(teaching_patterns);
    free(distorted_patterns);
    return EXIT_SUCCESS;
}

/*
* Read the input pattern vectors and the distorded patterns, which have to be recognised.
*/
void read_input_patterns(char*** teaching_patterns, size_t* teaching_patterns_count, char*** distorted_paterns, size_t* distorted_patterns_count){
    *teaching_patterns = (char**) malloc(MAX_INPUT_IMG_COUNT * sizeof(char*));
    *distorted_paterns = (char**) malloc(MAX_INPUT_IMG_COUNT * sizeof(char*));
    char* input_line = malloc((IMAGE_SIZE_X+1) * sizeof(char));
    memset(input_line, 0, (IMAGE_SIZE_X+1) * sizeof(char));
    unsigned char line_count = 0;
    
    for (int i = 0; i < MAX_INPUT_IMG_COUNT; i++) {
        (*teaching_patterns)[i] = malloc((IMAGE_SIZE_X*IMAGE_SIZE_Y+1) * sizeof(char));
        memset((*teaching_patterns)[i], '\0', (IMAGE_SIZE_X*IMAGE_SIZE_Y+1) * sizeof(char));
        do {
            scanf("%s\n", input_line);
            if (!strncmp(END_IMG_INPUT, input_line, strlen(END_IMG_INPUT)-1)) {
                memset(input_line, 0, (IMAGE_SIZE_X+1) * sizeof(char));
                (*teaching_patterns_count)++;
                goto read_distorted;
            }
            sscanf(input_line, "%s\n", &((*teaching_patterns)[i][IMAGE_SIZE_X * line_count++]));
        } while (strncmp(NEXT_IMG_DELIMITER, input_line, strlen(NEXT_IMG_DELIMITER)-1));
        (*teaching_patterns)[i][IMAGE_SIZE_X * IMAGE_SIZE_Y] = '\0';
        (*teaching_patterns_count)++;
        line_count = 0;
    }
read_distorted:
    line_count = 0;
    for (int i = 0; i < MAX_INPUT_IMG_COUNT; i++) {
        (*distorted_paterns)[i] = malloc((IMAGE_SIZE_X*IMAGE_SIZE_Y+1) * sizeof(char));
        memset((*distorted_paterns)[i], '\0', (IMAGE_SIZE_X*IMAGE_SIZE_Y+1) * sizeof(char));
        do {
            if (scanf("%s\n", input_line) == EOF) {
                (*distorted_patterns_count)++;
                free(input_line);
                return;
            }
            sscanf(input_line, "%s\n", &((*distorted_paterns)[i][IMAGE_SIZE_X * line_count++]));
        } while (strncmp(NEXT_IMG_DELIMITER, input_line, strlen(NEXT_IMG_DELIMITER)-1));
        (*distorted_paterns)[i][IMAGE_SIZE_X * IMAGE_SIZE_Y] = '\0';
        line_count = 0;
        (*distorted_patterns_count)++;
    }
    free(input_line);
}

/*
* Print the netowork output accoring to the image dimensions set.
*/
void print_output_patterns(Network* network){
    for (size_t row = 0; row < IMAGE_SIZE_Y; row++) {
        for (size_t character = 0; character < IMAGE_SIZE_X; character++) {
            printf("%c", denormalise_character(network->neurons[row*IMAGE_SIZE_X+character]->state));
        }
        printf("\n");
    }
}

/*
* Check if the network output is a valid pattern
*/
bool assert_correct_convergence(Network* network, char** teaching_patterns, size_t teaching_patterns_count){
    bool is_convergence_correct = false;
    bool true_for_one_example = false;
    for (size_t pattern_id = 0; pattern_id < teaching_patterns_count && !is_convergence_correct; pattern_id++) {
        true_for_one_example = true;
        for (size_t neuron_id = 0; neuron_id < network->size && true_for_one_example; neuron_id++) {
            true_for_one_example &= (network->neurons[neuron_id]->state == teaching_patterns[pattern_id][neuron_id]);
        }
        is_convergence_correct |= true_for_one_example;
    }
    return is_convergence_correct;
}

/*
* Shake the network states if the pattern of convergence is wrong. 
*/
void noisify_the_network_state(Network* network){
    size_t victims_count = (size_t)((NOISE_PERCENTAGE/100.0) * network->size);
    for (size_t i = 0; i < victims_count; i++) {
        int victim_neuron_id = rand() % network->size;
        network->neurons[victim_neuron_id]->state = (rand()&1? -1 : 1);
    }
}

/*
* Compute the output if a distorted pattern is presented and print it.
*/
void test_network(Network* network, char* distorted_pattern, char** teaching_patterns, size_t teaching_patterns_count){
//    print_matrix(network->weights);
    initialise_states_network(network, distorted_pattern);
    // TODO: check if the convergense is to one of the input samples or something else!
    bool is_converged = false;
    time_t time_at_beginning = time(0);
    while (time(0) - time_at_beginning < 10) {
        while (!is_converged) {
            is_converged = true;
            for (int neuron_id = 0; neuron_id < network->size; neuron_id++) {
                int new_state = network->neurons[neuron_id]->switching_function(network, neuron_id);
                is_converged &= (new_state == network->neurons[neuron_id]->state);
                network->neurons[neuron_id]->state = new_state;
            }
            //        printf("Intermediate Network output:\n");
            //        print_output_patterns(network);
        }
        if (!assert_correct_convergence(network, teaching_patterns, teaching_patterns_count)){
//            printf("Not correctly converged\n");
            noisify_the_network_state(network);
            is_converged = false;
        } else {
            break;
        }
    }
    print_output_patterns(network);
}

/*
* Iterate over all distorded patterns which were given at the beginning.
*/
void test_distorted_patterns(Network* network,
                             char** distorted_patterns, size_t distorted_patterns_count,
                             char** teaching_patterns, size_t teaching_patterns_count){
    for (size_t i = 0; i < distorted_patterns_count-1; i++) {
        test_network(network, distorted_patterns[i], teaching_patterns, teaching_patterns_count);
        printf("-\n");
    }
    test_network(network, distorted_patterns[distorted_patterns_count-1], teaching_patterns, teaching_patterns_count);
}

/*
* Set the states of the network from the converted input.
*/
void initialise_states_network(Network* network, char* pattern){
    for (size_t neuron_id = 0; neuron_id < network->size; neuron_id++) {
        network->neurons[neuron_id]->state = pattern[neuron_id];
    }
}

/* 
* Read the . * image and convert it to +1/-1 valued patterns.
*/
void normalise_input_patterns(char** teaching_patterns, size_t teaching_patterns_count, char** distorted_paterns, size_t distorted_patterns_count){
    for (size_t i = 0; i < teaching_patterns_count; i++) {
        for (size_t ch = 0; ch < IMAGE_SIZE_X*IMAGE_SIZE_Y; ch++) {
            if (teaching_patterns[i][ch] == '*') {
                teaching_patterns[i][ch] = -1;
            } else if (teaching_patterns[i][ch] == '.') {
                teaching_patterns[i][ch] = 1;
            } else {
#if ASSERTIONS_ENABLED
                printf("Unknown character in the image!");
                assert(0);
#endif
            }
        }
    }
    
    for (size_t i = 0; i < distorted_patterns_count; i++) {
        for (size_t ch = 0; ch < IMAGE_SIZE_X*IMAGE_SIZE_Y; ch++) {
            if (distorted_paterns[i][ch] == '*') {
                distorted_paterns[i][ch] = -1;
            } else if (distorted_paterns[i][ch] == '.') {
                distorted_paterns[i][ch] = 1;
            } else {
#if ASSERTIONS_ENABLED
                printf("Unknown character in the image: %d\n", distorted_paterns[i][ch]);
                assert(0);
#endif
            }
        }
    }
}

/*
* Represent the numerical state in . and * 
*/
char denormalise_character(int state){
#if ASSERTIONS_ENABLED
    if (state == 0) {
        printf("Unknown state in the network!");
        assert(0);
    }
#endif
    if (state == -1) {
        return '*';
    } else {
        return '.';
    }
}

/*
* Train network on all input patterns presented.
*/
void train_network(Network* network, char** teaching_patterns, size_t patterns_count){
    for (size_t tp = 0; tp < patterns_count; tp++) {
        single_shot_learning(network, teaching_patterns[tp]);
    }
}

/* 
* Compute the neural weights according to the Hebbian rule.
*/
void single_shot_learning(Network* network, char* teaching_pattern){
//    print_matrix(network->weights);
    for (size_t neuron_id = 0; neuron_id < network->size; neuron_id++) {
        for (size_t weight_id = 0; weight_id < network->size; weight_id++) {
            if (weight_id == neuron_id) {
                continue;
            }
            network->weights->columns[neuron_id]->scalars[weight_id] += teaching_pattern[neuron_id] * teaching_pattern[weight_id];
        }
    }
//    print_matrix(network->weights);
}
