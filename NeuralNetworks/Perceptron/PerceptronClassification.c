//
//  main.c
//  Input: 
//      1. a set of (linearly separable) training points consisting in x and y coordinates and class +1 or -1
//      2. a set of testing points (only x and y coordinates)
//  Output: The class of each of the testing points
//
//  Created by dikov on 24/12/15.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MAX_INPUT_LENGHT 1000
#define true 1
#define false 0
typedef unsigned char bool;

#define TERMINATING_STR "0,0,0\n"

typedef struct Vector{
    double* scalars;
    size_t length;
} Vector;

Vector* new_vec(size_t len){
    Vector* vec = malloc(sizeof(Vector));
    vec->length = len;
    vec->scalars = malloc(len*sizeof(double));
    for (int i = 0; i < len; i++) {
        vec->scalars[i] = 0.0;
    }
    return vec;
}

void delete_vec(Vector* vec){
    if (vec->scalars != NULL) {
        free(vec->scalars);
    }
    free(vec);
}

// --------- Neuron Structure ---------- //
#define DIMENSION_INPUT 2
#define DIMENSION_OUTPUT 1
#define LEARNING_RATE 0.01

double evaluate_activation_function(double input){
    float out = tanh(input);
    if (out > 0.0){
        return 1.0;
    }
    return -1.0;
}

typedef struct Neuron{
    double weights[DIMENSION_INPUT+1];   // +bias
    double (*evaluate_output)(double input);
} Neuron;

Neuron* new_neuron(){
    Neuron* neuron = malloc(sizeof(Neuron));
    neuron->evaluate_output = evaluate_activation_function;
    for (int i = 0; i < DIMENSION_INPUT+1; i++) {
        // initialise the neuron with small random weights
        double random_weight = (double)rand() / ((double) RAND_MAX);
        if (random_weight > 0.01 || random_weight < -0.01) {
            random_weight /= 100.0;
        }
        neuron->weights[i] = random_weight;
    }
    return neuron;
}

void delete_neuron(Neuron* neuron){
    free(neuron);
}

// ----------- HELPER FUNCTIONS ------------- //
size_t read_input(Vector* input_points[], Vector* true_results[]);
void print_output(Neuron* neuron);

// -------- BACKPROPAGATION FUNCTIONS -------- //
double compute_net_value(Vector* inputs, Neuron* neuron);
double error_total(Vector* training_points_result[], Vector* true_results[], Neuron* neuron, size_t training_set_size);
double error_single_point(double training_point_true, double neuron_output);

double compute_gradient(Vector* training_point, double neuron_output, double real_result, size_t weight_index);
void update_weights(Neuron* neuron, double delta_weights[]);

// ------------------- MAIN ----------------- //
// ------------------------------------------ //
int main(){
    
    Vector* input_points[MAX_INPUT_LENGHT];
    Vector* input_points_true_result[MAX_INPUT_LENGHT];
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        input_points[i] = new_vec(4);
        input_points_true_result[i] = new_vec(1);
    }
    size_t TRAINING_SET_SIZE = 0;
    TRAINING_SET_SIZE = read_input(input_points, input_points_true_result);
    
    Neuron* neuron = new_neuron();
    
    double current_total_error = 1.0;
    // assume linear separability
    while ((current_total_error = error_total(input_points, input_points_true_result, neuron, TRAINING_SET_SIZE)) > 0.00001) {
//        printf("Total error: %lf\n", current_total_error);
        for (int i = 0; i < TRAINING_SET_SIZE; i++) {
            double neuron_out = neuron->evaluate_output(compute_net_value(input_points[i], neuron));
//            printf("Neuron Output: %lf\tReal Output: %lf\n", neuron_out, input_points[i]->scalars[3]);
            double delta_weights[DIMENSION_INPUT+1];
            for (int j = 0; j < DIMENSION_INPUT+1; j++) {
                delta_weights[j] = compute_gradient(input_points[i], neuron_out, input_points_true_result[i]->scalars[0], j);
//                printf("Weight_%d: %lf\n",j, delta_weights[j]);
            }
            update_weights(neuron, delta_weights);
        }
    }
    
    print_output(neuron);
    
    delete_neuron(neuron);
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        delete_vec(input_points[i]);
        delete_vec(input_points_true_result[i]);
    }
    
    return EXIT_SUCCESS;
}
// ---------------------------------------- //
// --------------- END OF MAIN ------------ //


/*
* Read the training samples and the corresponding classes.
*/
size_t read_input(Vector* input_points_training[], Vector* true_results[]){
    
    size_t input_size = 0;
    char* input_line = malloc(50*sizeof(char));
    
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        if (scanf("%s\n", input_line) == EOF){
            break;
        }
        if (!strncmp(input_line, TERMINATING_STR, (int)strlen(TERMINATING_STR)-1)) {
            break;
        }
        sscanf(input_line, "%lf,%lf,%lf\n",
               &input_points_training[i]->scalars[0],
               &input_points_training[i]->scalars[1],
               &true_results[i]->scalars[0]);
        input_points_training[i]->scalars[2] = 1.0f;
        input_size++;
    }

    free(input_line);
    return input_size;
}

/*
* Read the testing samples, compute the class and print it out.
*/
void print_output(Neuron* neuron){
    char* input_line = malloc(50*sizeof(char));
    Vector* testing_point_in = new_vec(3);
    double testing_point_out = 0.0;
    size_t test_set_size = 0;
    while (scanf("%s", input_line) != EOF) {
        test_set_size++;
        sscanf(input_line, "%lf,%lf\n", &testing_point_in->scalars[0], &testing_point_in->scalars[1]);
        testing_point_in->scalars[2] = 1.0;     // bias
        testing_point_out = neuron->evaluate_output(compute_net_value(testing_point_in, neuron));
        if (testing_point_out >= 0) {
            printf("+1\n");
        }else{
            printf("-1\n");
        }
    }
    delete_vec(testing_point_in);
    free(input_line);
}

/*
* Compute the weighted sum of all inputs.
*/
double compute_net_value(Vector* inputs, Neuron* neuron){
    double net_value = 0.0;
    for (int i = 0; i < inputs->length; i++) {
        net_value += inputs->scalars[i] * neuron->weights[i];
    }
    return net_value;
}

/*
* Estimate the error for all training samples.
*/
double error_total(Vector* training_points[], Vector* true_results[], Neuron* neuron, size_t training_set_size){
    double err_accumulator = 0.0;
    for (int i = 0; i < training_set_size; i++) {
        err_accumulator += error_single_point(true_results[i]->scalars[0],
                    neuron->evaluate_output(compute_net_value(training_points[i], neuron)));
    }
    return err_accumulator;
}

/*
* Estimate the squared error for one training sample
*/
double error_single_point(double training_point_result, double neuron_output){
    return (training_point_result - neuron_output) * (training_point_result - neuron_output);
}

/*
* Compute the change in weights and scale it with the learning rate.
*/
double compute_gradient(Vector* training_point, double neuron_output, double real_result, size_t weight_index){
    return (LEARNING_RATE * (real_result - neuron_output) * training_point->scalars[weight_index]);
}

/*
* Update the new weights of the perceptron.
*/
void update_weights(Neuron* neuron, double delta_weights[]){
    for (int i = 0; i < DIMENSION_INPUT+1; i++) {
        neuron->weights[i] += delta_weights[i];
    }
}
