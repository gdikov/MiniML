//
//  main.c
//  Homework
//
//  Created by dikov on 04/12/15.
//  Copyright © 2015 dikov. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdarg.h>

#define MAX_INPUT_LENGHT 1001
#define true 1
#define false 0
typedef unsigned char bool;
#define FLOAT_MIN -0x1.fffffep+127f
#define FLOAT_MAX 0x1.fffffep+127f

#define SQR(x) ((x)*(x))

#define TERMINATING_STR "0,0,0\n"
#define REGRESSION false
#define CLASSIFICATION !REGRESSION

#define ASSERTIONS_ENABLED false

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

void print_vector(Vector* vec){
    printf("[");
    for (size_t i = 0; i < vec->length-1; i++) {
        printf("%lf, ", vec->scalars[i]);
    }
    printf("%lf]\n", vec->scalars[vec->length-1]);
}

void delete_vec(Vector* vec){
    if (vec->scalars != NULL) {
        free(vec->scalars);
    }
    free(vec);
}

//----------------------------------------//
// --------- Neuron Structure ----------- //
//----------------------------------------//
#define DIMENSION_INPUT 1
#define DIMENSION_OUTPUT 1

#define BIAS 1.0

#define MAX_LEARNING_RATE 0.05
#define LEARNING_RATE_MODIFIER_TRESHOLD -100000.0
#define WEIGHTS_SHAKING_FACTOR 4.5
#define MOMENTUM 0.9
#define PERFORMANCE_RATIO 0.5


static double MAX_VALUES_INPUT[] = {0.0, 0.0, 0.0};
static double MIN_VALUES_INPUT[] = {0.0, 0.0, 0.0};
static size_t HIDDEN_LAYERS_COUNT = 0;

struct Neuron;
typedef double (*function)(struct Neuron* self, Vector* input);

typedef struct Neuron{
    Vector* weights;
    function neuron_function;
    function neuron_function_derivative;
    int id;
    double stored_output;
} Neuron;

double activation_function_tanh(double input){
    return tanh(input);
}

double activation_function_lin(double input){
    return 1.0*input;
}

double propagation_function(Neuron* neuron, Vector* input){
    double net_sum = 0.0;
    for (int i = 0; i < input->length; i++) {
        net_sum += neuron->weights->scalars[i] * input->scalars[i];
    }
    return net_sum;
}

double output_function(double activated_net_sum){
    return activated_net_sum;
}

double neuron_function_hidden(Neuron* neuron, Vector* input){
    double output = output_function(activation_function_tanh(propagation_function(neuron, input)));
    neuron->stored_output = output;
    return output;
}

/*
* The derivative of the activation function of a hidden layer neuron.
*/
double neuron_function_hidden_derivative(Neuron* neuron, Vector* input){
    double derived = 1.0 - SQR(neuron->stored_output);
    return derived;
}

double neuron_function_input(Neuron* neuron, Vector* input){
    double to_normalise_vector_component = input->scalars[neuron->id];
//    to_normalise_vector_component =
//        (2.0*to_normalise_vector_component - MAX_VALUES_INPUT[neuron->id] - MIN_VALUES_INPUT[neuron->id]) /
//            (MAX_VALUES_INPUT[neuron->id] - MIN_VALUES_INPUT[neuron->id]);
    neuron->stored_output = to_normalise_vector_component;
    return to_normalise_vector_component;
}

double neuron_function_input_derivative(Neuron* neuron, Vector* input){
    return 1.0;
}

double neuron_function_output(Neuron* neuron, Vector* input){
    double output = output_function(activation_function_lin(propagation_function(neuron, input)));
#if REGRESSION
//    double denormalised_output =
//        (MAX_VALUES_INPUT[DIMENSION_INPUT]*(output + 1.0) - MIN_VALUES_INPUT[DIMENSION_INPUT]*(output - 1.0))/2.0;
    neuron->stored_output = output;
    return output;
#elif CLASSIFICATION
    double rounded_output = (output >= 0)? 1.0 : -1.0;
    neuron->stored_output = rounded_output;
    return rounded_output;
#endif
}

/*
* Output neurons have linear activation function.
*/
double neuron_function_output_derivative(Neuron* neuron, Vector* input){
//    double derived = (MAX_VALUES_INPUT[DIMENSION_INPUT] - MIN_VALUES_INPUT[DIMENSION_INPUT])/2.0;
//    return derived;
    return 1.0;
}

Neuron* new_neuron(int id, int layer_id, size_t input_connections_count){
    Neuron* neuron = malloc(sizeof(Neuron));
    neuron->id = id;
    if (layer_id == 0) {
        neuron->neuron_function = neuron_function_input;
        neuron->neuron_function_derivative = neuron_function_input_derivative;
    } else if (layer_id == HIDDEN_LAYERS_COUNT+1) {
        neuron->neuron_function = neuron_function_output;
        neuron->neuron_function_derivative = neuron_function_output_derivative;
    } else {
        neuron->neuron_function = neuron_function_hidden;
        neuron->neuron_function_derivative = neuron_function_hidden_derivative;
    }
    neuron->stored_output = 0.0;
    size_t weights_size = input_connections_count+1; //+bias
    neuron->weights = new_vec(weights_size);
    for (int i = 0; i < weights_size; i++) {
        double random_weight = (double)rand() / ((double) RAND_MAX);
        if (random_weight > 0.01 || random_weight < -0.01) {
            random_weight /= 100.0;
        }
        if (layer_id == 0) {
            neuron->weights->scalars[i] = 1.0;
        } else {
            neuron->weights->scalars[i] = random_weight * (rand()&1 ? 1.0: -1.0);
        }
    }
    return neuron;
}

void delete_neuron(Neuron* neuron){
    free(neuron->weights);
    free(neuron);
}

void print_neuron(Neuron* neuron, bool with_bias){
    printf("\tNeuronID: %d\n\t\t", neuron->id);
    for (size_t i = 0; i < neuron->weights->length-1; i++) {
        printf("Weight_%ld: %lf ", i, neuron->weights->scalars[i]);
    }
    if (with_bias) {
        printf("Weight_bias: %lf\n", neuron->weights->scalars[neuron->weights->length-1]);
    } else {
        printf("Weight_%ld: %lf\n", neuron->weights->length-1, neuron->weights->scalars[neuron->weights->length-1]);
    }
}

void print_detailed_neuron(Neuron* neuron){
    printf("\tNeuronID: %d\n\t\t", neuron->id);
    printf("Weights: ");
    print_vector(neuron->weights);
    printf("\t\tStored output: %lf\n", neuron->stored_output);
}

//--------------------------------------------//
// ----------- Layer Structure ---------------//
//--------------------------------------------//
typedef struct Layer{
    Neuron** neurons;
    int id;
    size_t size;
    struct Layer* next_layer;
    struct Layer* prev_layer;
    Vector* deltas;
    double learning_rate;
} Layer;

Layer* new_layer(size_t number_of_neurons, size_t connections_per_neuron, int id, Layer* prev, Layer* next){
    Layer* layer = malloc(sizeof(Layer));
    layer->id = id;
    layer->size = number_of_neurons;
    layer->next_layer = next;
    layer->prev_layer = prev;
    layer->deltas = new_vec(number_of_neurons);
    layer->learning_rate = MAX_LEARNING_RATE;// - abs(id - 1)*0.002;
    layer->neurons = malloc(number_of_neurons*sizeof(Neuron*));
    for (int i = 0; i < number_of_neurons; i++) {
        layer->neurons[i] = new_neuron(i, id, prev != NULL? prev->size : connections_per_neuron);
    }
    
    return layer;
}

void set_prev_layer(Layer* self, Layer* prev){
    if (self->id == 0) {
        self->prev_layer = NULL;
        return;
    }
    self->prev_layer = prev;
}

void set_next_layer(Layer* self, Layer* next){
    self->next_layer = next;
}

void delete_layer(Layer* layer){
    for (size_t neuron_id = 0; neuron_id < layer->size; neuron_id++) {
        if (layer->neurons[neuron_id] != NULL) {
            delete_neuron(layer->neurons[neuron_id]);
        }
        //        free(layer->neurons[neuron_id]);
    }
    free(layer->neurons);
    delete_vec(layer->deltas);
    free(layer);
}

void print_layer(Layer* layer){
    if (layer->id == 0 || layer->id == HIDDEN_LAYERS_COUNT+1) {
        printf("LayerId: %d contains %ld neurons:\n", layer->id, layer->size);
    } else {
        printf("LayerId: %d contains %ld neurons. Learning rate: %.10lf\n", layer->id, layer->size, layer->learning_rate);
    }
    for (int i = 0; i < layer->size; i++) {
        if (layer->id == 0) {
            print_neuron(layer->neurons[i], false);
            continue;
        }
        print_neuron(layer->neurons[i], true);
    }
}

void print_detailed_layer(Layer* layer){
    if (layer->id == 0) {
        printf("LayerId: %d contains %ld neurons:\n", layer->id, layer->size);
    } else {
        printf("LayerId: %d contains %ld neurons. Learning rate: %.14lf\n", layer->id, layer->size, layer->learning_rate);
    }
    for (int i = 0; i < layer->size; i++) {
        print_detailed_neuron(layer->neurons[i]);
    }
}

//-------------------------------------------//
// ----------- Network Structure ------------//
//-------------------------------------------//
typedef struct Network{
    Layer* input_layer;
    Layer* output_layer;
    Layer** hidden_layers;
    size_t hidden_layers_count;
}Network;

Network* new_network(size_t input_layer_size, size_t output_layer_size, size_t hidden_layers_count, ...){
    Network* network = malloc(sizeof(Network));
    HIDDEN_LAYERS_COUNT = hidden_layers_count;
    network->input_layer = new_layer(input_layer_size, 0, 0, NULL, NULL);
    network->hidden_layers = malloc(hidden_layers_count*sizeof(Layer*));
    network->hidden_layers_count = hidden_layers_count;
    va_list args;
    va_start(args, hidden_layers_count);
    size_t count_connections_prev_layer = input_layer_size;
    for (int i = 0; i < hidden_layers_count; i++) {
        size_t hidden_layer_size = va_arg(args, size_t);
        network->hidden_layers[i] = new_layer(hidden_layer_size, count_connections_prev_layer, i+1, NULL, NULL);
        count_connections_prev_layer = hidden_layer_size;
    }
    va_end(args);
    network->output_layer = new_layer(output_layer_size, count_connections_prev_layer, (int)hidden_layers_count+1, NULL, NULL);
    
    if (hidden_layers_count > 0) {
        set_next_layer(network->input_layer, network->hidden_layers[0]);
        set_prev_layer(network->hidden_layers[0], network->input_layer);
        for (int i = 0; i < hidden_layers_count-1; i++) {
            set_next_layer(network->hidden_layers[i], network->hidden_layers[i+1]);
            set_prev_layer(network->hidden_layers[hidden_layers_count-i-1], network->hidden_layers[hidden_layers_count-i-2]);
        }
        set_next_layer(network->hidden_layers[hidden_layers_count-1], network->output_layer);
        set_prev_layer(network->output_layer, network->hidden_layers[hidden_layers_count-1]);
    } else {
        set_next_layer(network->input_layer, network->output_layer);
        set_prev_layer(network->output_layer, network->input_layer);
    }
    
    return network;
}

void delete_network(Network* network){
    for (size_t hlayer_id = 0; hlayer_id < network->hidden_layers_count; hlayer_id++) {
        if (network->hidden_layers[hlayer_id] != NULL) {
            delete_layer(network->hidden_layers[hlayer_id]);
        }
    }
    free(network->hidden_layers);
    delete_layer(network->input_layer);
    delete_layer(network->output_layer);
    free(network);
}

void print_network(Network* network){
    printf("####### NETWORK CONFIGURATION #######\n");
    print_layer(network->input_layer);
    for (size_t i = 0; i < network->hidden_layers_count; i++) {
        print_layer(network->hidden_layers[i]);
    }
    print_layer(network->output_layer);
    printf("#####################################\n\n");
}

//--------------------------------------------//
// ----------- HELPER FUNCTIONS ------------- //
//--------------------------------------------//
size_t read_input(Vector* training_data[], Vector* teaching_data[]);
void test_network(Network* network);

double normalise_data(double value, double max, double min);
double denormalise_data(double value, double max, double min);
void scramble_data(Vector* training_data[], Vector* teaching_data[], size_t dataset_size);
size_t split_training_data(Vector** training_data, Vector** teaching_data, Vector*** training_portion,
                           Vector*** training_portion_teaching, Vector*** testing_portion, Vector*** testing_portion_teaching,
                           size_t training_data_size, double ratio);

void dump_weights(Network* network, Vector*** weights);
void load_weights(Network* network, Vector*** weights);

// -------- BACKPROPAGATION FUNCTIONS -------- //
void train_network_with_backprop(Network* network, Vector* training_point, Vector* teaching_point);
Vector* compute_output_network(Network* network, Vector* input);
Vector* compute_output_layer(Layer* layer, Vector* input);
double compute_output_neuron(Neuron* neuron, Vector* input);

void update_parameters(Network* network, Vector* teaching_output, Vector* network_output);
void update_learning_rate(Network* network, size_t epoch_count);
void shake_weights(Network* network);

double error_total(Network* network, Vector* training_data[], Vector* teaching_data[], size_t training_set_size);
double error_function(double training_point_true, double network_output);

// ------------------- MAIN ----------------- //
// ------------------------------------------ //
int main(){
    srand((unsigned int)time(NULL));
    
    Vector* training_data[MAX_INPUT_LENGHT]; Vector* teaching_data[MAX_INPUT_LENGHT];
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        training_data[i] = new_vec(DIMENSION_INPUT+1);
        teaching_data[i] = new_vec(DIMENSION_OUTPUT);
    }
    size_t TRAINING_SET_SIZE = read_input(training_data, teaching_data);
    
    Vector** training_set_70 = NULL; Vector** teaching_set_70 = NULL; Vector** validation_set_30 = NULL; Vector** teaching_set_30 = NULL;
    size_t training_set_70_size = split_training_data(training_data, teaching_data, &training_set_70,
                                                       &teaching_set_70, &validation_set_30, &teaching_set_30,
                                                       TRAINING_SET_SIZE, 0.7);
    
    size_t validation_set_30_size = TRAINING_SET_SIZE - training_set_70_size;
    // in_layer, out_layer, hid_layer_count, hid_layers
    Network* network = new_network(DIMENSION_INPUT, DIMENSION_OUTPUT, 2, 7, 7);
//    print_network(network);
    
    Vector*** best_weights = malloc((network->hidden_layers_count+1) * sizeof(Vector**));
    for (size_t layer = 0; layer < network->hidden_layers_count; layer++) {
        best_weights[layer] = malloc(network->hidden_layers[layer]->size * sizeof(Vector*));
        for (size_t neuron_id = 0; neuron_id < network->hidden_layers[layer]->size; neuron_id++) {
            best_weights[layer][neuron_id] = new_vec(network->hidden_layers[layer]->neurons[neuron_id]->weights->length);
        }
    }
    best_weights[network->hidden_layers_count] = malloc(network->output_layer->size * sizeof(Vector*));
    for (size_t neuron_id = 0; neuron_id < network->output_layer->size; neuron_id++) {
        best_weights[network->hidden_layers_count][neuron_id] = new_vec(network->output_layer->neurons[neuron_id]->weights->length);
    }
    
    time_t time_at_beginning = time(0);
    //early stopping if nosy data: TODO: implement define capsulation
    double total_error_training_old = FLOAT_MAX;
    double total_error_training = error_total(network, training_set_70, teaching_set_70, training_set_70_size);
    double minimum_error_training = total_error_training;
    double total_error_validation = error_total(network, validation_set_30, teaching_set_30, validation_set_30_size);
    double minimum_error_validation = total_error_validation;
    double minimum_error_avg = FLOAT_MAX;
    
    double epsilon = 0.001;
    size_t epoch_count = 0;
    
//    FILE* fd = fopen("errors.txt", "w");
    
    while ((time(0) - time_at_beginning) < 60 && total_error_training > epsilon) {
        
//        fprintf(fd, "%.5lf,%.5lf \n", total_error_training, total_error_validation);
        if (minimum_error_training > total_error_training) {
            minimum_error_training = total_error_training;
        }
        if (minimum_error_validation > total_error_validation) {
            minimum_error_validation = total_error_validation;
        }
        if (fabs(total_error_validation - total_error_training) < 0.2 && total_error_validation < 10.0) {
            minimum_error_avg = (total_error_validation + total_error_training)/2.0;
            dump_weights(network, best_weights);
//            printf("Weights dumped at epochs count: %ld\n", epoch_count);
        }
        for (size_t i = 0; i < training_set_70_size; i++) {
            train_network_with_backprop(network, training_set_70[i], teaching_set_70[i]);
        }
        
        if (epoch_count % 1000 == 0) {
            
//            printf("Epochs count: %ld\n", epoch_count);
            if (fabs(total_error_training - total_error_training_old) < 0.01 && total_error_training > 1.0) {
//                printf("Shaking Weights at epoch: %ld\n", epoch_count);
                shake_weights(network);
            }
            total_error_training_old = total_error_training;
//            printf("Total error: %.15lf\n", total_error_training);
//            printf("Total error testing: %.15lf\n", total_error_validation);
            
        }
        epoch_count++;
        update_learning_rate(network, epoch_count);
        scramble_data(training_set_70, teaching_set_70, training_set_70_size);
        
        total_error_validation = error_total(network, validation_set_30, teaching_set_30, validation_set_30_size);
        total_error_training = error_total(network, training_set_70, teaching_set_70, training_set_70_size);
    }
//    fclose(fd);
//    printf("Network training finished with a total error: %.15lf\n", total_error_training);
//    printf("Network training achieved a minimum total error: %.15lf\n", minimum_error_training);
    load_weights(network, best_weights);
//    printf("Total error on validating set: %lf\n", error_total(network, validation_set_30, teaching_set_30, validation_set_30_size));

    test_network(network);
    
    for (size_t layer = 0; layer < network->hidden_layers_count; layer++) {
        for (size_t neuron_id = 0; neuron_id < network->hidden_layers[layer]->size; neuron_id++) {
            delete_vec(best_weights[layer][neuron_id]);
        }
    }
    for (size_t neuron_id = 0; neuron_id < network->output_layer->size; neuron_id++) {
        delete_vec(best_weights[network->hidden_layers_count][neuron_id]);
    }
    
    free(training_set_70);
    free(teaching_set_70);
    free(validation_set_30);
    free(teaching_set_30);
    
    delete_network(network);
    
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        delete_vec(training_data[i]);
        delete_vec(teaching_data[i]);
    }
    
    
    return EXIT_SUCCESS;
}
// ---------------------------------------- //
// --------------- END OF MAIN ------------ //



size_t read_input(Vector* training_data[], Vector* teaching_data[]){
    
    size_t input_size = 0;
    char* input_line = malloc(50*sizeof(char));
    Vector* not_normalised_training[MAX_INPUT_LENGHT];
    Vector* not_normalised_teaching[MAX_INPUT_LENGHT];
    
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        not_normalised_training[i] = new_vec(DIMENSION_INPUT+1); // +bias
        not_normalised_teaching[i] = new_vec(DIMENSION_OUTPUT);
    }
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        if (scanf("%s\n", input_line) == EOF){
            break;
        }
        if (!strncmp(input_line, TERMINATING_STR, (int)strlen(TERMINATING_STR)-1)) {
            break;
        }
#if REGRESSION
        sscanf(input_line, "%lf,%lf\n",
               &not_normalised_training[i]->scalars[0],
               &not_normalised_teaching[i]->scalars[0]);
        not_normalised_training[i]->scalars[1] = BIAS;
#elif CLASSIFICATION
        sscanf(input_line, "%lf,%lf,%lf\n",
               &not_normalised_training[i]->scalars[0],
               &not_normalised_training[i]->scalars[1],
               &not_normalised_teaching[i]->scalars[0]);
        not_normalised_training[i]->scalars[2] = BIAS;
#endif
        input_size++;
    }
    
    double min = FLOAT_MAX;
    double max = FLOAT_MIN;
    for (int i = 0; i < input_size; i++) {
        if (not_normalised_training[i]->scalars[0] > max) {
            max = not_normalised_training[i]->scalars[0];
        }
        if (not_normalised_training[i]->scalars[0] < min) {
            min = not_normalised_training[i]->scalars[0];
        }
    }
    MAX_VALUES_INPUT[0] = max;
    MIN_VALUES_INPUT[0] = min;
    
#if REGRESSION
    min = FLOAT_MAX;
    max = FLOAT_MIN;
    for (int i = 0; i < input_size; i++) {
        if (not_normalised_teaching[i]->scalars[0] > max) {
            max = not_normalised_teaching[i]->scalars[0];
        }
        if (not_normalised_teaching[i]->scalars[0] < min) {
            min = not_normalised_teaching[i]->scalars[0];
        }
    }
    MAX_VALUES_INPUT[1] = max;
    MIN_VALUES_INPUT[1] = min;
#elif CLASSIFICATION
    min = FLOAT_MAX;
    max = FLOAT_MIN;
    for (int i = 0; i < input_size; i++) {
        if (not_normalised_training[i]->scalars[1] > max) {
            max = not_normalised_training[i]->scalars[1];
        }
        if (not_normalised_training[i]->scalars[1] < min) {
            min = not_normalised_training[i]->scalars[1];
        }
    }
    MAX_VALUES_INPUT[1] = max;
    MIN_VALUES_INPUT[1] = min;
#endif
    
    for (int i = 0; i < input_size; i++) {
#if REGRESSION
        
        training_data[i]->scalars[0] = normalise_data(not_normalised_training[i]->scalars[0], MAX_VALUES_INPUT[0], MIN_VALUES_INPUT[0]);
        training_data[i]->scalars[1] = BIAS;//normalised_training->scalars[1];
        teaching_data[i]->scalars[0] = normalise_data(not_normalised_teaching[i]->scalars[0], MAX_VALUES_INPUT[1], MIN_VALUES_INPUT[1]);
#elif CLASSIFICATION
        training_data[i]->scalars[0] = normalise_data(not_normalised_training[i]->scalars[0], MAX_VALUES_INPUT[0], MIN_VALUES_INPUT[0]);
        training_data[i]->scalars[1] = normalise_data(not_normalised_training[i]->scalars[1], MAX_VALUES_INPUT[1], MIN_VALUES_INPUT[1]);
        training_data[i]->scalars[2] = BIAS;//normalised_training->scalars[2];
        teaching_data[i]->scalars[0] = not_normalised_teaching[i]->scalars[0];
#endif
    }
    
    for (int i = 0; i < MAX_INPUT_LENGHT; i++) {
        delete_vec(not_normalised_training[i]);
        delete_vec(not_normalised_teaching[i]);
    }
    
    free(input_line);
    return input_size;
}

void test_network(Network* network){
    char* input_line = malloc(50*sizeof(char));
    Vector* testing_point_in = new_vec(DIMENSION_INPUT+1);
    Vector* testing_point_out = new_vec(DIMENSION_OUTPUT);
    size_t test_set_size = 0;
    
    while (scanf("%s\n", input_line) != EOF) {
        test_set_size++;
#if REGRESSION
        sscanf(input_line, "%lf\n", &testing_point_in->scalars[0]);
        testing_point_in->scalars[0] = normalise_data(testing_point_in->scalars[0], MAX_VALUES_INPUT[0], MIN_VALUES_INPUT[0]);
        testing_point_in->scalars[DIMENSION_INPUT] = BIAS;
        testing_point_out = compute_output_network(network, testing_point_in);
        printf("%.7lf\n", denormalise_data(testing_point_out->scalars[0], MAX_VALUES_INPUT[1], MIN_VALUES_INPUT[1]));
#elif CLASSIFICATION
        sscanf(input_line, "%lf,%lf\n", &testing_point_in->scalars[0], &testing_point_in->scalars[1]);
        testing_point_in->scalars[DIMENSION_INPUT] = BIAS;
        testing_point_out = compute_output_network(network, testing_point_in);
        if (testing_point_out->scalars[0] >= 0) {
            printf("+1\n");
        }else{
            printf("-1\n");
        }
#endif
        delete_vec(testing_point_out);
    }
    
    delete_vec(testing_point_in);
    free(input_line);
}

void train_network_with_backprop(Network* network, Vector* training_point, Vector* teaching_point){
    Vector* network_output = compute_output_network(network, training_point);
//    printf("\t Network output: %.10lf\tTrue value: %lf\n", network_output->scalars[0], teaching_point->scalars[0]);
    update_parameters(network, teaching_point, network_output);
    delete_vec(network_output);
}

Vector* compute_output_network(Network* network, Vector* input){
//    printf("Not-normalised input to network: ");
//    print_vector(input);
    Vector* prev_layer_output = compute_output_layer(network->input_layer, input);
//    printf("Normalised input to network: ");
//    print_vector(prev_layer_output);
    for (size_t layer_id = 0; layer_id < network->hidden_layers_count; layer_id++) {
        Vector* input_vector_for_next_layer = compute_output_layer(network->hidden_layers[layer_id], prev_layer_output);
        delete_vec(prev_layer_output);
        prev_layer_output = input_vector_for_next_layer;
//        printf("Output from %ld hidden layer: ", layer_id);
//        print_vector(prev_layer_output);
    }
    Vector* network_output = compute_output_layer(network->output_layer, prev_layer_output);
//    printf("Not-denormalised output from network: ");
//    print_vector(network_output);
    delete_vec(prev_layer_output);
    return network_output;
}

Vector* compute_output_layer(Layer* layer, Vector* input){
#if ASSERTIONS_ENABLED
    if (layer->id == 0) {
        assert(input->length-1 == DIMENSION_INPUT && layer->size == DIMENSION_INPUT); // -bias
    } else {
        assert(input->length == layer->neurons[0]->weights->length); //+bias
    }
#endif
    Vector* layer_output;
    if (layer->id == HIDDEN_LAYERS_COUNT+1) {
        layer_output = new_vec(layer->size);
    } else {
        layer_output = new_vec(layer->size+1);
        layer_output->scalars[layer->size] = BIAS;
    }
    for (size_t in_neuron_id = 0; in_neuron_id < layer->size; in_neuron_id++) {
        layer_output->scalars[in_neuron_id] =
            layer->neurons[in_neuron_id]->neuron_function(layer->neurons[in_neuron_id], input);
    }
//    print_detailed_layer(layer);
    return layer_output;
}

double error_total(Network* network, Vector* training_data[], Vector* teaching_data[], size_t training_data_size){
    double err_accumulator = 0.0;
    for (int i = 0; i < training_data_size; i++) {
        Vector* network_output = compute_output_network(network, training_data[i]);
        err_accumulator += error_function(teaching_data[i]->scalars[0], network_output->scalars[0]);
        delete_vec(network_output);
    }
    
    double squared_weights_sum = 0.0;
    size_t weight_count = 0;
    for (Layer* layer = network->output_layer; layer->id != 0; layer = layer->prev_layer) {
        for (size_t neuron_id = 0; neuron_id < layer->size; neuron_id++) {
            for (size_t weight_id = 0; weight_id < layer->neurons[neuron_id]->weights->length; weight_id++) {
                squared_weights_sum += SQR(layer->neurons[neuron_id]->weights->scalars[weight_id]);
                weight_count++;
            }
        }
    }
    return PERFORMANCE_RATIO * err_accumulator + (1.0 - PERFORMANCE_RATIO) * squared_weights_sum / (double)weight_count;
}

double error_function(double teaching_point, double network_output){
    return 0.5 * SQR(teaching_point - network_output);
}

void update_parameters(Network* network, Vector* teaching_output, Vector* network_output){
    // compute deltas for output layer:
    for (size_t neuron_id = 0; neuron_id < network->output_layer->size; neuron_id++) {
        network->output_layer->deltas->scalars[neuron_id] =
            (network->output_layer->neurons[neuron_id]->stored_output - teaching_output->scalars[neuron_id]) *
            network->output_layer->neurons[neuron_id]->neuron_function_derivative(network->output_layer->neurons[neuron_id], NULL);
    }
    // compute deltas for all other layers:
    for (Layer* current_layer = network->hidden_layers[network->hidden_layers_count-1]; current_layer->id != 0; current_layer = current_layer->prev_layer) {
        for (size_t neuron_id = 0; neuron_id < current_layer->size; neuron_id++) {
            double sum_weighted_deltas_next_layer = 0.0;
            for (size_t neuron_id_next_layer = 0; neuron_id_next_layer < current_layer->next_layer->size; neuron_id_next_layer++) {
                sum_weighted_deltas_next_layer +=
                    current_layer->next_layer->deltas->scalars[neuron_id_next_layer] *
                    current_layer->next_layer->neurons[neuron_id_next_layer]->weights->scalars[neuron_id];
            }
            current_layer->deltas->scalars[neuron_id] =
                sum_weighted_deltas_next_layer *
                current_layer->neurons[neuron_id]->neuron_function_derivative(current_layer->neurons[neuron_id], NULL);
        }
    }
    
    // update all weights in all layers:
    for (Layer* current_layer = network->output_layer; current_layer->id != 0; current_layer = current_layer->prev_layer) {
        for (size_t neuron_id = 0; neuron_id < current_layer->size; neuron_id++) {
            for (size_t weight_id = 0; weight_id < current_layer->neurons[neuron_id]->weights->length; weight_id++) {
                current_layer->neurons[neuron_id]->weights->scalars[weight_id] +=
                    -1.0 *
                    current_layer->learning_rate *
                    current_layer->deltas->scalars[neuron_id] *
                    (weight_id == (current_layer->neurons[neuron_id]->weights->length - 1) ?
                        BIAS :
                        current_layer->prev_layer->neurons[weight_id]->stored_output);
            }
        }
    }
}

void update_learning_rate(Network* network, size_t epoch_count){
    for (Layer* layer = network->output_layer; layer->id != 0; layer = layer->prev_layer) {
        layer->learning_rate = MAX_LEARNING_RATE * exp(epoch_count/LEARNING_RATE_MODIFIER_TRESHOLD);
    }
}

void shake_weights(Network* network){
    for (Layer* layer = network->output_layer; layer->id != 0; layer = layer->prev_layer) {
        for (size_t neuron_id = 0; neuron_id < layer->size; neuron_id++) {
            for (size_t weight_id = 0; weight_id < layer->neurons[neuron_id]->weights->length; weight_id++) {
                layer->neurons[neuron_id]->weights->scalars[weight_id] *=
                    1.0 + ((rand()&1)? 1.0 : -1.0) * WEIGHTS_SHAKING_FACTOR;
            }
        }
    }
}

double normalise_data(double value, double max, double min){
#if ASSERTIONS_ENABLED
    assert(max != min);
#endif
    double to_normalise_value = (2.0 * value - max - min) / (max - min);
    return to_normalise_value;
}

double denormalise_data(double value, double max, double min){
    double denormalised_output = (max*(value + 1.0) - min*(value - 1.0))/2.0;
    return denormalised_output;
}

void scramble_data(Vector* training_data[], Vector* teaching_data[], size_t dataset_size){
    for (size_t i = 0; i < dataset_size/2; i++) {
        size_t random_index = rand() % (dataset_size/2);
        Vector* temp_training = training_data[random_index];
        Vector* temp_vec_teaching = teaching_data[random_index];
        
        training_data[random_index] = training_data[random_index+dataset_size/2];
        training_data[random_index+dataset_size/2] = temp_training;
        
        teaching_data[random_index] = teaching_data[random_index+dataset_size/2];
        teaching_data[random_index+dataset_size/2] = temp_vec_teaching;
    }
}

size_t split_training_data(Vector** training_data, Vector** teaching_data, Vector*** training_portion,
                           Vector*** training_data_teaching, Vector*** testing_portion, Vector*** testing_portion_teaching,
                           size_t training_data_size, double ratio){
    size_t training_portion_size = (size_t)(ratio * (double)training_data_size);
    size_t testing_portion_size = training_data_size - training_portion_size;
    if (*training_portion != NULL) free(*training_portion);
    if (*training_data_teaching != NULL) free(*training_data_teaching);
    if (*testing_portion_teaching != NULL) free(*testing_portion_teaching);
    if (*testing_portion != NULL) free(*testing_portion);
    *training_portion = (Vector**)malloc(training_portion_size * sizeof(Vector*));
    *training_data_teaching = (Vector**)malloc(training_portion_size * sizeof(Vector*));
    *testing_portion = (Vector**)malloc(testing_portion_size * sizeof(Vector*));
    *testing_portion_teaching = (Vector**)malloc(testing_portion_size * sizeof(Vector*));
    
    bool picked[training_data_size];
    memset(picked, false, training_data_size*sizeof(bool));
    for (size_t i = 0; i < training_portion_size; i++) {
        size_t random_index = (size_t)rand() % training_data_size;
        // check if already picked and skip until unpicked is found or all are picked
        for (int j = 0; j < training_data_size && picked[random_index]; j++) {
            random_index++;
            random_index %= training_data_size;
        }
        if (picked[random_index]) {
            // no more unpicked indices
            training_portion_size = i;
            break;
        }
        (*training_portion)[i] = training_data[random_index];
        (*training_data_teaching)[i] = teaching_data[random_index];
        picked[random_index] = true;
    }
    size_t index_testing_portion = 0;
    for (size_t i = 0; i < testing_portion_size; i++) {
        if (picked[index_testing_portion]) {
            i--;
            index_testing_portion++;
            continue;
        }
        (*testing_portion)[i] = training_data[index_testing_portion];
        (*testing_portion_teaching)[i] = training_data[index_testing_portion];
        index_testing_portion++;
    }
    return training_portion_size;
}

void dump_weights(Network* network, Vector*** best_weights){
    for (size_t layer = 0; layer < network->hidden_layers_count; layer++) {
        for (size_t neuron_id = 0; neuron_id < network->hidden_layers[layer]->size; neuron_id++) {
            for (size_t weight_id = 0; weight_id < network->hidden_layers[layer]->neurons[neuron_id]->weights->length; weight_id++) {
                best_weights[layer][neuron_id]->scalars[weight_id] =
                    network->hidden_layers[layer]->neurons[neuron_id]->weights->scalars[weight_id];
            }
        }
    }
    for (size_t neuron_id_out = 0; neuron_id_out < network->output_layer->size; neuron_id_out++) {
        for (size_t weight_id_out = 0; weight_id_out < network->output_layer->neurons[neuron_id_out]->weights->length; weight_id_out++) {
            best_weights[network->hidden_layers_count][neuron_id_out]->scalars[weight_id_out] =
                network->output_layer->neurons[neuron_id_out]->weights->scalars[weight_id_out];
        }
    }
}

void load_weights(Network* network, Vector*** best_weights){
    for (size_t layer = 0; layer < network->hidden_layers_count; layer++) {
        for (size_t neuron_id = 0; neuron_id < network->hidden_layers[layer]->size; neuron_id++) {
            for (size_t weight_id = 0; weight_id < network->hidden_layers[layer]->neurons[neuron_id]->weights->length; weight_id++) {
                network->hidden_layers[layer]->neurons[neuron_id]->weights->scalars[weight_id] =
                    best_weights[layer][neuron_id]->scalars[weight_id];
            }
        }
    }
    for (size_t neuron_id_out = 0; neuron_id_out < network->output_layer->size; neuron_id_out++) {
        for (size_t weight_id_out = 0; weight_id_out < network->output_layer->neurons[neuron_id_out]->weights->length; weight_id_out++) {
            network->output_layer->neurons[neuron_id_out]->weights->scalars[weight_id_out] =
                best_weights[network->hidden_layers_count][neuron_id_out]->scalars[weight_id_out];
        }
    }
}
