//
//  Kohonen self-organazing maps as a good TSP solver
//  Input: a list of cities (ids and x,y coordinates)
//  Output: ordered list of city ids
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
#define DATA_LOGGING false

#define SQR(x) ((x)*(x))

#define MAX_INPUT_LENGTH 1001
#define SPACE_DIMENSIONALITY 2

#define PERCENT_BIAS_NEURON_COUNT 0.0
#define INITIAL_NEIGHBOURHOOD_RADIUS_IN_PERCENT 15.0
#define INITIAL_LEARNING_RATE 0.99

//-----------------------------------------------//
//---------- CUSTOM TYPES -----------------------//
//-----------------------------------------------//
//----------- Vector structure ---------------//
typedef struct Vector{
    double* scalars;
    size_t length;
} Vector;

Vector* new_vector(size_t len){
    Vector* vec = malloc(sizeof(Vector));
    vec->length = len;
    vec->scalars = malloc(len*sizeof(double));
    for (size_t i = 0; i < len; i++) {
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

//----------- City structure ---------------//
typedef struct City {
    Vector* coordinates;
    int id;
    int affiliation_id;
    bool visited;
} City;

City* new_city(int id){
    City* city = malloc(sizeof(City));
    city->id = id;
    city->affiliation_id = -1;
    city->visited = false;
    city->coordinates = new_vector(SPACE_DIMENSIONALITY);
    return city;
}

void print_city(City* city){
    printf("CityID: %d, visited: %d, coordinates: ", city->id, city->visited);
    print_vector(city->coordinates);
}

void delete_city(City* city){
    delete_vector(city->coordinates);
    free(city);
}

//----------- Map structure ---------------//
typedef struct Map {
    City** cities;
    size_t size;
    size_t total_size;
} Map;

Map* new_map(size_t size){
    Map* map = malloc(sizeof(Map));
    map->size = 0;
    map->total_size = size;
    map->cities = malloc(size * sizeof(City*));
    for (size_t i = 0; i < size; i++) {
        map->cities[i] = new_city((int)i);
    }
    return map;
}

void print_map(Map* map){
    printf("Map of %ld cities:\n", map->size);
    for (size_t i = 0; i < map->size; i++) {
        printf("\t");
        print_city(map->cities[i]);
    }
}

void delete_map(Map* map){
    for (size_t i = 0; i < map->total_size; i++) {
        delete_city(map->cities[i]);
    }
    free(map->cities);
    free(map);
}

//------------- Neuron structure -------------//
typedef struct Neuron Neuron;

struct Neuron {
    Vector* position;
    int id;
    Neuron** neighbours;
    size_t neighbourhood_size;
    bool is_winner;
};

Neuron* new_neuron(int id){
    Neuron* neuron = malloc(sizeof(Neuron));
    neuron->id = id;
    neuron->is_winner = false;
    neuron->neighbourhood_size = 0;
    neuron->neighbours = NULL;
    neuron->position = new_vector(SPACE_DIMENSIONALITY);
    return neuron;
}

void print_neuron(Neuron* neuron){
    printf("NeuronID: %d", neuron->id);
    if (neuron->neighbourhood_size > 0) {
        printf(", NeighboursIDs: ");
        for (size_t i = 0; i < neuron->neighbourhood_size; i++) {
            printf("%d,", neuron->neighbours[i]->id);
        }
    }
    printf(" Position: ");
    print_vector(neuron->position);
}

void delete_neuron(Neuron* neuron){
    delete_vector(neuron->position);
    free(neuron);
}

// ------- Self-Organising Map structure ------//
typedef struct Network {
    Neuron** neurons;
    size_t size;
} Network;

Network* new_network(size_t size, size_t neighbourhood_radius_in_prcnt){
    Network* network = malloc(sizeof(Network));
    network->size = size;
    network->neurons = malloc(size * sizeof(Neuron*));
    for (size_t i = 0; i < size; i++) {
        network->neurons[i] = new_neuron((int)i);
        for (int d = 0; d < SPACE_DIMENSIONALITY; d++) {
            double random_value = (double)rand() / ((double) RAND_MAX);
            network->neurons[i]->position->scalars[d] = (rand()&1? 1.0 : -1.0) * random_value;
        }
    }
    size_t neighbours_count = 2 * (size_t)((neighbourhood_radius_in_prcnt/100.0) * (size-1));
    for (int i = 0; i < size; i++) {
        network->neurons[i]->neighbourhood_size = neighbours_count;
        network->neurons[i]->neighbours = malloc(neighbours_count * sizeof(Neuron*));
        for (int nb = 0; nb < neighbours_count/2; nb++) {
            size_t id_left = (i - nb - 1 >= 0 ? i - nb - 1: size + i - nb - 1);
            size_t id_right = (i + nb + 1) % size;
            if (id_left != id_right && id_left != i && id_right != i) {
                network->neurons[i]->neighbours[2*nb] = network->neurons[id_left];
                network->neurons[i]->neighbours[2*nb+1] = network->neurons[id_right];
            } else {
                break;
            }
        }
    }
    return network;
}

void print_network(Network* network){
    printf("Network size: %ld\n", network->size);
    for (size_t i = 0; i < network->size; i++) {
        printf("\t");
        print_neuron(network->neurons[i]);
    }
}

void delete_network(Network* network){
    for (size_t i = 0; i < network->size; i++) {
        delete_neuron(network->neurons[i]);
    }
    free(network->neurons);
    free(network);
}

//--------------------------------------------//
//---------- HELPER FUNCTIONS & CO. ----------//
//--------------------------------------------//
size_t read_input(Map* cities);
void print_output(Map* cities);

void scramble_map_ordering(Map* map);
void normalise_space(Map* cities);
void denormalise_space(Map* cities);
void denormalise_network(Network* network);
double normalise_value(double value, double min, double max);
double denormalise_value(double value, double min, double max);


static double MAX_VALUES[SPACE_DIMENSIONALITY] = {FLOAT_MIN, FLOAT_MIN};
static double MIN_VALUES[SPACE_DIMENSIONALITY] = {FLOAT_MAX, FLOAT_MAX};
static size_t CITIES_COUNT = 0;
static size_t ITERATION_STEP = 0;
static double LEARNING_RATE = INITIAL_LEARNING_RATE;
#if DATA_LOGGING
FILE* fd = NULL;
#endif
//--------------------------------------------//
//------ SELF-ORGANISING MAPS FUNCTIONS ------//
//--------------------------------------------//
void train_network(Network* network, Map* map);
Neuron* find_closest(Network* network, City* city);
void update_positions(Neuron* winner, City* current_city);

double compute_distance(const Vector* point1, const Vector* point2);
double neighbourhood_function(int distance_from_winner);
void adjust_learning_rate();

//--------------------------------------------//
//------------------ MAIN --------------------//
//--------------------------------------------//
int main(){
    srand((unsigned int)time(NULL));
    
    Map* cities = new_map(MAX_INPUT_LENGTH);
    CITIES_COUNT = read_input(cities);
#if DATA_LOGGING
    fd = fopen("som.txt", "w");
    if (fd == NULL) {
        printf("error opening file.");
    }
#endif
    normalise_space(cities);
//    printf("%ld", CITIES_COUNT);
//    fflush(stdout);
    Network* network = new_network(CITIES_COUNT + (size_t)(PERCENT_BIAS_NEURON_COUNT/100.0 * CITIES_COUNT), INITIAL_NEIGHBOURHOOD_RADIUS_IN_PERCENT);
    
    time_t time_at_beginning = time(0);
    while (time(0) - time_at_beginning < 20) {
        train_network(network, cities);

    }
//    denormalise_network(network);
//    print_network(network);
#if DATA_LOGGING
    for (int i = 0; i < network->size; i++) {
        if (true) {
            printf("%d,%lf,%lf\n", network->neurons[i]->id,
                   denormalise_value(network->neurons[i]->position->scalars[0], MIN_VALUES[0], MAX_VALUES[0]),
                   denormalise_value(network->neurons[i]->position->scalars[1], MIN_VALUES[1], MAX_VALUES[1]));
            fprintf(fd, "%d,%lf,%lf\n", network->neurons[i]->id,
                    denormalise_value(network->neurons[i]->position->scalars[0], MIN_VALUES[0], MAX_VALUES[0]),
                    denormalise_value(network->neurons[i]->position->scalars[1], MIN_VALUES[1], MAX_VALUES[1]));
            fflush(fd);
            printf("iterations: %ld\n", ITERATION_STEP);
            fflush(stdout);
        }
    }
#endif
    print_output(cities);
    
    delete_network(network);
    delete_map(cities);
#if DATA_LOGGING
    fclose(fd);
#endif
    return EXIT_SUCCESS;
}

/*
* Read the list of cities and coordinates.
*/
size_t read_input(Map* map){
    size_t cities_count = 0;
    char* input_line = malloc(50*sizeof(char));

    for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
        if (scanf("%s\n", input_line) == EOF){
            break;
        }
        sscanf(input_line, "%d,%lf,%lf\n", &map->cities[i]->id, &map->cities[i]->coordinates->scalars[0], &map->cities[i]->coordinates->scalars[1]);
        for (size_t d = 0; d < SPACE_DIMENSIONALITY; d++) {
            if (map->cities[i]->coordinates->scalars[d] > MAX_VALUES[d]) {
                MAX_VALUES[d] = map->cities[i]->coordinates->scalars[d];
            }
            if (map->cities[i]->coordinates->scalars[d] < MIN_VALUES[d]) {
                MIN_VALUES[d] = map->cities[i]->coordinates->scalars[d];
            }
        }
        map->size++;
        cities_count++;
    }
    
    free(input_line);
    return cities_count;
}

/*
* Print the path according to the neuron affiliation of each city.
*/
void print_output(Map* map){
    for (int j = 0; j < map->size; j++) {
        for (int i = 0; i < map->size; i++) {
            if (map->cities[i]->affiliation_id == j) {
                printf("%d\n", map->cities[i]->id);
            }
        }
    }
}

/*
* Normalise the input domain space.
*/
void normalise_space(Map* map){
    for (size_t i = 0; i < map->size; i++) {
        for (size_t d = 0; d < SPACE_DIMENSIONALITY; d++) {
            map->cities[i]->coordinates->scalars[d] = normalise_value(map->cities[i]->coordinates->scalars[d], MIN_VALUES[d], MAX_VALUES[d]);
        }
    }
}

/*
* Denormalise the input space.
*/
void denormalise_space(Map* map){
    for (size_t i = 0; i < map->size; i++) {
        for (size_t d = 0; d < SPACE_DIMENSIONALITY; d++) {
            map->cities[i]->coordinates->scalars[d] = denormalise_value(map->cities[i]->coordinates->scalars[d], MIN_VALUES[d], MAX_VALUES[d]);
        }
    }
}

/*
* Denormalise the SOM output space 
*/
void denormalise_network(Network* network){
    for (size_t i = 0; i < network->size; i++) {
        for (size_t d = 0; d < SPACE_DIMENSIONALITY; d++) {
            network->neurons[i]->position->scalars[d] = denormalise_value(network->neurons[i]->position->scalars[d], MIN_VALUES[d], MAX_VALUES[d]);
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

/*
* Shuffle the order of iterations over the cities
*/
void scramble_map_ordering(Map* map){
    for (size_t i = 0; i < map->size/2; i++) {
        size_t random_index = rand() % (map->size/2);
        City* temp_city_holder = map->cities[random_index];
        map->cities[random_index] = map->cities[random_index+map->size/2];
        map->cities[random_index+map->size/2] = temp_city_holder;
    }
}

/*
* For each city update the posision of the best matching unit.
*/
void train_network(Network* network, Map* map){
    scramble_map_ordering(map);
    for (size_t city_id = 0; city_id < map->size; city_id++) {
        Neuron* winner_neuron = find_closest(network, map->cities[city_id]);
        update_positions(winner_neuron, map->cities[city_id]);
    }
    ITERATION_STEP++;
#if DATA_LOGGING
    if (ITERATION_STEP % 100000 == 0) {
        printf("Iterationstep: %ld\n", ITERATION_STEP);
        printf("Learningrate: %lf\n", LEARNING_RATE);
    }
#endif    
}

/*
* Compute the best matching unit according to the distance measure
*/
Neuron* find_closest(Network* network, City* city){
    double least_distance = FLOAT_MAX;
    int winner_neuron_id = -1;
    for (int neuron_id = 0; neuron_id < network->size; neuron_id++) {
        double current_distance = compute_distance(network->neurons[neuron_id]->position, city->coordinates);
        if (current_distance < least_distance) {
            least_distance = current_distance;
            winner_neuron_id = neuron_id;
        }
    }
#if ASSERTIONS_ENABLED
    if (winner_neuron_id == -1) {
        assert(0);
    }
#endif
    network->neurons[winner_neuron_id]->is_winner = true;
    city->affiliation_id = winner_neuron_id;
    return network->neurons[winner_neuron_id];
}

/*
* Compute euclidean distance in 2d space
*/
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

/*
* Update the position of the best matching unit and its neighbours within the 
* local neighbourhood radius.
*/
void update_positions(Neuron* winner_neuron, City* current_city){
    adjust_learning_rate();
    for (size_t d = 0; d < SPACE_DIMENSIONALITY; d++) {
        winner_neuron->position->scalars[d] += LEARNING_RATE * (current_city->coordinates->scalars[d] - winner_neuron->position->scalars[d]);
    }
//    printf("winnerID: %d\n", winner_neuron->id);
    for (int i = 0; i < winner_neuron->neighbourhood_size; i++) {
        double proximity_scaling_factor = neighbourhood_function(i/2+1);
//        printf("\tProximity scaling: %lf for neighbourID %d\n", proximity_scaling_factor, winner_neuron->neighbours[i]->id);
        for (size_t d = 0; d < SPACE_DIMENSIONALITY; d++) {
            winner_neuron->neighbours[i]->position->scalars[d] +=
                LEARNING_RATE *
                (current_city->coordinates->scalars[d] - winner_neuron->neighbours[i]->position->scalars[d]) *
                proximity_scaling_factor;
        }
    }
}

/*
* Compute the distance scaling factor of each neighbour. Use Gaussian neighbourhood
*/
double neighbourhood_function(int distance_from_winner){
    double CONST = 1.0;
    double SIGMA = 5.0 * exp(ITERATION_STEP/-2000.0);
    return CONST * exp(-SQR(distance_from_winner) / (2*SQR(SIGMA > 0.01 ? SIGMA : 0.01)));
}

/*
* Update the learning rate (decaying exponentially)
*/
void adjust_learning_rate(){
    if (LEARNING_RATE >= 0.001) {
        LEARNING_RATE = INITIAL_LEARNING_RATE * exp(ITERATION_STEP/-50000.0);
    }
}

