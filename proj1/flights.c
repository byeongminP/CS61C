/*
 * CS61C Summer 2019
 * Name: Byeong Min Park
 * Login: cs61c-acw
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "flights.h"
#include "flight_structs.h"
#include "timeHM.h"

/*
 *  This should be called if memory allocation failed.
 */
static void allocation_failed(void) {
  fprintf(stderr, "Out of memory.\n");
  exit(EXIT_FAILURE);
}


/*
 *  Creates and initializes a flight system, which stores the flight schedules of several airports.
 *  Returns a pointer to the system created.
 */
flightSys_t* createSystem(void) {
  // allocate memory for flight system
  flightSys_t* sys = malloc(sizeof(flightSys_t));
  if (sys == NULL) {
    allocation_failed();
  }
  sys->airports = NULL;

  return sys;
}


/*
 *   Given a destination airport, departure and arrival times, and a cost, return a pointer to new flight_t.
 *
 *   Note that this pointer must be available to use even after this function returns.
 *   (What does this mean in terms of how this pointer should be instantiated?)
 *   Additionally you CANNOT assume that the `departure` and `arrival` pointers will persist after this function completes.
 *   (What does this mean about copying dep and arr?)
 */

flight_t* createFlight(airport_t* destination, timeHM_t* departure, timeHM_t* arrival, int cost) {
  if (destination == NULL || departure == NULL || arrival == NULL || cost <= 0) {
    return NULL;
  } else if (!isAfter(arrival, departure)) {
    return NULL;
  }
  flight_t* flight = malloc(sizeof(flight_t));
  if (flight == NULL) {
    allocation_failed();
  }
  // allocate more memory
  timeHM_t* dep_copy = malloc(sizeof(timeHM_t));
  timeHM_t* arr_copy = malloc(sizeof(timeHM_t));
  if (dep_copy == NULL || arr_copy == NULL) {
    free(flight);
    allocation_failed();
  }
  // make a copy of dep and arr
  memcpy(dep_copy, departure, sizeof(timeHM_t));
  memcpy(arr_copy, arrival, sizeof(timeHM_t));
  // assign variables
  flight->destination = destination;
  flight->departure = dep_copy;
  flight->arrival = arr_copy;
  flight->cost = cost;
  flight->next = NULL;

  return flight;
}

/*
 *  Frees all memory allocated for a single flight. You may or may not decide
 *  to use this in delete system but you must implement it.
 */
void deleteFlight(flight_t* flight) {
  // null check
  if (flight == NULL) {
    return;
  }
  // free components
  flight->destination = NULL;
  free(flight->departure);
  free(flight->arrival);
  // now delete flight
  free(flight);
}

/*
 *  Helper function to simplify code from deleteSystem; takes advantage of deleteFlight.
 */
void deleteAirport(airport_t* airport) {
  // null check
  if (airport == NULL) {
    return;
  }
  // start deleting components
  free(airport->name);
  // temporary allocation to hold flights
  flight_t* temp;
  // delete an array of flights using a loop and temp
  while (airport->flights != NULL) {
    temp = airport->flights->next;
    deleteFlight(airport->flights);
    airport->flights = temp;
  }
  // no more flights, now delete airport
  free(airport);
}

/*
 *  Frees all memory associated with this system; that's all memory you dynamically allocated in your code.
 */
void deleteSystem(flightSys_t* system) {
  // null check
  if (system == NULL) {
    return;
  }
  // temporary allocation to hold airports
  airport_t* temp;
  // delete an array of airports using a loop and temp
  while (system->airports != NULL) {
    temp = system->airports->next;
    deleteAirport(system->airports);
    system->airports = temp;
  }
  // no more airports, now delete system
  free(system);
}

/*
 *  Adds a airport with the given name to the system. You must copy the string and store it.
 *  Do not store `name` (the pointer) as the contents it point to may change.
 */
void addAirport(flightSys_t* system, char* name) {
  // null check
  if (system == NULL || name == NULL) {
    return;
  }
  // temporary allocation
  airport_t* temp = system->airports;
  airport_t* new_node = malloc(sizeof(airport_t));
  if (new_node == NULL) {
    allocation_failed();
  }
  // fill in information
  new_node->name = malloc(strlen(name) + 1);
  if (new_node->name == NULL) {
    free(new_node);
    allocation_failed();
  }
  strcpy(new_node->name, name); // copy the name in case the contents change
  new_node->next = NULL;
  new_node->flights = NULL;
  // another allocation for accessing linked list
  if (temp == NULL) {
    system->airports = new_node;
    return;
  }
  while (temp->next != NULL) {
    temp = temp->next;
  }
  temp->next = new_node;
}


/*
 *  Returns a pointer to the airport with the given name.
 *  If the airport doesn't exist, return NULL.
 */
airport_t* getAirport(flightSys_t* system, char* name) {
  // null check
  if (system == NULL || name == NULL) {
    return NULL;
  }
  // temporary variable to copy the memory of airport
  airport_t* temp = system->airports;
  // keep moving on until name is found
  while (temp != NULL) {
    if (strcmp(temp->name, name) == 0) {
      return temp;
    }
    temp = temp->next;
  }
  // return null if the airport does not exist
  return NULL;
}


/*
 *  Print each airport name in the order they were added through addAirport, one on each line.
 *  Make sure to end with a new line. You should compare your output with the correct output
 *  in `flights.out` to make sure your formatting is correct.
 */
void printAirports(flightSys_t* system) {
  // null check
  if (system == NULL || system->airports == NULL) {
    return;
  }
  // copy airports from system
  airport_t* temp = system->airports;
  while (temp != NULL) {
    printf("%s\n", temp->name);
    temp = temp->next;
  }
}


/*
 *  Adds a flight to source's schedule, stating a flight will leave to destination at departure time and arrive at arrival time.
 */
void addFlight(airport_t* source, airport_t* destination, timeHM_t* departure, timeHM_t* arrival, int cost) {
  // null check
  if (source == NULL || destination == NULL || departure == NULL || arrival == NULL || cost <= 0) {
    return;
  }
  // temporary variable
  flight_t* temp = source->flights;
  flight_t* new_flight = createFlight(destination, departure, arrival, cost);
  if (new_flight == NULL) {
    return;
  }
  // same as adding airport
  if (temp == NULL) {
    source->flights = new_flight;
    return;
  }
  while (temp->next != NULL) {
    temp = temp->next;
  }
  temp->next = new_flight;
}


/*
 *  Prints the schedule of flights of the given airport.
 *
 *  Prints the airport name on the first line, then prints a schedule entry on each
 *  line that follows, with the format: "destination_name departure_time arrival_time $cost_of_flight".
 *
 *  You should use `printTime()` (look in `timeHM.h`) to print times, and the order should be the same as
 *  the order they were added in through addFlight. Make sure to end with a new line.
 *  You should compare your output with the correct output in flights.out to make sure your formatting is correct.
 */
void printSchedule(airport_t* airport) {
  // null check
  if (airport == NULL || airport->flights == NULL) {
    return;
  }
  // print airport name
  printf("%s\n", airport->name);
  // temporary pointer to array of flights
  flight_t* temp = airport->flights;
  // iterate through the array
  while (temp != NULL) {
    printf("%s ", temp->destination->name);
    printTime(temp->departure);
    printf(" ");
    printTime(temp->arrival);
    printf(" $%d\n", temp->cost);
    temp = temp->next;
  }
}


/*
 *   Given a source and destination airport, and the time now, finds the next flight to take based on the following rules:
 *   1) Finds the earliest arriving flight from source to destination that departs after now.
 *   2) If there are multiple earliest flights, take the one that costs the least.
 *
 *   If a flight is found, you should store the flight's departure time, arrival time, and cost in the `departure`, `arrival`,
 *   and `cost` params and return true. Otherwise, return false.
 *
 *   Please use the functions `isAfter()` and `isEqual()` from `timeHM.h` when comparing two timeHM_t objects and compare
 *   the airport names to compare airports, not the pointers.
 */
bool getNextFlight(airport_t* source, airport_t* destination, timeHM_t* now, timeHM_t* departure, timeHM_t* arrival,
                   int* cost) {
  // null check
  if (source == NULL || destination == NULL || now == NULL) {
    return false;
  }
  // allocate temporary variables
  int cheapest = 99999;
  flight_t* temp = source->flights;
  if (temp == NULL) {
    return false;
  }
  timeHM_t* best_arrival = source->flights->arrival;
  if (best_arrival == NULL) {
    return false;
  }
  // run the loop
  while (temp != NULL) {
    if (isAfter(temp->departure, now) && isAfter(temp->arrival, temp->departure) && strcmp(temp->destination->name, destination->name) == 0) {
      if (isAfter(best_arrival, temp->arrival) || (isEqual(best_arrival, temp->arrival) && cheapest > temp->cost)) {
        cheapest = temp->cost;
        *cost = cheapest;
        *departure = *temp->departure;
        *arrival = *temp->arrival;
      }
    }
    temp = temp->next;
  }
  // check if cheapest variable changed
  if (cheapest == 99999) {
    return false;
  }
  return true;
}

/*
 *  Given a list of flight_t pointers (`flight_list`) and a list of destination airport names (`airport_name_list`),
 *  first confirm that it is indeed possible to take these sequences of flights, (i.e. be sure that the i+1th flight departs
 *  after or at the same time as the ith flight arrives) (HINT: use the `isAfter()` and `isEqual()` functions).
 *  Then confirm that the list of destination airport names match the actual destination airport names of the provided flight_t structs.
 *
 *  `size` tells you the number of flights and destination airport names to consider. Be sure to extensively test for errors.
 *  As one example, if you encounter NULL's for any values that you might expect to be non-NULL return -1, but test for other possible errors too.
 *
 *  Return from this function the total cost of taking these sequence of flights.
 *  If it is impossible to take these sequence of flights,
 *  if the list of destination airport names doesn't match the actual destination airport names provided in the flight_t struct's,
 *  or if you run into any errors mentioned previously or any other errors, return -1.
 */
int validateFlightPath(flight_t** flight_list, char** airport_name_list, int size) {
  // null check
  if (flight_list == NULL || airport_name_list == NULL || size < 0) {
    return -1;
  }
  // use index for the list
  int total_cost = 0;
  bool is_there = false;
  // check for invalid list
  for (int i = 0; i < size; ++i) {
    if (flight_list[i] == NULL || airport_name_list[i] == NULL) {
      return -1;
    }
  }
  // check list for validation
  for (int i = 0; i < size; ++i) {
    if (i < size - 1 && isAfter(flight_list[i]->arrival, flight_list[i + 1]->departure)) {
      total_cost = -1;
      break;
    }
    for (int j = 0; j < size; ++j) {
      if (strcmp(flight_list[i]->destination->name, airport_name_list[j]) == 0) {
        is_there = true;
      }
    }
    if (is_there) {
      is_there = false;
      total_cost += flight_list[i]->cost;
    } else {
      total_cost = -1;
      break;
    }
  }
  return total_cost;
}
