/*
 * CS61C Summer 2019
 * Name: Byeong Min Park
 * Login: cs61c-acw
 */

#ifndef FLIGHT_STRUCTS_H
#define FLIGHT_STRUCTS_H

#include "timeHM.h"

typedef struct flightSys flightSys_t;
typedef struct airport airport_t;
typedef struct flight flight_t;

struct flightSys {
  airport_t* airports;
};

struct airport {
  char* name;
  flight_t* flights;
  airport_t* next;
};

struct flight {
  airport_t* destination;
  timeHM_t* departure;
  timeHM_t* arrival;
  int cost;
  flight_t* next;
};

#endif
