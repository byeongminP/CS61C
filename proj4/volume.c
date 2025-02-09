#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "volume.h"

inline double volume_get(volume_t* v, int x, int y, int d) {
  return v->weights[((v->width * y) + x) * v->depth + d];
}

inline void volume_set(volume_t* v, int x, int y, int d, double value) {
  v->weights[((v->width * y) + x) * v->depth + d] = value;
}

volume_t* make_volume(int width, int height, int depth, double value) {
  volume_t* new_vol = malloc(sizeof(struct volume));
  new_vol->weights = malloc(sizeof(double) * width * height * depth);

  new_vol->width  = width;
  new_vol->height = height;
  new_vol->depth  = depth;
  double* weights = new_vol->weights;

  int currIndex;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      currIndex = ((width * y) + x) * depth;
      weights[currIndex] = value;
      weights[currIndex + 1] = value;
      weights[currIndex + 2] = value;
      for (int d = 3; d < depth; ++d) {
        weights[currIndex + d] = value;
      }
    }
  }

  return new_vol;
}

void copy_volume(volume_t* dest, volume_t* src) {
  assert(dest->width == src->width);
  assert(dest->height == src->height);
  assert(dest->depth == src->depth);

  int width = dest->width;
  int height = dest->height;
  int currIndex;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      currIndex = ((width * y) + x) * 3;
      dest->weights[currIndex] = src->weights[currIndex];
      dest->weights[currIndex + 1] = src->weights[currIndex + 1];
      dest->weights[currIndex + 2] = src->weights[currIndex + 2];
    }
  }
}

void free_volume(volume_t* v) {
  free(v->weights);
  free(v);
}
