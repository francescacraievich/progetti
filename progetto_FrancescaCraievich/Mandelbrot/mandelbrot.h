#ifndef _MANDELBROT_H
#define _MANDELBROT_H

#define RADIUS 2.0

#include <complex.h>

void is_in_mandelbrot(const double complex *grid, int *iterations, int nrows, int ncols, int max_iterations);

#endif
