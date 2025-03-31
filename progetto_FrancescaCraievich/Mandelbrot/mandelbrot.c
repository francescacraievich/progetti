//Francesca Craievich IN0501042

#include "mandelbrot.h"
#include <complex.h>
#include <stdlib.h>
#include <omp.h>

//svolgo in parallelo le iterazioni per calcolare i punti presenti dentro l'insieme di mandelbrot

void is_in_mandelbrot(const double complex *grid, int *iterations, int nrows, int ncols, int max_iterations) {
    
    #pragma omp parallel for
    for (int i = 0; i < nrows * ncols; ++i) {
        double complex z = 0 + 0 * I;
        int iter;
        for (iter = 0; iter < max_iterations; ++iter) {
            z = z * z + grid[i];
            if (cabs(z) >= RADIUS) {
                break;
            }
        }
        iterations[i] = iter;
    }
}


