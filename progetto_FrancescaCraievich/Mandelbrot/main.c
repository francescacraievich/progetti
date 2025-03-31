//Francesca Craievich IN0501042

#include "mandelbrot.h"
#include "pgm.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){
	if(argc != 4){
		fprintf(stderr, "Usage: %s <filename> <max_iterations> <nrows>", argv[0]);
		return 1;
		}
	
	char *filename = argv[1];
	int M = atoi(argv [2]);
	int nrows = atoi(argv[3]);
	int ncols = (int)(1.5 * nrows);

	if(M <= 0){
		fprintf(stderr, "Iterations must be greater than 0");
		return 1;
		}
	if(nrows <=0 ){
		fprintf(stderr, "Invalid number of rows: %s", argv[2]);
		return 1;
	}
	
	double complex *grid = malloc(nrows * ncols * sizeof(double complex));
	if(grid == NULL) {
		perror("Failed to allocate memory for the grid");
		return 1;
	}
	
	// riempo la griglia equispaziata con i pixel 
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            double real = -2.0 + 3.0 * j / (ncols - 1);
            double imag = -1.0 + 2.0 * i / (nrows - 1);
            grid[i * ncols + j] = real + imag * I;
        }
    }
    
    int *iterations = malloc(nrows * ncols * sizeof(int));
    if (iterations == NULL || iterations == 0) {
        perror("Failed to allocate memory for iterations");
        free(grid);
        return 1;
    }
    
	is_in_mandelbrot(grid, iterations, nrows, ncols, M);
    
	if (save_to_pgm(filename, iterations, nrows, ncols, M) != 0) {
        fprintf(stderr, "Failed to write PGM file\n");
        free(grid);
        free(iterations);
        return 1;
    }

	free(grid);
	free(iterations);
	
    	return 0;
	
}
