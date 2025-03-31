//Francesca Craievich IN0501042

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "pgm.h"
#include <math.h>
#include <string.h>

int save_to_pgm(const char *filename, const int *iterations, int nrows, int ncols, int max_iterations)
{
    int tmp;
    size_t image_size = 0;
    char head_buffer[64];

   int fd = open(filename, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        perror("Failed to open file");
        return 1;
    }

    memset(head_buffer, 0, sizeof(char) * 64);
    int header_size = sprintf(head_buffer, "P5\n%d %d\n255\n", ncols, nrows);

    //calcola la dimensione dell'immagine PGM, inclusa l'intestazione(15 bytes)
    image_size = nrows * ncols + header_size;
    if (ftruncate(fd, image_size) != 0) {
        perror("Failed to set file size");
        close(fd);
        return 1;
    }

    //mappa il file in memoria
    char *data = mmap(NULL, image_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        perror("Failed to map file");
        close(fd);
        return 1;
    }

    //intestazione file PGM
    unsigned char *pixel_data = (unsigned char *)(data + header_size);
    memcpy(data, head_buffer, sizeof(char)*header_size);

    //scrive i dati dei pixel
    for (int i = 0; i < nrows * ncols; i++) {
        int iter = iterations[i];
        pixel_data[i] = (iter == max_iterations) ? 255 : (unsigned char)(255.0 * log(iter) / log(max_iterations));
    }

    //unmap il file e chiude il file descriptor
    if (munmap(data, image_size) != 0) {
        perror("Failed to unmap file");
        close(fd);
        return 1;
    }

    close(fd);
    return 0;
}
