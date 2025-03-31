# Nome: Francesca
# Cognome: Craievich
# Numero di matricola: IN0501042
CC = gcc
CFLAGS = -fopenmp -O3
TARGET = mandelbrot
OBJS = main.o mandelbrot.o pgm.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

main.o: main.c mandelbrot.h pgm.h
	$(CC) $(CFLAGS) -c main.c

mandelbrot.o: mandelbrot.c mandelbrot.h
	$(CC) $(CFLAGS) -c mandelbrot.c

pgm.o: pgm.c pgm.h
	$(CC) $(CFLAGS) -c pgm.c

clean:
	rm -f $(TARGET) $(OBJS)
	rm -f *.pgm
