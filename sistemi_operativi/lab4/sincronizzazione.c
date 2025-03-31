#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#define MAX_CARS 100
#define MAX_LEN 100

typedef struct {
    char id[MAX_LEN];
    int busy;  // 0: free, 1: busy
} Car;

Car cars[MAX_CARS];
int car_count = 0;
sem_t *sem;

void load_cars(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file '%s': %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    while (fscanf(file, "%s", cars[car_count].id) != EOF) {
        cars[car_count].busy = 0;
        car_count++;
    }

    fclose(file);
}

void save_cars(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Cannot open file '%s': %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < car_count; i++) {
        fprintf(file, "%s\n", cars[i].id);
    }

    fclose(file);
}

void view_cars() {
    sem_wait(sem);
    for (int i = 0; i < car_count; i++) {
        printf("Car: %s, status: %s\n", cars[i].id, cars[i].busy ? "busy" : "free");
    }
    sem_post(sem);
}

void lock_car(const char *car_id) {
    sem_wait(sem);
    int found = 0;
    for (int i = 0; i < car_count; i++) {
        if (strcmp(cars[i].id, car_id) == 0) {
            found = 1;
            if (cars[i].busy) {
                printf("Error. Car %s already locked\n", car_id);
            } else {
                cars[i].busy = 1;
                printf("Car: %s is now locked\n", car_id);
            }
            break;
        }
    }
    if (!found) {
        printf("Cannot find car %s\n", car_id);
    }
    sem_post(sem);
}

void release_car(const char *car_id) {
    sem_wait(sem);
    int found = 0;
    for (int i = 0; i < car_count; i++) {
        if (strcmp(cars[i].id, car_id) == 0) {
            found = 1;
            if (cars[i].busy) {
                cars[i].busy = 0;
                printf("Car: %s is now free\n", car_id);
            } else {
                printf("Error. Car %s already free\n", car_id);
            }
            break;
        }
    }
    if (!found) {
        printf("Cannot find car %s\n", car_id);
    }
    sem_post(sem);
}

void *operator_thread(void *arg) {
    char command[MAX_LEN];
    while (1) {
        printf("Command: ");
        fflush(stdout); // Ensure "Command: " is printed immediately

        if (scanf("%s", command) != 1) {
            continue;
        }

        if (strcmp(command, "view") == 0) {
            view_cars();
        } else if (strcmp(command, "lock") == 0) {
            char car_id[MAX_LEN];
            if (scanf("%s", car_id) == 1) {
                lock_car(car_id);
            }
        } else if (strcmp(command, "release") == 0) {
            char car_id[MAX_LEN];
            if (scanf("%s", car_id) == 1) {
                release_car(car_id);
            }
        } else if (strcmp(command, "quit") == 0) {
            save_cars("catalog.txt");
            sem_close(sem);
            sem_unlink("/car_rental_sem");
            exit(0);
        } else {
            printf("Unknown Command\n");
        }
    }
    return NULL;
}

int main() {
    const char *filename = "catalog.txt";
    if (access(filename, F_OK) == -1) {
        printf("File '%s' does not exist\n", filename);
        exit(EXIT_FAILURE);
    }

    load_cars(filename);

    sem = sem_open("/car_rental_sem", O_CREAT, 0644, 1);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        exit(EXIT_FAILURE);
    }

    pthread_t tid;
    pthread_create(&tid, NULL, operator_thread, NULL);
    pthread_join(tid, NULL);

    return 0;
}

