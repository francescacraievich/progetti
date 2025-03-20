#include <stdio.h>
#include <stdlib.h> //system
#include <unistd.h> //fork
#include <sys/wait.h> //wait
#include <string.h> //manipolazione string
#include <sys/stat.h> //stat
#define MAXPROCESSES 256
#define MAXCMD 256

int main(int argc, char *argv[]) {
//controlli iniziali
    struct stat buf;

    if (argc != 4){
        printf("Devi fornire:\n1) il nome di un file\n2) il numero di processi da eseguire in parallelo\n3) il comando da eseguire, tra virgolette\n");
        exit (1); }

    if (stat(argv[1], &buf) < 0){
        printf("Impossibile leggere il file %s\n", argv[1]);
        exit (1); }

    if (!(S_ISREG(buf.st_mode))){
        printf("%s, deve essere un file\n", argv[1]);
        exit (1); }

//apertura file e controllo
    FILE *f;
    f = fopen(argv[1], "r");
    if ((f = fopen(argv[1], "r")) == NULL) {
        printf("Impossibile aprire il file %s\n", argv[1]);
        exit(1); }

//conversione di argv[2] a numero e controllo
    int n = atoi(argv[2]);
    if (n <= 0 || n > MAXPROCESSES){
        printf("Numero di processi non valido, %d\n", n);
        exit(1); }

//creazione pipe e controllo
    int pipes[MAXPROCESSES][2], i;
    char comando[MAXCMD];
    char provvisorio[MAXCMD];
    pid_t pid;

    for (i = 0; i < n; i++){
        if (pipe(pipes[i]) == -1) {
            printf("Errore nella creazione della pipe");
            exit(1);
        }
    }
//creo npp fork
    for (i = 0; i < n; i++){
        pid = fork();
        if (pid == -1){
            printf("Errore nella fork");
            exit(1);
        }
        else if (pid == 0) {
            break;
        }
    }
//padre
    if (pid > 0) {
        for (i = 0; i < n; i++){
            close(pipes[i][0]);
        }
        i = 0;
        while (fgets(provvisorio, MAXCMD, f) != NULL) {
            provvisorio[strcspn(provvisorio, "\n")] = '\0'; //rimuovo il carattere di newline
//sostituisco il carattere "%" con il comando
            strcpy(comando, argv[3]);
            char *posizione = strstr(comando, "%");
            if (posizione != NULL) {
                strcat(provvisorio, posizione + 1);
                *posizione = '\0';
                strcat(comando, provvisorio);
            }
            else {
                printf("Comando \"%s\" non valido\n", argv[3]);
                exit (1);
            }
//scrivo nella pipe corrispondente
            write(pipes[i % n][1], comando, MAXCMD);
            i++;
        }
//close e wait
        for (i = 0; i < n; i++){
            close(pipes[i][1]);
        }
        for (i = 0; i < n; i++){
            wait(NULL);
        }
        exit(0);
    }
//figlio
    else {
        for (i = 0; i < n; i++) {
            close(pipes[i][1]);
        }
        for (i = 0; i <n; i++){
            while ((read(pipes[i][0], comando, MAXCMD)) > 0) {
                system(comando);
            }
        }
        for (i = 0; i < n; i++) {
            close(pipes[i][0]);
        }
        exit(0);
    }
    fclose(f);
    return 0;
}
