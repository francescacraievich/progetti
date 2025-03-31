# Laboratori di Programmazione

Questo repository contiene cinque laboratori di programmazione su **Bash** e **C**, ognuno con un obiettivo specifico. Di seguito vengono descritte le funzionalità di ciascun laboratorio e i comandi necessari per eseguire i programmi.

---

## Lab 1: Bash Programming - Address Book

### Descrizione
Questo laboratorio prevede la creazione di uno script Bash `address-book.sh` che gestisce una rubrica salvata in un file CSV (`address-book-database.csv`). L'utente può **visualizzare, cercare, inserire o cancellare contatti** tramite comandi specifici.

### Esecuzione
```bash
chmod +x address-book.sh  # Rendi eseguibile lo script
./address-book.sh view    # Visualizza tutti i contatti
./address-book.sh search <string>  # Cerca un contatto
./address-book.sh insert  # Inserisce un nuovo contatto
./address-book.sh delete <mail>  # Cancella un contatto tramite email
```

---

## Lab 2: Gestione dei file in C

### Descrizione
Il programma in **C** esplora ricorsivamente una cartella e stampa informazioni sui file, come **nome, inode, tipo, dimensione e proprietario**.

### Compilazione ed esecuzione
```bash
gcc -o file_manager file_manager.c  # Compila il programma
./file_manager <directory>  # Esegui il programma sulla directory specificata
```

---

## Lab 3: Processi - Simulazione di GNU Parallel

### Descrizione
Questo laboratorio implementa una versione semplificata del comando **parallel**, che esegue comandi Bash in parallelo usando un numero definito di processi concorrenti.

### Compilazione ed esecuzione
```bash
gcc -o parallel parallel.c -pthread  # Compila il programma con supporto ai thread
./parallel args.txt 2 "ls % -lh"  # Esegue i comandi definiti in args.txt con 2 processi paralleli
```

---

## Lab 4: Sincronizzazione - Gestione di un autonoleggio

### Descrizione
Un programma **C** che gestisce un **autonoleggio concorrente**, permettendo di **visualizzare lo stato delle auto, prenotarle (lock) e rilasciarle (release).**

### Compilazione ed esecuzione
```bash
gcc -o car_rental car_rental.c -pthread  # Compila con supporto ai thread
./car_rental  # Avvia il programma interattivo
```

### Comandi interni al programma
```bash
view  # Mostra lo stato delle auto
lock <vettura>  # Prenota un'auto
release <vettura>  # Rilascia un'auto
quit  # Esci dal programma
```

---

## Lab 5: Container Engine in Bash

### Descrizione
Uno script **Bash** che emula un **ambiente isolato** per eseguire un comando con un set limitato di file e cartelle specificati in un file di configurazione.

### Esecuzione
```bash
chmod +x container-run.sh  # Rendi eseguibile lo script
./container-run.sh conf-file.txt /bin/ls  # Esegue il comando /bin/ls all'interno del container
```

