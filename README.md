# Progetti  
Progetti svolti durante il percorso formativo.  

## Lab 1: Configurazione di Reti IPv6 con Mininet  
**Corso:** Advanced Internet Technologies 
**Descrizione:**  
Configurazione di reti IPv6 su un ambiente virtuale usando Mininet.  
Assegnazione di indirizzi IPv6, analisi del traffico generato e confronto con IPv4, gestione delle tabelle di routing  
e test di comunicazione tra host virtuali.  
Esperimenti sulla duplicazione di indirizzi IPv6 e gestione delle tabelle di routing per la comunicazione tra sottoreti.  

## Lab 2: Virtual Private Networks (VPN)  
**Corso:** Advanced Internet Technologies   
**Descrizione:**  
Configurazione e utilizzo di una VPN basata su L2TP e IPSec.  
Analisi del traffico di rete prima e dopo l'attivazione della VPN per verificare il cambio di indirizzo IP pubblico  
e l'instradamento del traffico attraverso il server VPN.  

**Esperimenti effettuati:**  
- Configurazione della VPN con i parametri forniti.  
- Analisi dell'indirizzo IP prima e dopo la connessione alla VPN.  
- Modifica della tabella di routing per esaminare come il traffico viene instradato attraverso la VPN.  
- Utilizzo di Wireshark per analizzare il processo di autenticazione e verificare se i dati sono crittografati.  
- Ispezione del traffico sulla rete fisica e sulla VPN per verificare la cifratura dei pacchetti.  
- Analisi del caricamento di una pagina web con e senza VPN per verificare eventuali differenze nei tempi di risposta e negli IP contattati.  

## Lab 3: Analisi delle Prestazioni di HTTP e QUIC  
**Corso:** Advanced Internet Technologies   
**Descrizione:**  
Esperimenti su TCP, HTTP (diverse versioni) e QUIC per valutarne le prestazioni.  
Utilizzo di strumenti come Google Chrome Debugger, BrowserTime e `curl` per ispezionare le transazioni HTTP,  
effettuare scraping e misurare le prestazioni delle pagine web con diversi protocolli.  

**Esperimenti effettuati:**  
- Analisi del traffico HTTP tramite Google Chrome Debugger.  
- Scraping di dati meteorologici con `curl` e strumenti Bash.  
- Misurazione delle prestazioni di caricamento delle pagine web con BrowserTime, testando diversi protocolli (HTTP/1.1, HTTP/2, HTTP/3).  
- Confronto dei tempi di caricamento delle pagine con diverse versioni di HTTP.  
- Applicazione di ritardi artificiali sulla rete per osservare l'impatto sulle prestazioni di caricamento.  

## Database - Gestione dati Motomondiale  
**Corso:** Basi di Dati  
**Descrizione:**  
Progettazione e sviluppo di un database per la gestione del campionato mondiale di motociclismo (classi 125, 250 e 500).  
Il sistema prevede la registrazione delle case produttrici, dei piloti, dei team tecnici e degli sponsor.  
Inoltre, gestisce i Gran Premi, suddivisi in eventi e gare, includendo classifiche e penalità.  

## Google Meet - Analisi del Traffico di Rete  
**Corso:** Advanced Internet Technologies  
**Descrizione:**  
Analisi del traffico di rete generato da Google Meet attraverso strumenti di ispezione e monitoraggio.  

**Strumenti utilizzati:**  
- **Wireshark:** Analisi dei pacchetti di rete  
- **Network Tab:** Osservazione delle richieste e risposte HTTP  
- **WebRTC-internals:** Analisi approfondita del traffico WebRTC  

**Ambiente di test:**  
- **Sistema operativo:** VM con Debian  
- **Browser:** Firefox  

## Summary - Analisi del Phishing con Free Website Builders  
**Tesi di Laurea**  
**Tema della ricerca:**  
Studio delle tecniche di phishing utilizzando website builder gratuiti.  

**Analisi:**  
- Come i malintenzionati creano siti fraudolenti  
- L’impatto sulle aziende e sulla reputazione dei brand  
