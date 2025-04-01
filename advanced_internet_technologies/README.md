# Advanced Internet Technologies

### 1. **Laboratorio su IPv6**
In questo laboratorio, si è sperimentato con **IPv6** assegnando parametri di configurazione a una rete virtuale creata con **Mininet**.
- Installazione di **Mininet** per la creazione di reti virtuali.
- Configurazione di **IPv6** sugli host e verifica della connettività tramite `ping6`.
- Assegnazione di indirizzi IPv6 duplicati e analisi del comportamento della rete.
- Confronto con IPv4 analizzando il numero di **Neighbour Solicitations** e **ARP Replies**.

### 2. **Laboratorio su Virtual Private Networks (VPN)**
In questo laboratorio, è stata configurata una **VPN con L2TP e IPSec** per analizzarne il funzionamento e il traffico generato.
- Configurazione della VPN con i seguenti parametri:
  - **Gateway**: `sonda2.polito.it`
  - **Username**: `studente`
  - **Password**: `IeZae0ti` (la prima lettera è una 'I' maiuscola)
  - **IPSec Pre-Shared Key**: `retidicalcolatori`
- Analisi delle interfacce di rete prima e dopo l'attivazione della VPN.
- Verifica del cambio dell'IP pubblico utilizzando `curl ifconfig.me`.
- Ispezione del traffico VPN con **Wireshark**:
  - Analisi della fase di autenticazione e verifica della crittografia.
  - Identificazione dei protocolli e delle intestazioni nei pacchetti di dati.
  - Confronto del traffico con e senza VPN.

### 3. **Laboratorio su HTTP e QUIC**
Questo laboratorio si concentra sul confronto delle prestazioni tra diversi protocolli di trasporto web: **HTTP/1.1, HTTP/2 e HTTP/3 (QUIC)**.
- Uso del **Chrome Debugger** per analizzare le richieste HTTP in tempo reale.
- Misurazione delle prestazioni di **HTTP/1.1, HTTP/2 e HTTP/3** con **BrowserTime**:
  - **Page Load Time**: tempo di caricamento della pagina.
  - **Speed Index**: velocità percepita del caricamento.
- Scripting in **Bash** per eseguire web scraping:
  - Recupero della temperatura attuale di **Muggia** dal sito `https://www.3bmeteo.com/centraline-meteo` utilizzando `curl`, `grep` e `tail`.

### 4. **Google Meet Traffic Analysis**
Un'analisi dettagliata del traffico di rete generato da **Google Meet** utilizzando vari strumenti di monitoraggio.
- **Wireshark**: cattura e analisi dei pacchetti di rete.
- **Network Tab (Chrome DevTools)**: ispezione delle richieste HTTP.
- **WebRTC Internals**: monitoraggio delle connessioni WebRTC.
- **Ambiente di test**:
  - Sistema operativo: **VM con Debian**
  - Browser utilizzato: **Firefox**





