# Agente RAG sui propri documenti

Un assistente da riga di comando che risponde a domande sui tuoi documenti
(`.pdf`, `.md`, `.txt`) citando sempre la fonte, e che tiene l'indice
allineato alla cartella dei documenti senza rifare tutto ogni volta.

## Da dove nasce

È partito come unsa serie di esercizi di tutorial di LangGraph seguendo il video di freeCodeCamp.org. Nell'ultima sezione del video l'esercizio che proponeva era un grafo minimo con due nodi — il modello che decide, il tool che cerca — giusto per capire come si
costruisce un ciclo agentico e come lo stato passa da un nodo all'altro.

Funzionava, ma era uno script monolitico: a ogni avvio ricalcolava gli
embedding di tutti i documenti, e ogni prova costava quota API. Da lì il
progetto è cresciuto dovuto alle mie curiositá ed in base ad un piano che ho richiesto di redarre a Claude su come progredirlo in base alle mie specifiche sono arrivato a questa versione basilare.

**gestione dell'indice**:

- ogni chunk ha un id derivato dal suo contenuto, quindi reinserirlo lo
  aggiorna invece di duplicarlo;
- l'impronta di ogni file è salvata nell'indice, quindi il programma sa dire
  cosa è cambiato sul disco **senza chiamare nessuna API**;
- la sincronizzazione tocca solo i file nuovi o modificati, e cancella i chunk
  di un documento prima di riscriverlo (altrimenti, accorciando un file, i
  paragrafi tolti resterebbero nell'indice per sempre);
- tutto passa da un menu, con l'anteprima di cosa sta per succedere prima di
  spendere quota.

L'agent, che era il punto di partenza, è finito per essere la parte più
piccola del codice.

## Requisiti

- Python 3.10 o superiore
- una chiave API di Google AI Studio (il piano gratuito basta)

## Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Crea un file `.env` accanto a `main.py`:

```
GOOGLE_API_KEY=la-tua-chiave
GEMINI_MODEL=gemini-3.7-flash     # facoltativo
```

`GEMINI_MODEL` serve perché il piano gratuito ha una quota **per modello**:
quando la esaurisci puoi cambiare modello e ripartire subito, invece di
aspettare il reset.

## I documenti

Metti i file in `documenti/`, raggruppati per materia in sottocartelle:

```
documenti/
    progettazione-del-software/
        09 - Design patterns.md
    basi-di-dati/
        01 - Modello ER.pdf
```

Il nome della sottocartella diventa il metadato `materia` di ogni chunk che ne
proviene. I file lasciati direttamente in `documenti/` finiscono in `generale`.

L'indice è **unico**: una domanda può pescare da tutte le materie insieme. Le
cartelle servono a organizzare e a etichettare, non a separare la ricerca.

I nomi delle cartelle devono essere validi per Chroma: minuscole, trattini al
posto degli spazi, niente accenti.

## Uso

```bash
python3 main.py
```

```
1. Fai domande ai documenti
2. Sincronizza l'indice
3. Stato dell'indice
4. Reindicizza da zero
0. Esci
```

**2 — Sincronizza l'indice.** Da fare per prima, e ogni volta che aggiungi o
modifichi documenti. Mostra prima l'anteprima (nuovi / modificati / rimossi /
invariati) e chiede conferma: la quota si spende solo dopo un sì.

**1 — Fai domande.** Il loop di chat: `exit` torna al menu. Ogni domanda costa
due chiamate al modello — una per decidere di cercare, una per rispondere — e
la risposta cita sempre file e sezione. Le ultime dieci coppie
domanda/risposta restano nel contesto, così i follow-up funzionano.

**3 — Stato dell'indice.** Quanti chunk per materia e per file, con quale
modello di embedding sono stati creati, e i parametri di chunking. Non spende
niente.

**4 — Reindicizza da zero.** Svuota la collezione e la ricostruisce. Serve
quando cambi `CHUNK_SIZE` o il modello di embedding — è l'operazione più
costosa, quindi chiede conferma mostrando quanti chunk sta per ricalcolare.

Un errore in una voce (quota esaurita, file illeggibile) stampa il messaggio e
riporta al menu: il programma non si chiude.

## Com'è organizzato

| File | Contiene |
|---|---|
| `ingest.py` | costanti, caricamento, chunking, id, vector store, analisi e sincronizzazione |
| `agent.py` | il tool di ricerca, il grafo LangGraph, la gestione della quota, la chat |
| `main.py` | il menu — l'unico file da lanciare |

Importare `ingest` o `agent` non apre il database e non costruisce il grafo:
tutto nasce alla prima richiesta, così entrare nel menu è istantaneo e
guardare lo stato dell'indice non costa niente.

Il tool di ricerca chiede il vector store **al momento della chiamata**
(`ingest.get_store()`), mai a una variabile catturata alla definizione: è ciò
che permette di sincronizzare e poi chattare subito, senza riavviare, vedendo
l'indice aggiornato.

## Come funziona, in breve

1. **Caricamento** — i PDF diventano un documento per pagina (così la
   citazione può indicare la pagina), markdown e testo un documento intero.
2. **Chunking** — il markdown si taglia prima sui titoli, che finiscono nei
   metadata; poi i blocchi ancora troppo lunghi si spezzano a 1000 caratteri
   con 200 di sovrapposizione.
3. **Embedding** — ogni chunk diventa un vettore con `gemini-embedding-001` e
   finisce in Chroma, su disco in `chroma_db/`.
4. **Ricerca** — la domanda diventa un vettore, Chroma restituisce i 5 chunk
   più simili, il modello risponde solo con quelli.

## Parametri

In cima a `ingest.py`:

| Costante | Effetto |
|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | dimensione dei chunk. Cambiandoli serve la voce 4 |
| `TOP_K` | quanti chunk passare al modello per ogni domanda |
| `EMBEDDING_MODEL` | modello di embedding. Cambiandolo serve la voce 4 |
| `COLLECTION_NAME` | nome della collezione in Chroma |

In cima a `agent.py`:

| Costante | Effetto |
|---|---|
| `MODEL` | modello di chat (da `GEMINI_MODEL` nel `.env`) |
| `MAX_SCAMBI` | quante coppie domanda/risposta restano nel contesto |
| `MAX_ATTESA_QUOTA` | oltre questi secondi di attesa, rinuncia invece di bloccarsi |

## Avvertenze

**Vettori di modelli diversi non sono confrontabili.** Se cambi
`EMBEDDING_MODEL` senza reindicizzare, la ricerca continua a rispondere ma con
risultati privi di senso. Il modello usato è salvato nei metadata e la voce 3
avvisa se nell'indice ce n'è più di uno.

**Il percorso del file è la sua identità.** Rinominare o spostare un documento
equivale, per il sync, a cancellarlo e reindicizzarlo da capo.

**La quota gratuita è limitata per modello e per minuto.** Le operazioni che
la consumano sono solo la sincronizzazione, il rebuild e le domande in chat:
tutto il resto — analisi delle differenze, stato dell'indice — lavora in
locale e non costa niente, é inserito all ineterno un pezzo di codice che vi indica
quando la quota gratuita é finita.
