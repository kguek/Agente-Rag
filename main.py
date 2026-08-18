"""Punto di ingresso: il menu.

    python3 main.py
"""

import agent
import ingest

MENU = """
=== AGENTE RAG ===
1. Fai domande ai documenti
2. Sincronizza l'indice
3. Stato dell'indice
4. Reindicizza da zero
0. Esci
"""


def chiedi_conferma(domanda: str) -> bool:
    return input(f"{domanda} [s/N] ").strip().lower() in ("s", "si", "sì", "y", "yes")


def voce_sincronizza() -> None:
    """Prima l'anteprima, poi la conferma: la quota si spende solo dopo un sì."""
    differenze = ingest.analizza(ingest.get_store())

    print(f"\nNuovi      : {', '.join(differenze.nuovi) or '-'}")
    print(f"Modificati : {', '.join(differenze.modificati) or '-'}")
    print(f"Rimossi    : {', '.join(differenze.rimossi) or '-'}")
    print(f"Invariati  : {len(differenze.invariati)} file")

    if differenze.da_fare() == 0:
        print("\nL'indice è già aggiornato: niente da fare.")
        return

    da_embeddare = len(differenze.nuovi) + len(differenze.modificati)
    if not chiedi_conferma(f"\nProcedo? ({da_embeddare} file da rileggere, consuma quota)"):
        print("Annullato: l'indice non è stato toccato.")
        return

    ingest.sincronizza(ingest.get_store(), differenze)


def voce_stato() -> None:
    stato = ingest.stato_indice(ingest.get_store())

    print(f"\nCartella   : {ingest.DOCS_DIR}")
    print(f"Collezione : {ingest.COLLECTION_NAME} in {ingest.PERSIST_DIR}")
    print(f"Chunk      : {stato['totale']}")
    print(f"Chunking   : {ingest.CHUNK_SIZE} caratteri, {ingest.CHUNK_OVERLAP} di sovrapposizione")
    print(f"Embedding  : {ingest.EMBEDDING_MODEL} (configurato)")
    print(f"Chat       : {agent.MODEL}")

    if stato["per_materia"]:
        print("\nFile indicizzati:")
        for materia, files in stato["per_materia"].items():
            print(f"  [{materia}]")
            for source, quanti in files.items():
                print(f"    {source.split('/')[-1]}: {quanti} chunk")
    else:
        print("\nL'indice è vuoto: usa la voce 2 per popolarlo.")

    # Vettori prodotti da modelli diversi non sono confrontabili tra loro: se
    # succede, la ricerca continua a rispondere ma con risultati senza senso.
    altri = [m for m in stato["modelli"] if m not in (ingest.EMBEDDING_MODEL, "sconosciuto")]
    if altri:
        print(f"\nATTENZIONE: l'indice contiene vettori di {', '.join(altri)}.")
        print("Non sono confrontabili con quelli attuali: usa la voce 5.")


def voce_reindicizza() -> None:
    totale = ingest.stato_indice(ingest.get_store())["totale"]
    documenti = len(ingest.hash_su_disco())

    print(f"\nCancello {totale} chunk e li ricalcolo da {documenti} file.")
    print("Serve un embedding per ogni chunk: è l'operazione più costosa del programma.")

    if not chiedi_conferma("Confermi?"):
        print("Annullato: l'indice non è stato toccato.")
        return

    ingest.reindicizza_da_zero()


def main() -> None:
    while True:
        print(MENU)

        try:
            scelta = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAlla prossima!")
            return

        if scelta == "0":
            print("Alla prossima!")
            return

        # Un errore in una voce non deve chiudere il programma: si stampa e si
        # torna al menu, così non perdi la sessione per un 429 o un file rotto.
        try:
            match scelta:
                case "1":
                    agent.running_agent()
                case "2":
                    voce_sincronizza()
                case "3":
                    voce_stato()
                case "4":
                    voce_reindicizza()
                case _:
                    print("Selezione non valida.")
        except agent.QuotaEsauritaError as e:
            agent.spiega_quota(e)
        except KeyboardInterrupt:
            print("\nInterrotto: torno al menu.")
        except Exception as e:
            print(f"\nErrore durante l'operazione: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
