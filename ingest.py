"""Indicizzazione: caricamento, chunking, id, analisi e sincronizzazione.

Importare questo modulo non apre il database e non chiama nessuna API:
lo store e il modello di embedding nascono alla prima richiesta.
"""

import glob
import hashlib
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DOCS_DIR = os.path.join(BASE_DIR, "documenti")  # cartella con i documenti da indicizzare
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge-base"
SUPPORTED_EXTENSIONS = (".pdf", ".md", ".txt")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"
TOP_K = 5  # quanti chunk passare al modello per ogni domanda


# --------------------------------------------------------------------------- #
# 1. Caricamento dei documenti
# --------------------------------------------------------------------------- #
def file_hash(path: str) -> str:
    """Impronta del file intero: letta in binario, così non dipende dalla codifica."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def source_di(path: str, docs_dir: str = DOCS_DIR) -> str:
    """Percorso relativo a documenti/: identifica il file in modo univoco.

    Il solo nome non basterebbe: due materie possono avere entrambe un
    "01 - Introduzione.md", e per il sync sarebbero lo stesso documento.
    """
    return os.path.relpath(path, docs_dir).replace(os.sep, "/")


def materia_di(source: str) -> str:
    """La prima cartella del percorso; 'generale' per i file lasciati alla radice."""
    testa, _, resto = source.partition("/")

    return testa if resto else "generale"


def metadati_base(path: str) -> dict:
    """I metadata comuni a tutti i chunk di un file."""
    source = source_di(path)

    return {
        "source": source,
        "materia": materia_di(source),
        "file_hash": file_hash(path),
    }


def load_pdf(path: str) -> list[Document]:
    """Un Document per pagina, così la citazione può indicare la pagina."""
    reader = PdfReader(path)
    base = metadati_base(path)  # calcolato una volta sola, non per pagina
    pages: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:  # pagine vuote o solo immagini
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={**base, "page": page_number},
            )
        )

    return pages


def load_text(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    return [Document(page_content=text, metadata=metadati_base(path))]


def load_file(path: str) -> list[Document]:
    """Carica un singolo file: sceglie il loader in base all'estensione."""
    extension = os.path.splitext(path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return []

    return load_pdf(path) if extension == ".pdf" else load_text(path)


def percorsi_documenti(docs_dir: str = DOCS_DIR) -> list[str]:
    """I file indicizzabili nella cartella e nelle sue sottocartelle, in ordine stabile."""
    return [
        path
        for path in sorted(glob.glob(os.path.join(docs_dir, "**", "*"), recursive=True))
        if os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS
    ]


def load_documents(docs_dir: str = DOCS_DIR) -> list[Document]:
    documents: list[Document] = []

    for path in percorsi_documenti(docs_dir):
        loaded = load_file(path)
        if not loaded:
            print(f"Attenzione: nessun testo estratto da {os.path.basename(path)}")
            continue

        documents.extend(loaded)

    return documents


# --------------------------------------------------------------------------- #
# 2. Chunking
#    Nel markdown si taglia prima sui titoli (che finiscono nei metadata),
#    poi si spezzano i blocchi ancora troppo lunghi.
# --------------------------------------------------------------------------- #
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "titolo"), ("##", "sezione"), ("###", "sottosezione")],
    strip_headers=False,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)


def split_documents(documents: list[Document]) -> list[Document]:
    prepared: list[Document] = []

    for document in documents:
        if not document.metadata["source"].lower().endswith(".md"):
            prepared.append(document)
            continue

        for section in markdown_splitter.split_text(document.page_content):
            section.metadata = {**document.metadata, **section.metadata}
            prepared.append(section)

    return text_splitter.split_documents(prepared)


def build_ids(chunks: list[Document]) -> list[str]:
    """Id deterministici: stesso contenuto -> stesso id anche fra run diversi."""
    ids: list[str] = []
    seen: dict[str, int] = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "sconosciuto")
        page = chunk.metadata.get("page")
        fingerprint = hashlib.sha256(chunk.page_content.encode()).hexdigest()
        key = f"{source}:{page}:{fingerprint}"

        occorrenze = seen.get(key, 0)
        seen[key] = occorrenze + 1
        if occorrenze:  # testo identico già incontrato: lo distinguo col progressivo
            key = f"{key}:{occorrenze}"

        ids.append(hashlib.sha256(key.encode()).hexdigest())

    return ids


# --------------------------------------------------------------------------- #
# 3. Vector store, costruito pigramente e condiviso da tutto il programma
#    embed_documents usa RETRIEVAL_DOCUMENT, embed_query usa RETRIEVAL_QUERY:
#    sono i default della libreria, quindi task_type non va fissato qui.
# --------------------------------------------------------------------------- #
_embeddings: GoogleGenerativeAIEmbeddings | None = None
_store: Chroma | None = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings

    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    return _embeddings


def get_store() -> Chroma:
    """Lo store aperto una volta sola. Aprirlo non costa nessuna chiamata API."""
    global _store

    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings(),
        )

    return _store


def reset_store() -> None:
    """Butta via l'oggetto: la prossima get_store() ne apre uno nuovo.

    Serve solo dopo delete_collection(), perché quell'oggetto punta a una
    collezione che non esiste più. Chi usa lo store deve sempre chiederlo a
    get_store() al momento del bisogno, mai tenerselo in una variabile propria.
    """
    global _store

    _store = None


# --------------------------------------------------------------------------- #
# 4. Analisi delle differenze disco <-> indice (sola lettura, zero chiamate API)
# --------------------------------------------------------------------------- #
@dataclass
class Differenze:
    nuovi: list[str]
    modificati: list[str]
    rimossi: list[str]
    invariati: list[str]

    def da_fare(self) -> int:
        return len(self.nuovi) + len(self.modificati) + len(self.rimossi)


def hash_indicizzati(store: Chroma) -> dict[str, str]:
    """Mappa nome file -> impronta registrata nell'indice."""
    registrati: dict[str, str] = {}

    for meta in store.get(include=["metadatas"])["metadatas"]:
        source = meta.get("source")
        if source is None:
            continue
        # Un chunk senza file_hash viene da un'indicizzazione vecchia: la stringa
        # vuota non può coincidere con nessuna impronta, quindi risulta modificato.
        registrati[source] = meta.get("file_hash", "")

    return registrati


def hash_su_disco(docs_dir: str = DOCS_DIR) -> dict[str, str]:
    """La stessa mappa, ma calcolata sui file presenti nella cartella."""
    return {source_di(path, docs_dir): file_hash(path) for path in percorsi_documenti(docs_dir)}


def analizza(store: Chroma) -> Differenze:
    disco = hash_su_disco()
    indice = hash_indicizzati(store)
    comuni = disco.keys() & indice.keys()

    return Differenze(
        nuovi=sorted(disco.keys() - indice.keys()),
        modificati=sorted(f for f in comuni if disco[f] != indice[f]),
        rimossi=sorted(indice.keys() - disco.keys()),
        invariati=sorted(f for f in comuni if disco[f] == indice[f]),
    )


# --------------------------------------------------------------------------- #
# 5. Applicazione delle differenze (qui si spende quota)
# --------------------------------------------------------------------------- #
def rimuovi_file(store: Chroma, source: str) -> int:
    """Cancella tutti i chunk provenienti da un file. Restituisce quanti ne ha tolti."""
    ids = store.get(where={"source": source}, include=[])["ids"]
    if ids:  # delete([]) solleva ValueError
        store.delete(ids=ids)

    return len(ids)


def indicizza_file(store: Chroma, path: str) -> int:
    """Carica, spezza e inserisce un singolo file. Restituisce quanti chunk ha scritto."""
    documents = load_file(path)
    if not documents:
        print(f"Attenzione: nessun testo estratto da {os.path.basename(path)}")
        return 0

    chunks = split_documents(documents)
    for chunk in chunks:
        # Vettori di modelli diversi non sono confrontabili: registrare quale
        # modello ha prodotto il chunk permette di accorgersene.
        chunk.metadata["embedding_model"] = EMBEDDING_MODEL

    store.add_documents(documents=chunks, ids=build_ids(chunks))

    return len(chunks)


def sincronizza(store: Chroma, differenze: Differenze | None = None) -> Differenze:
    """Esegue le decisioni prese da analizza(). Gli invariati non vengono toccati."""
    if differenze is None:
        differenze = analizza(store)

    for source in differenze.rimossi:
        print(f"- {source}: rimossi {rimuovi_file(store, source)} chunk")

    for source in differenze.modificati:
        # Cancellare prima di riscrivere: se il file si è accorciato, i chunk
        # della versione vecchia non verrebbero sovrascritti da nessun id nuovo.
        rimuovi_file(store, source)
        aggiunti = indicizza_file(store, os.path.join(DOCS_DIR, source))
        print(f"~ {source}: reindicizzato in {aggiunti} chunk")

    for source in differenze.nuovi:
        aggiunti = indicizza_file(store, os.path.join(DOCS_DIR, source))
        print(f"+ {source}: indicizzato in {aggiunti} chunk")

    print(f"{differenze.da_fare()} operazioni, {len(differenze.invariati)} file invariati")

    return differenze


def reindicizza_da_zero() -> Differenze:
    """Svuota la collezione e la ricostruisce. Costa un embedding per ogni chunk."""
    get_store().delete_collection()
    reset_store()  # l'oggetto vecchio punta a una collezione che non esiste più

    return sincronizza(get_store())


# --------------------------------------------------------------------------- #
# 6. Stato dell'indice (sola lettura)
# --------------------------------------------------------------------------- #
def stato_indice(store: Chroma) -> dict:
    """Quanti chunk per materia e per file, e con quale modello sono stati creati."""
    per_materia: dict[str, dict[str, int]] = {}
    modelli: set[str] = set()

    for meta in store.get(include=["metadatas"])["metadatas"]:
        source = meta.get("source", "sconosciuto")
        materia = meta.get("materia") or materia_di(source)
        files = per_materia.setdefault(materia, {})
        files[source] = files.get(source, 0) + 1
        modelli.add(meta.get("embedding_model", "sconosciuto"))

    return {
        "per_materia": {m: dict(sorted(f.items())) for m, f in sorted(per_materia.items())},
        "totale": sum(sum(f.values()) for f in per_materia.values()),
        "modelli": sorted(modelli),
    }
