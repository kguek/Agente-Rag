"""L'agent: tool di ricerca, grafo LangGraph, gestione della quota e loop di chat.

Come ingest, importare questo modulo non costruisce niente: il modello e il
grafo nascono alla prima domanda.
"""

import os
import re
import time
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import ingest  # importa il modulo, non i singoli nomi: lo store va sempre richiesto a lui

# Il piano gratuito ha una quota giornaliera di richieste PER MODELLO: quando la
# esaurisci puoi passare a un altro modello (la quota riparte) invece di aspettare.
# Ogni domanda costa 2 chiamate: una per decidere di cercare, una per rispondere.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
MAX_ATTESA_QUOTA = 60  # secondi: oltre questa soglia non ha senso restare bloccati
MAX_SCAMBI = 10  # quante coppie domanda/risposta restano nel contesto


# --------------------------------------------------------------------------- #
# 1. Il tool di ricerca
# --------------------------------------------------------------------------- #
def formatta_documenti(docs) -> str:
    """Testo dei chunk con il riferimento alla fonte, per la risposta citata."""
    results = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        riferimento = meta.get("source", "sconosciuto")
        if "page" in meta:
            riferimento += f", pagina {meta['page']}"
        sezione = meta.get("sezione") or meta.get("titolo")
        if sezione:
            riferimento += f", sezione: {sezione}"
        results.append(f"Documento {i} — fonte: {riferimento}\n{doc.page_content}")

    return "\n\n".join(results)


@tool
def retriever_tool(query: str) -> str:
    """
    Questo tool cerca e ritorna le informazioni contenute nella cartella documenti.
    """
    # Lo store viene chiesto adesso, non alla definizione del tool: così dopo una
    # sincronizzazione o un rebuild la ricerca vede sempre l'indice aggiornato.
    docs = ingest.get_store().similarity_search(query, k=ingest.TOP_K)
    if not docs:
        return "Nessun documento rilevante trovato."

    return formatta_documenti(docs)


tools = [retriever_tool]
tools_dict = {t.name: t for t in tools}


# --------------------------------------------------------------------------- #
# 2. Il grafo
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


system_prompt = """
Sei un assistente AI esperto della documentazione personale dell'utente.
Rispondi soltanto con i contenuti che ottieni dal tool retriever_tool: non usare la tua conoscenza generale.
Chiama sempre il tool prima di rispondere; se la prima ricerca non basta, richiamalo riformulando la query.
Se il tool non restituisce nulla di pertinente, dillo esplicitamente invece di inventare una risposta.
Cita sempre la fonte (nome del file e sezione) che hai usato per costruire la risposta.
"""

_llm = None
_rag_agent = None


def get_llm():
    global _llm

    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0).bind_tools(tools)

    return _llm


def should_continue(state: AgentState) -> bool:
    """Controlla se nell'ultimo messaggio il modello ha chiesto di usare un tool."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


def call_llm(state: AgentState) -> AgentState:
    """Chiama la LLM sullo stato corrente."""
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    return {"messages": get_llm().invoke(messages)}


def call_retriever(state: AgentState) -> AgentState:
    """Esegue le chiamate ai tool richieste dalla LLM."""
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Cerco nei documenti: {t['args'].get('query', 'nessuna query')}")

        if t["name"] not in tools_dict:
            print(f"Il tool {t['name']} non esiste")
            result = "Nome del tool errato: riprova scegliendo un tool dalla lista."
        else:
            result = tools_dict[t["name"]].invoke(t["args"])

        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))

    return {"messages": results}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever", call_retriever)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever", False: END})
    graph.add_edge("retriever", "llm")
    graph.add_edge(START, "llm")

    return graph.compile()


def get_agent():
    global _rag_agent

    if _rag_agent is None:
        _rag_agent = build_agent()

    return _rag_agent


# --------------------------------------------------------------------------- #
# 3. Gestione della quota (errore 429 RESOURCE_EXHAUSTED)
# --------------------------------------------------------------------------- #
_RETRY_DELAY = re.compile(r"'retryDelay': '(\d+(?:\.\d+)?)s'")


class QuotaEsauritaError(Exception):
    """La quota Gemini è finita e riprovare subito non ha senso."""

    def __init__(self, attesa=None, giornaliera=False):
        self.attesa = attesa
        self.giornaliera = giornaliera
        super().__init__("Quota dell'API Gemini esaurita")


def is_quota_error(err: Exception) -> bool:
    testo = str(err)
    return "RESOURCE_EXHAUSTED" in testo or "429" in testo


def spiega_quota(e: QuotaEsauritaError) -> None:
    """Messaggio unico per il menu e per la chat."""
    print(f"\nQuota dell'API Gemini esaurita per il modello {MODEL}.")
    if e.giornaliera:
        print("È la quota GIORNALIERA del piano gratuito: aspettare non serve.")
        print("Il limite è per modello, quindi puoi cambiare GEMINI_MODEL nel .env")
        print("e continuare subito, oppure attendere il reset o attivare il billing.")
    elif e.attesa:
        print(f"Google chiede di riprovare tra circa {e.attesa:.0f} secondi.")


def chiedi_all_agent(history: list[BaseMessage], tentativi: int = 2):
    """Invoca il grafo; se Google chiede un'attesa breve riprova da solo, altrimenti alza QuotaEsauritaError."""
    for tentativo in range(1, tentativi + 1):
        try:
            return get_agent().invoke({"messages": history})
        except Exception as e:
            if not is_quota_error(e):
                raise  # errori diversi dalla quota devono restare visibili

            testo = str(e)
            # Il quotaId dice quale limite hai sfondato: "...PerDay..." è quello
            # giornaliero, e in quel caso aspettare i secondi suggeriti è inutile.
            giornaliera = "PerDay" in testo
            match = _RETRY_DELAY.search(testo)
            attesa = float(match.group(1)) + 1 if match else None

            if giornaliera or tentativo == tentativi or attesa is None or attesa > MAX_ATTESA_QUOTA:
                raise QuotaEsauritaError(attesa, giornaliera) from e

            print(f"Quota momentaneamente esaurita: riprovo tra {attesa:.0f}s...")
            time.sleep(attesa)


# --------------------------------------------------------------------------- #
# 4. Loop di conversazione
# --------------------------------------------------------------------------- #

"""
Attualmente la history prende solamente solamente HumanMessage e AiMessage(senza i chuck di ToolMessage)
In caso volessi prendere anche il chunck dell ultima risposta, devo filtrare le utlime 10 conversazioni
per includere solamente HumanMessage e AiMessage, infine prendo tutto dell'ultima domanda(HumanMessage
, ToolMessage e AiMessage).
Cambiare il system_prompt in mnaiera tale che riusa gli ultimi chunks

def domanda_risposta(message:list[BaseMessage])->list[BaseMessage
puliti=[]
for m in message:
    if isinstance(m,ToolMessage):
        continue:
    if isinstance(m,AIMessage):
        if m.tool_calls:
            continue
            puliti.append(AIMessage(content=m.text))
    else:
        puliti.append(m)
return puliti

in def running_aget() diventa:
completo = list(result["messages"])
turno = completo[len(history) - 1:]# domanda + tool + risposta
precedenti = solo_scambi(completo[: len(history) - 1])[-2 * MAX_SCAMBI:]
history = precedenti + turno

"""
def running_agent() -> None:
    """Ogni giro riusa lo storico, così i follow-up hanno contesto.

    Nello storico restano solo le coppie domanda/risposta: i chunk ripescati dal
    tool servono a produrre la risposta, non a essere riletti al giro dopo, e
    pesano ~1300 token l'uno. Tenerli farebbe crescere ogni richiesta all'infinito.
    """
    print("\n=== CHAT ===")
    print("Fai una domanda sui tuoi documenti. Scrivi 'exit' per tornare al menu.")
    print(f"Chunk nell'indice: {len(ingest.get_store().get(include=[])['ids'])}")

    history: list[BaseMessage] = []

    while True:
        user_input = input("\nDomanda: ").strip()

        if user_input.lower() in ("exit", "quit", "q"):
            print("Torno al menu.")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))

        try:
            result = chiedi_all_agent(history)
        except QuotaEsauritaError as e:
            history.pop()  # la domanda non ha avuto risposta: non sporcare lo storico
            spiega_quota(e)
            print("La chat resta aperta: la domanda non è stata registrata.")
            continue

        # .text concatena i blocchi di testo: gemini-3.x restituisce content
        # come lista di blocchi, quindi .content stamperebbe la struttura grezza.
        risposta = result["messages"][-1].text
        history.append(AIMessage(content=risposta))
        history = history[-2 * MAX_SCAMBI:]

        print("\n=== RISPOSTA ===")
        print(risposta)
