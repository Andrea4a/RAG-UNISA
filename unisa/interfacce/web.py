"""Interfaccia Streamlit per la pipeline UniSA già esistente."""
import os
import time

import streamlit as st
from groq import APIError, Groq
from unisa.recupero import TOP_K_FINALE
from unisa.recupero.generazione import descrivi_fonte, genera_risposta


@st.cache_resource
def carica_pipeline():
    from unisa.recupero.pipeline import RetrievalPipeline

    return RetrievalPipeline()


def mostra_messaggio(m):
    with st.chat_message(m['role']):
        st.markdown(m['testo'])
        if m.get('fonti'):
            with st.expander('Fonti citate'):
                for n, p in m['fonti'].items():
                    st.write(f'[{n}] {descrivi_fonte(p)}')
                    url = p['metadata'].get('url')
                    if url and url.startswith(('https://', 'http://')):
                        st.write(url)
                    st.text(p['testo'])
                    st.divider()
        if m.get('stats'):
            s = m['stats']
            st.caption(f"Retrieval: {s['latenza_retrieval']:.2f} s · LLM: {s['latenza_llm']:.2f} s · Totale: {s['latenza_totale']:.2f} s · Token: {s['total_tokens']}")


def main():
    st.set_page_config(page_title='UniSA RAG', page_icon='🎓', layout='centered')
    st.title('Assistente UniSA')
    st.caption('Informazioni su tasse, contributi e procedure universitarie. Progetto dimostrativo non ufficiale.')
    with st.sidebar:
        st.header('Informazioni')
        st.write('Dense + BM25 → RRF → Cross-Encoder → Parent Expansion → Groq')
        st.caption(f'Top {TOP_K_FINALE} child · Fonti documentali locali')
        st.caption('La domanda e i brani recuperati sono inviati a Groq. Non inserire dati personali.')
        if st.button('Nuova conversazione'):
            st.session_state.messaggi = []
            st.session_state.domanda_pendente = None
            st.rerun()

    if 'messaggi' not in st.session_state:
        st.session_state.messaggi = []
    if 'domanda_pendente' not in st.session_state:
        st.session_state.domanda_pendente = None

    chiave = os.getenv('GROQ_API_KEY', '').strip()
    if not chiave:
        st.error('GROQ_API_KEY assente. Imposta la variabile nel terminale e riavvia Streamlit.')
        st.stop()

    for m in st.session_state.messaggi:
        mostra_messaggio(m)

    if domanda := st.chat_input('Chiedi informazioni sui documenti UniSA'):
        domanda = domanda.strip()
        if not domanda:
            st.stop()
        utente = {'role': 'user', 'testo': domanda}
        st.session_state.messaggi.append(utente)
        mostra_messaggio(utente)
        completa = domanda
        if st.session_state.domanda_pendente is not None:
            completa = f'{st.session_state.domanda_pendente}\nChiarimento utente: {domanda}'
        try:
            with st.spinner('Ricerca e generazione...'):
                pipeline = carica_pipeline()  # Inizializzazione esclusa dalla latenza per query.
                client = Groq(api_key=chiave, timeout=30.0, max_retries=0)
                inizio = time.perf_counter()
                parents = pipeline.recupera_contesto(completa)
                retrieval = time.perf_counter() - inizio
                tipo, testo, fonti, stats = genera_risposta(client, completa, parents)
                stats.update(latenza_retrieval=retrieval, latenza_totale=time.perf_counter() - inizio)
        except APIError as exc:
            st.error(f"Richiesta Groq non riuscita ({getattr(exc, 'status_code', type(exc).__name__)}). Riprova.")
        except ValueError as exc:
            st.error(f'Risposta non valida: {exc}')
        except Exception as exc:
            st.error(f'Errore durante il retrieval o la generazione: {type(exc).__name__}')
        else:
            m = {'role': 'assistant', 'testo': testo, 'fonti': fonti, 'stats': stats, 'tipo': tipo}
            st.session_state.messaggi.append(m)
            st.session_state.domanda_pendente = (f'{completa}\nDomanda di chiarimento: {testo}' if tipo == 'chiarimento' else None)
            mostra_messaggio(m)
