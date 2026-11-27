# RAG UniSA

Progetto dimostrativo per rispondere in italiano a domande sui documenti dell’Università di Salerno. La ricerca ibrida combina embedding e BM25. Un reranker ordina i risultati prima di recuperare le sezioni di provenienza. La generazione usa `openai/gpt-oss-20b` tramite Groq e riporta i riferimenti ai brani citati.

## Struttura

```text
config/                  Regole di pulizia del testo
unisa/
  ingestione/            Raccolta, estrazione, pulizia e struttura dei documenti
  indicizzazione/        Chunking, controlli, embedding e indice ChromaDB
  recupero/              Retrieval, reranking e generazione della risposta
  interfacce/            Applicazione Streamlit e terminale
  valutazione/           Annotazioni, metriche e flussi di valutazione
strumenti/               Ricerca di controllo nei testi estratti
documenti/               Dati locali e risultati intermedi
chroma_db/               Indice vettoriale locale
```

## Installazione su Windows

Apri PowerShell nella cartella del progetto. Le dipendenze in `requirements.txt` corrispondono alle versioni usate con Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Preparazione dei dati e dell’indice

I file di input vanno aggiunti localmente dopo aver clonato il repository.

| Percorso | Contenuto |
| --- | --- |
| `documenti/inventario_documenti_unisa.xlsx` | Inventario delle fonti, nel foglio `Inventario` |
| `documenti/domande_di_prova_unisa.xlsx` | Domande e fatti attesi per i controlli, nel foglio `Domande` |
| `documenti/manuali/` | File da inserire a mano, con il nome che inizia dall’ID della fonte |
| `documenti/raw/` | Copie locali delle fonti HTML o PDF per la raccolta offline |

L’inventario usa le colonne `ID`, `Documento`, `URL`, `Versione / anno`, `Stato`, `Nel corpus?`, `Download` e `Formato`. Imposta `Nel corpus?` a `sì` per le fonti da elaborare. I controlli della catena leggono `ID`, `Fatti verificati`, `Fatti attesi (separati da |)`, `URL della fonte` e `Fonte` nel foglio `Domande`. I campi aggiuntivi `Set`, `Tipo`, `Domanda` e `Risposta attesa` servono alla valutazione della generazione sul set tenuto da parte.

Esegui la catena con i file locali e poi costruisci l’indice. Per scaricare o aggiornare le fonti automatiche dell’inventario, aggiungi `--scarica` al primo comando.

```powershell
.\.venv\Scripts\python.exe esegui_catena.py
.\.venv\Scripts\python.exe -m unisa.indicizzazione.costruisci
```

La costruzione dell’indice usa un modello da Hugging Face. Anche la ricerca carica da Hugging Face il modello di reranking. Al primo utilizzo i pesi vengono scaricati nella cache utente. Gli input, i risultati di valutazione, i dati elaborati, l’indice e `.venv` sono esclusi da Git.

## Uso

La ricerca da terminale mostra i brani quando la variabile `GROQ_API_KEY` è assente.

```powershell
.\.venv\Scripts\python.exe -m unisa.interfacce.terminale
```

Per generare risposte, crea una [chiave Groq](https://console.groq.com/keys) e impostala nella sessione PowerShell. Groq riceve la domanda e i brani recuperati. L’indice ChromaDB rimane locale. La stessa chiave serve per avviare la modalità chat in Streamlit.

```powershell
$env:GROQ_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Chiave Groq' -AsSecureString)).Password
.\.venv\Scripts\python.exe -m streamlit run unisa/app.py
```

I comandi storici `python unisa/chunking_unisa.py` e `python unisa/retrieval_unisa.py` restano disponibili. Nel terminale interattivo, digita `exit` per uscire.

## Valutazione

I comandi che usano le annotazioni richiedono `documenti/ground_truth.json` e un indice coerente con `documenti/parents.json`. I risultati vengono scritti in `documenti/`. Esegui i comandi dalla radice con `.venv` attivo, oppure sostituisci `python` con `.\.venv\Scripts\python.exe`.

| Comando Python | Funzione |
| --- | --- |
| `python -m unisa.valutazione.evidenze` | Mappa le evidenze annotate ai child |
| `python -m unisa.valutazione.confronto` | Confronta i metodi di recupero sul development set |
| `python -m unisa.valutazione.finale` | Misura le fasi della configurazione finale di recupero |
| `python -m unisa.valutazione.generazione run --set holdout` | Genera le risposte del set tenuto da parte. Richiede `GROQ_API_KEY` e il file Excel delle domande |
| `python -m unisa.valutazione.generazione review --set holdout` | Esporta le risposte per la revisione manuale |
| `python -m unisa.valutazione.generazione summary --set holdout` | Riepiloga la revisione |
