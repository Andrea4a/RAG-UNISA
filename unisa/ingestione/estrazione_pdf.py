"""
Fermata 4 della linea: estrarre il testo dai PDF di UNISA.
Usa solo quello che hai visto nel quaderno «Agenti, chunking e retrieval» (B6, B7)
e nel laboratorio Jupyter (carica_pdf_con_tabelle).

Per ogni pagina del PDF:
  1. trova le tabelle con la griglia                       (B6, B7 passo 4: pdfplumber find_tables)
  2. taglia la pagina in fasce sopra e sotto ogni tabella (laboratorio: pagina.crop), così ogni tabella resta al suo posto
  3. riscrive ogni riga di tabella come frase con le intestazioni delle colonne   (B6, B7)
  4. segna il numero di pagina, per poterlo citare           (B7: la pagina va nei metadati)
Poi controlla le pagine senza testo (PDF scansionato?)       (B8, passo 1)

Non pulisce e non ricompone i paragrafi: sono le fermate 5 e 6.

Uso:  python -m unisa.ingestione.estrazione_pdf
"""
import json
import re
from pathlib import Path

import pdfplumber

from unisa.percorsi import DOCUMENTI, RADICE

CARTELLA = DOCUMENTI
INVENTARIO = CARTELLA / "inventario.json"
TESTO = CARTELLA / "testo"
X_TOLERANCE = 1.5          # questo PDF non contiene spazi: con il valore predefinito (3) le parole si attaccano
TABLE_SETTINGS = {"snap_tolerance": 6, "join_tolerance": 5, "intersection_tolerance": 5}   # unisce le linee doppie delle griglie
TITOLO = re.compile(r"^(\d+(\.\d+)+|\d+\.)\s+[A-ZÀ-Ý]")   # «3. QUANTO PAGARE», «1.5 TASSA REGIONALE…»; non «45 CFU (se…»


def riga_in_frase(titolo, intestazione, riga):
    """B6: «Napoli 85 25 40» diventa «Tetti di spesa, Napoli: albergo 85; pasto 25; taxi 40.»"""
    coppie = [f"{(h or '').strip()} {(v or '').strip()}" for h, v in zip(intestazione[1:], riga[1:]) if (v or "").strip()]
    inizio = f"{titolo}, " if titolo else ""
    return f"{inizio}{(riga[0] or '').strip()}: " + "; ".join(coppie) + "."


def tabella_in_frasi(t, stato):
    """Le righe di una tabella diventano frasi; stato ricorda intestazione e titolo tra una pagina e l'altra."""
    righe = [[" ".join((c or "").split()) for c in r] for r in t.extract(x_tolerance=X_TOLERANCE)]
    righe = [r for r in righe if any(r)]
    if not righe:
        return []
    # la prima riga è l'intestazione, se non contiene numeri; altrimenti la tabella continua dalla pagina prima
    ha_numeri = any(re.search(r"[0-9]", cella) for cella in righe[0])   # solo cifre vere: «(²)» negli apici non conta
    if not ha_numeri:
        stato["intestazione"], dati = righe[0], righe[1:]
    else:
        dati = righe
    if not stato["intestazione"] or len(stato["intestazione"]) != len(righe[0]):
        return [" | ".join(r) for r in righe]                     # non so a cosa corrispondono le colonne: la lascio com'è
    stato["tabelle"] += 1
    return [riga_in_frase(stato["titolo"], stato["intestazione"], r) for r in dati]   # 3. righe in frasi


def estrai_pdf(percorso):
    pagine, pagine_vuote = [], []
    stato = {"intestazione": None, "titolo": "", "tabelle": 0}   # intestazione e titolo valgono anche nelle pagine dopo
    with pdfplumber.open(percorso) as pdf:
        for numero, pagina in enumerate(pdf.pages, start=1):
            tabelle = sorted(pagina.find_tables(TABLE_SETTINGS), key=lambda t: t.bbox[1])   # 1. tabelle, dall'alto in basso
            pezzi, y = [], 0
            # 2. la pagina tagliata in fasce orizzontali: testo sopra la tabella, tabella, testo sotto…
            #    così ogni tabella resta sotto il suo titolo, al suo posto
            for t in tabelle + [None]:
                fondo = t.bbox[1] if t else pagina.height
                if fondo > y:
                    testo = pagina.crop((0, y, pagina.width, fondo)).extract_text(x_tolerance=X_TOLERANCE) or ""
                    titoli = [r.strip() for r in testo.split("\n") if TITOLO.match(r.strip())]
                    if titoli:
                        stato["titolo"] = titoli[-1]
                    if testo.strip():
                        pezzi.append(testo)
                if t:
                    frasi = tabella_in_frasi(t, stato)
                    if frasi:
                        pezzi.append("\n".join(frasi))
                    y = t.bbox[3]
            if not "".join(pezzi).strip():
                pagine_vuote.append(numero)
            pagine.append(f"<!-- Pagina PDF {numero} -->\n" + "\n\n".join(pezzi))   # 4. numero di pagina
    return "\n\n".join(pagine), stato["tabelle"], pagine_vuote


def main():
    TESTO.mkdir(parents=True, exist_ok=True)
    inventario = json.loads(INVENTARIO.read_text(encoding="utf-8"))
    for doc in inventario:
        if not doc.get("file", "").lower().endswith(".pdf"):
            continue
        testo, n_tab, vuote = estrai_pdf(RADICE / doc["file"])
        intestazione = f"---\nid: {doc['id']}\ndocumento: {doc['documento']}\nurl: {doc['url']}\nversione: {doc['versione']}\n---\n\n"
        destinazione = TESTO / f"{doc['id']}.md"
        destinazione.write_text(intestazione + testo + "\n", encoding="utf-8")
        doc.update(testo=str(destinazione.relative_to(RADICE)), caratteri_testo=len(testo), tabelle_estratte=n_tab,
                   stato_estrazione="da_verificare", pagine_senza_testo=vuote)
        print(f"{doc['id']}: {len(testo)} caratteri, {n_tab} tabelle riscritte in frasi"
              + (f", pagine senza testo: {vuote} (scansionate? serve l'OCR)" if vuote else ""))
    INVENTARIO.write_text(json.dumps(inventario, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nOra confronta a occhio documenti/testo/D01.md con il PDF: tabelle, salti pagina, articoli.")


if __name__ == "__main__":
    main()
