"""
Raccolta dei documenti per l'assistente UNISA (fermata 3 della linea del progetto RAG).

Cosa fa:
  1. legge inventario_documenti_unisa.xlsx;
  2. scarica le pagine e i PDF segnati «automatico», rispettando robots.txt e con una pausa tra le richieste;
  3. per i documenti «a mano» cerca il file che hai scaricato tu in documenti/manuali/ (nome che inizia con l'ID, es. D01.pdf);
  4. salva i file originali in documenti/raw/ e, per le pagine web, una prima versione del solo contenuto principale in documenti/testo/;
  5. per i PDF controlla se c'è testo (altrimenti sono scansionati e serve l'OCR);
  6. scrive documenti/inventario.json (URL, data, impronta sha256, dimensioni) e segnala i documenti cambiati dall'ultima raccolta;
  7. controlla che i fatti delle domande di prova compaiano nei documenti raccolti.

Uso:
  pip install requests beautifulsoup4 pdfplumber pandas openpyxl
  python -m unisa.ingestione.raccolta                 # raccolta completa
  python -m unisa.ingestione.raccolta --solo-locale   # rielabora i file già in documenti/raw e documenti/manuali

La pulizia vera e la ricomposizione del testo sono le fermate 4-6: qui si raccoglie e si controlla.
"""
import argparse, hashlib, json, re, sys, time, unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib import robotparser

import pandas as pd

from unisa.percorsi import DOCUMENTI, DOMANDE, INVENTARIO_EXCEL, RADICE

CARTELLA = DOCUMENTI
RAW, TESTO, MANUALI = CARTELLA / "raw", CARTELLA / "testo", CARTELLA / "manuali"
AGENTE = "Mozilla/5.0 (progetto di studio RAG; raccolta occasionale di pagine pubbliche)"
PAUSA = 2.0  # secondi tra una richiesta e l'altra
LETTERE = {"R": "D01", "T": "D02", "P": "D03", "C": "D04", "F": "D05"}  # sigle usate nel file delle domande


def norm(t: str) -> str:
    """Normalizza per il confronto dei fatti: minuscole, spazi singoli, apostrofi e virgolette uniformi."""
    t = unicodedata.normalize("NFC", t).lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    return re.sub(r"\s+", " ", t).strip()


def permesso(url: str, cache: dict) -> bool:
    base = "{0.scheme}://{0.netloc}".format(urlparse(url))
    if base not in cache:
        rp = robotparser.RobotFileParser(); rp.set_url(base + "/robots.txt")
        try: rp.read()
        except Exception: rp = None
        cache[base] = rp
    rp = cache[base]
    return True if rp is None else rp.can_fetch(AGENTE, url)


def scarica(url: str):
    import requests
    r = requests.get(url, headers={"User-Agent": AGENTE}, timeout=30)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


def html_a_testo(html: bytes) -> str:
    """Solo il contenuto principale, con titoli in Markdown ed elenchi; via menu, piè di pagina, moduli e questionari."""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, "html.parser")
    corpo = s.find(id="unisa-content") or s.find("main") or s.body or s
    # Schede (tab): ogni pannello riceve come titolo il nome della sua scheda, e il menu delle schede sparisce.
    # Così le FAQ restano divise per categoria (Tasse e Pagamenti, Carriera…) e i contatti per ufficio (Carriera, Didattica).
    for a in corpo.find_all("a", attrs={"data-toggle": "tab"}):
        bersaglio = (a.get("href") or "").split("#")[-1]
        pannello = corpo.find(id=bersaglio) if bersaglio else None
        if pannello is not None:
            titolo = s.new_tag("h2"); titolo.string = a.get_text(" ", strip=True)
            pannello.insert(0, titolo)
    for menu in corpo.find_all("ul", class_=lambda c: c and "nav-tabs" in c):
        menu.decompose()
    for tag in corpo.find_all(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()
    for tag in corpo.find_all(["a", "strong", "b", "em", "i", "u", "span", "abbr", "small", "sup", "sub"]):
        tag.unwrap()   # i tag dentro la frase non devono spezzarla: niente più «HELPDESK ,» o «Area Utente .»
    corpo.smooth()
    for h in corpo.find_all(re.compile(r"^h[1-4]$")):
        h.string = "\n" + "#" * int(h.name[1]) + " " + h.get_text(" ", strip=True) + "\n"
    for li in corpo.find_all("li"):
        if not li.find(["ul", "ol"]):
            li.string = "\n- " + li.get_text(" ", strip=True) + "\n"
    for tr in corpo.find_all("tr"):  # tabelle: una riga per riga di tabella, celle separate da « | »
        celle = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        tr.replace_with("\n" + " | ".join(celle) + "\n")
    testo = corpo.get_text("\n")
    testo = re.sub(r"[ \t]+([.,;:?!])", r"\1", testo)
    righe = [re.sub(r"[ \t]+", " ", r).strip() for r in testo.split("\n")]
    out, vuote = [], 0
    for r in righe:
        if not r: vuote += 1; continue
        if vuote and out: out.append("")
        out.append(r); vuote = 0
    return "\n".join(out)


def controlla_pdf(percorso: Path) -> dict:
    import pdfplumber
    with pdfplumber.open(percorso) as pdf:
        car = [len(p.extract_text() or "") for p in pdf.pages]
        tabelle = sum(len(p.find_tables()) for p in pdf.pages)
    media = sum(car) / max(1, len(car))
    return {"pagine": len(car), "caratteri": sum(car), "tabelle_con_griglia": tabelle,
            "scansionato": media < 50, "nota": "poco o nessun testo: probabilmente scansionato, serve l'OCR" if media < 50 else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventario", type=Path, default=INVENTARIO_EXCEL)
    ap.add_argument("--domande", type=Path, default=DOMANDE)
    ap.add_argument("--solo-locale", action="store_true", help="non scaricare nulla, usa i file già presenti")
    a = ap.parse_args()
    mancanti = []
    for modulo, pacchetto in [("requests", "requests"), ("bs4", "beautifulsoup4"), ("pdfplumber", "pdfplumber"), ("openpyxl", "openpyxl")]:
        try: __import__(modulo)
        except ImportError: mancanti.append(pacchetto)
    if mancanti:
        print("Mancano alcune librerie in questo Python:", ", ".join(mancanti))
        print("Installale con lo stesso Python che usi per lo script:")
        print(f'  "{sys.executable}" -m pip install {" ".join(mancanti)}')
        print("Da Jupyter, in una cella:  %pip install " + " ".join(mancanti))
        sys.exit(1)
    for c in (RAW, TESTO, MANUALI): c.mkdir(parents=True, exist_ok=True)
    inv = pd.read_excel(a.inventario, sheet_name="Inventario", dtype=str).fillna("")
    prima = {}
    manifesto = CARTELLA / "inventario.json"
    if manifesto.exists():
        prima = {d["id"]: d for d in json.loads(manifesto.read_text(encoding="utf-8"))}
    robots, esiti = {}, []
    for _, d in inv.iterrows():
        if str(d["Nel corpus?"]).strip().lower() == "no":
            continue
        did, url, auto = d["ID"], d["URL"], str(d["Download"]).startswith("automatico")
        è_pdf = "PDF" in d["Formato"] and "pagina" not in d["Formato"]
        estensione = ".pdf" if è_pdf else ".html"
        dest = RAW / f"{did}{estensione}"
        stato, nota = "", ""
        manuale = next(iter(sorted(MANUALI.glob(f"{did}*"))), None)
        if manuale:
            dest = RAW / f"{did}{manuale.suffix.lower()}"; dest.write_bytes(manuale.read_bytes()); stato = "da file manuale"
        elif a.solo_locale or not auto:
            stato = "presente" if dest.exists() else "MANCANTE"
            if not dest.exists(): nota = "scaricalo dal browser e mettilo in documenti/manuali/ con nome che inizia per " + did
        elif not permesso(url, robots):
            stato, nota = "MANCANTE", "il sito non consente il download automatico: scaricalo a mano in documenti/manuali/"
        else:
            try:
                contenuto, tipo = scarica(url); time.sleep(PAUSA)
                if "pdf" in tipo.lower(): dest = RAW / f"{did}.pdf"
                dest.write_bytes(contenuto); stato = "scaricato"
            except Exception as e:
                stato, nota = "ERRORE", f"{type(e).__name__}: {e}"
        info = {"id": did, "documento": d["Documento"], "url": url, "versione": d["Versione / anno"], "stato_inventario": d["Stato"],
                "raccolto": stato, "nota": nota, "data": date.today().isoformat()}
        if dest.exists() and stato != "MANCANTE":
            dati = dest.read_bytes(); info["sha256"] = hashlib.sha256(dati).hexdigest(); info["byte"] = len(dati); info["file"] = str(dest.relative_to(RADICE))
            if dest.suffix == ".pdf":
                info.update(controlla_pdf(dest))
            else:
                testo = html_a_testo(dati); (TESTO / f"{did}.md").write_text(
                    f"---\nid: {did}\ndocumento: {d['Documento']}\nurl: {url}\nversione: {d['Versione / anno']}\nraccolto: {info['data']}\n---\n\n{testo}\n", encoding="utf-8")
                info["caratteri_testo"] = len(testo); info["testo"] = str((TESTO / f"{did}.md").relative_to(RADICE))
            if did in prima and prima[did].get("sha256") and prima[did]["sha256"] != info["sha256"]:
                info["cambiato"] = f"diverso dalla raccolta del {prima[did].get('data')}"
        vecchio = prima.get(did, {})
        for k, v in vecchio.items():   # conserva i campi aggiunti da altri script (es. l'estrazione dei PDF)
            if k not in info and not (k in ("testo", "caratteri_testo") and dest.suffix != ".pdf"):
                info[k] = v
        if vecchio.get("sha256") and info.get("sha256") and vecchio["sha256"] != info["sha256"] and dest.suffix == ".pdf":
            info["stato_estrazione"] = "da_rifare: il PDF è cambiato"
        esiti.append(info)
    manifesto.write_text(json.dumps(esiti, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== Raccolta ==")
    for e in esiti:
        extra = f"{e.get('pagine', '')} pagine, {e.get('tabelle_con_griglia', '')} tabelle" if "pagine" in e else (f"{e.get('caratteri_testo', '')} caratteri di testo" if "caratteri_testo" in e else "")
        print(f"{e['id']}  {e['raccolto']:<16} {e['documento'][:55]:<55} {extra}")
        for k in ("nota", "cambiato"):
            if e.get(k): print(f"      ↳ {e[k]}")
    if any(e.get("scansionato") for e in esiti):
        print("\nATTENZIONE: almeno un PDF sembra scansionato: alla fermata 4 serve l'OCR.")

    # i fatti delle domande di prova compaiono nei documenti raccolti?
    testi = {}
    for e in esiti:
        if e.get("testo") and (RADICE / e["testo"]).exists(): testi[e["id"]] = norm((RADICE / e["testo"]).read_text(encoding="utf-8"))
        elif e.get("file", "").endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(RADICE / e["file"]) as pdf: testi[e["id"]] = norm("\n".join(p.extract_text() or "" for p in pdf.pages))
    url_a_id = {norm(str(u)): i for i, u in zip(inv["ID"], inv["URL"])}
    dom = pd.read_excel(a.domande, sheet_name="Domande", dtype=str).fillna("")
    mancano, controllate = [], 0
    for _, q in dom.iterrows():
        if not str(q["Fatti verificati"]).startswith("sì") or not q["Fatti attesi (separati da |)"]:
            continue
        fonti = [url_a_id.get(norm(str(q["URL della fonte"])))] if q["URL della fonte"] else [LETTERE[x] for x in re.findall(r"\b[RTPCF]\b", str(q["Fonte"]))]
        fonti = [f for f in fonti if f in testi]
        if not fonti:
            continue
        controllate += 1
        for f in str(q["Fatti attesi (separati da |)"]).split("|"):
            if f.strip() and not any(norm(f) in testi[x] for x in fonti):
                mancano.append((q["ID"], f.strip(), "/".join(fonti)))
    print(f"\n== Fatti delle domande di prova ==\nDomande controllate: {controllate}. Fatti non trovati nel testo raccolto: {len(mancano)}.")
    for qid, f, fonti in mancano:
        print(f"  {qid} [{fonti}] «{f}»")
    if mancano:
        print("Per ognuno: la pagina è cambiata, oppure il testo estratto scrive la frase in modo diverso (a capo, virgolette, abbreviazioni). Correggi il fatto nel file delle domande o l'estrazione.")


if __name__ == "__main__":
    main()
