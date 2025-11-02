"""

Entradas:
- Carpeta con documentos de texto plano (.txt)
- Archivo de stopwords (una palabra por línea)
- Archivo de lematización en formato JSON: {"is":"be","was":"be",...}

Salidas:
- Para cada documento: CSV con columnas [term_index, term, TF, IDF, TF-IDF]
- Matriz de similitud coseno entre documentos: similarities.csv

Uso (por ejemplo):
    python main.py --docs_dir ./docs --stopwords ./stop-words/stop-words-en.txt --lemmatization ./corpus/corpus-en.txt --outdir ./results

"""

from pathlib import Path
import argparse
import string
import math
import json
from collections import Counter
import numpy as np
import pandas as pd


# ------------------------ LECTURA DE ARCHIVOS ------------------------

def load_stopwords(path: Path):
  """Carga un conjunto de stopwords (una por línea)."""
  stop = set()
  with path.open(encoding='utf-8') as f:
    for line in f:
      w = line.strip().lower()
      if w:
        stop.add(w)
  return stop


def load_lemmatization_json(path: Path):
  """Carga un diccionario de lematización desde un JSON: {"is":"be",...}"""
  with path.open(encoding='utf-8') as f:
    data = json.load(f)
  # asegurar que las claves/valores estén en minúsculas
  return {k.lower(): v.lower() for k, v in data.items()}


# ------------------------ PREPROCESAMIENTO ------------------------

def tokenize(text: str):
  """Tokeniza texto: minúsculas, elimina puntuación, separa por espacios."""
  text = text.lower()
  trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
  text = text.translate(trans)
  tokens = [t for t in text.split() if t]
  return tokens


def preprocess(tokens, stopwords, lemmatization):
  """Elimina stopwords, aplica lematización y descarta números."""
  out = []
  for t in tokens:
    if t.isdigit():
      continue
    if t in stopwords:
      continue
    if t in lemmatization:
      out.append(lemmatization[t])
    else:
      out.append(t)
  return out


# ------------------------ CÁLCULO TF-IDF ------------------------

def compute_tfidf(docs_tokens):
  """
  docs_tokens: lista de listas de tokens (ya preprocesados)
  Devuelve vocabulario, TFs, IDF y vectores TF-IDF.
  """
  vocab = sorted({t for doc in docs_tokens for t in doc})
  term_to_index = {t: i for i, t in enumerate(vocab)}
  N = len(docs_tokens)

  # DF: en cuántos documentos aparece cada término
  doc_freq = Counter()
  raw_tfs = []
  for doc in docs_tokens:
    c = Counter(doc)
    raw_tfs.append(c)
    for term in set(doc):
      doc_freq[term] += 1

  # IDF suavizado: log((N + 1)/(df + 1)) + 1
  idf = {term: math.log((N + 1) / (doc_freq[term] + 1)) + 1 for term in vocab}

  # TF normalizado: frecuencia relativa
  tfs = []
  for c in raw_tfs:
    total = sum(c.values())
    tf_norm = {t: (c[t] / total) if total > 0 else 0.0 for t in vocab}
    tfs.append(tf_norm)

  # TF-IDF
  tfidf_vectors = []
  for tf in tfs:
    v = {t: tf[t] * idf[t] for t in vocab}
    tfidf_vectors.append(v)

  return vocab, term_to_index, tfs, idf, tfidf_vectors


# ------------------------ SALIDA Y SIMILITUD ------------------------

def save_term_tables(outdir: Path, filenames, vocab, tfs, idf, tfidf_vectors):
  outdir.mkdir(parents=True, exist_ok=True)
  for fname, tf, vec in zip(filenames, tfs, tfidf_vectors):
    rows = []
    for idx, term in enumerate(vocab):
      rows.append((idx, term, tf[term], idf[term], vec[term]))
    df = pd.DataFrame(rows, columns=['term_index', 'term', 'TF', 'IDF', 'TF-IDF'])
    csv_path = outdir / f"{Path(fname).stem}_term_table.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')


def vectors_to_matrix(vocab, tfidf_vectors):
  """Convierte lista de vectores TF-IDF (dicts) en una matriz NumPy."""
  M = np.array([[vec[t] for t in vocab] for vec in tfidf_vectors], dtype=float)
  return M


def cosine_similarity_matrix(mat):
  """Devuelve matriz de similitud coseno entre filas."""
  norms = np.linalg.norm(mat, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  normalized = mat / norms
  sim = normalized @ normalized.T
  return np.clip(sim, -1.0, 1.0)


# ------------------------ PROCESO PRINCIPAL ------------------------

def process_documents(docs_dir: Path, stopwords_file: Path, lemmatization_file: Path, outdir: Path):
  # 1. Cargar stopwords y lematización
  stopwords = load_stopwords(stopwords_file)
  lemmatization = load_lemmatization_json(lemmatization_file)

  # 2. Leer documentos .txt
  txt_files = sorted([p for p in docs_dir.glob("*.txt")])
  if not txt_files:
    raise ValueError(f"No se encontraron documentos .txt en {docs_dir}")
  filenames = [p.name for p in txt_files]

  docs_tokens = []
  for p in txt_files:
    text = p.read_text(encoding='utf-8')
    tokens = tokenize(text)
    processed = preprocess(tokens, stopwords, lemmatization)
    docs_tokens.append(processed)

  # 3. Calcular TF-IDF
  vocab, term_to_index, tfs, idf, tfidf_vectors = compute_tfidf(docs_tokens)

  # 4. Guardar tablas por documento
  save_term_tables(outdir, filenames, vocab, tfs, idf, tfidf_vectors)

  # 5. Calcular similitudes
  M = vectors_to_matrix(vocab, tfidf_vectors)
  sim = cosine_similarity_matrix(M)
  sim_df = pd.DataFrame(sim, index=filenames, columns=filenames)
  sim_path = outdir / "similarities.csv"
  sim_df.to_csv(sim_path, encoding='utf-8')

  print(f"\nProcesamiento completado. Resultados guardados en: {outdir}")
  print(f"- Archivos de términos: *_term_table.csv")
  print(f"- Matriz de similitud: similarities.csv")

  return vocab, sim_df


# ------------------------ EJECUCIÓN DESDE TERMINAL ------------------------

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sistema de recomendación basado en contenido (TF-IDF + similitud coseno)")
  parser.add_argument("--docs_dir", type=str, required=True, help="Carpeta con documentos .txt")
  parser.add_argument("--stopwords", type=str, required=True, help="Archivo de stopwords (una por línea)")
  parser.add_argument("--lemmatization", type=str, required=True, help="Archivo JSON de lematización")
  parser.add_argument("--outdir", type=str, default="./salida", help="Directorio de salida (por defecto ./salida)")
  args = parser.parse_args()

  process_documents(
    docs_dir=Path(args.docs_dir),
    stopwords_file=Path(args.stopwords),
    lemmatization_file=Path(args.lemmatization),
    outdir=Path(args.outdir)
  )