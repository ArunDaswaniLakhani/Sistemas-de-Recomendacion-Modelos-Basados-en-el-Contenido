# Sistemas de recomendación. Modelos basados en el contenido

Este proyecto implenta un análsis de contenido basado en TF-iDF y además
la similitud coseno para comparar documentos
Genera para cada docuemento pasado un arhcigvo csv con toda la información tf-idf y
además una amtriz de similitud entre documentos para poder observar la información

###  Dependencias necesarias

* Python 3
* `numpy`
* `pandas`

###  Instalación

```bash
pip install numpy pandas
```


## Ejecución del proyecto

### Estructura

Nuestra estructura es una carpeta
docs/ donde esta los documentos
stop-words/ donde esta el fichero de stop words
corpus/ donde se ecnuentra el fichero de lematización

### Comando para ejecutar el programa y ejemplo de uso

```bash
python main.py --docs_dir ./docs --stopwords ./stop-words/stop-words-en.txt --lemmatization ./corpus/corpus-en.txt --outdir ./results
```
Como vemos hay 4 parametros diferentes, que indican las rutas necesarias


## Descripción del codigo

### Carga de datos

- `load_stopwords(path)` : Carga un conjunto de stopword desde un archivo
- `load_lemmatization_json(path)`: Pasandole el path a un archivo carga un json de lematizacion

### Procesamiento de texto

- `tokenize(text)` : Le pasas un texto y te lo pasa a minúsculas, te quita puntuación y lo tokeniza
- `preprocess(tokens, stopwords, lemmatization)` : Se encarga de eliminar las stopwords, numero y aplica lematizacion

### Calcular TF_IDF

- `compute_tfidf(docs_tokens)` : Calcula el vocavulario, tf normalizado, idf suavizado y matriz tf-idf de todos los documentos 

### Generación de resultados

- `save_term_tables(outdir: Path, filenames, vocab, tfs, idf, tfidf_vectors)`: Crea archivos CSV por documento con los valores TF‑IDF.
- `vectors_to_matrix(vocab, tfidf_vectors)`: Convierte los vectores en una matriz NumPy.
- `cosine_similarity_matrix(mat)`: Genera la matriz de similitud coseno.

###  Ejecución principal

- `process_documents()`: Funcion principal que sen encarga de cargar datos, procesar, calcular TF‑IDF y guardar resultados. 