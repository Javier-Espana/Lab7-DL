# Laboratorio 7 - NLP End-to-End y Embeddings

**CC3092 - Deep Learning y Sistemas Inteligentes**
**Alumno:** Javier Espana (23361)
**Fecha de entrega:** 05 de octubre, 2026

---

## Descripcion

Este repositorio contiene el laboratorio 7 del curso, que cubre el pipeline completo de NLP:
desde la exploracion y preprocesamiento de texto hasta la clasificacion de noticias usando
word embeddings entrenados desde cero y preentrenados.

## Contenido

```
Lab7-DL/
  notebooks/
    lab7_nlp_embeddings.ipynb   Notebook principal con todo el codigo y analisis
  docs/
    Laboratorio_7_NLP_Embeddings.pdf   Enunciado del laboratorio
    informe.tex                         Informe en LaTeX (fuente)
    informe.pdf                         Informe compilado
```

## Datos utilizados

- **WikiText-103** (`Salesforce/wikitext`, `wikitext-103-raw-v1`): corpus para entrenar embeddings.
- **AG News** (`fancyzhx/ag_news`): dataset de clasificacion de noticias en 4 categorias.
- **GloVe** (`glove-wiki-gigaword-100`): embeddings preentrenados de 100 dimensiones.

## Modelos implementados

1. **SGNS propio (PyTorch):** Skip-gram con negative sampling implementado desde cero usando
   `nn.Embedding`. Dos tablas de embeddings: entrada (`W`) y salida (`W_out`).

2. **Word2Vec de gensim:** Skip-gram entrenado con los mismos hiperparametros que el mejor
   modelo SGNS, para validacion cruzada.

3. **GloVe preentrenado:** `glove-wiki-gigaword-100`, cargado directamente con gensim.

## Reproducibilidad

Todos los experimentos usan `SEED = 42`. Para reproducir:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gensim datasets nltk scipy scikit-learn matplotlib numpy tqdm
jupyter notebook notebooks/lab7_nlp_embeddings.ipynb
```

## Vectores del mejor modelo

Los vectores del mejor modelo SGNS se guardan en `models/mejor_sgns.txt` en formato word2vec
estandar. Por el tamano del archivo, se recomienda descargarlo desde el repositorio de GitHub:

https://github.com/Javier-Espana/Lab7-DL.git

## Benchmarks de evaluacion

- **WordSim-353:** correlacion de Spearman para seleccion de hiperparametros.
- **questions-words.txt (14 categorias):** accuracy de analogias con 3CosAdd y 3CosMul.
- **SimLex-999:** evaluacion final de similitud semantica (no usada en seleccion).
