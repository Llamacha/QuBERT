# QuBERT
LlamaRoBERTa es un modelo de lenguaje basado en RoBERTa para el quechua. Nuestro modelo de lenguaje fue pre-entrenado con texto del quechua sureño (Collao y Chanka).

## Acerca del modelo
|Modulo| Descarga |
|------|----------|
| LlamaRoBERTa | [config.json](https://drive.google.com/file/d/1lDaVeJc90TKbBrhxZKZbIfRTPv9VSsOg/view?usp=sharing), [pytorch_model.bin](https://drive.google.com/file/d/16SkLOsfja22kIwExs4NiU5pjrOV7SUdP/view?usp=sharing) |
| Tokenizer | [merges.txt](https://drive.google.com/file/d/1PrM9LMJ9Pmrc8yqKBT1OMRPXD1urkJ1r/view?usp=sharing), [vocab.json](https://drive.google.com/file/d/1i6L13u5P9HVzzmKsNZxe_wICteulIWY5/view?usp=sharing) |

El modelo utiliza un tokenizador Byte-level BPE con un vocabulario de 52000 tokens de subpalabras.

## Usabilidad
Una vez descargado los pesos y el tokenizador en la seccion de arriba es necesario adjuntarlo en un sola carpeta, en este caso fue `LlamaRoBERTa`.

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./LlamaRoBERTa",
    tokenizer="./LlamaRoBERTa"
)
```
Se hace la prueba, la cual esta en fases de mejoras.

```python
fill_mask("allinllachu <mask> allinlla huk wasipita.")
```
    [{'score': 0.23992203176021576,
     'sequence': 'allinllachu nisqaqa allinlla huk wasipita.',
     'token': 334,
     'token_str': ' nisqaqa'},
    {'score': 0.061005301773548126,
     'sequence': 'allinllachu, allinlla huk wasipita.',
     'token': 16,
     'token_str': ','},
     {'score': 0.028720015659928322,
     'sequence': "allinllachu' allinlla huk wasipita.",
     'token': 11,
     'token_str': "'"},
    {'score': 0.012927944771945477,
    'sequence': 'allinllachu kay allinlla huk wasipita.',
    'token': 377,
    'token_str': ' kay'},
    {'score': 0.01230092253535986,
    'sequence': 'allinllachu. allinlla huk wasipita.',
     'token': 18,
    'token_str': '.'}]
