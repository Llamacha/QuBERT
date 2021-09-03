# bert-quechua
Bert-quechua es un modelo de lenguage basado en redes Transformers para el quechua, estando aun en fase de desarrollo.Bert-quechua fue pre-entrenado con corpus netamente en quechua sureño (collao y chanka), donde la cantidad de corpus sigue siendo muy poca.Asi mismo bert-quechua fue pre-entrenado mediante el enfoque de RoBERTa: A Robustly Optimized BERT Pretraining Approach.

## Acerca del modelo
|Modulo| Descarga |
|------|----------|
| SpanBERTa | [config.json](https://drive.google.com/file/d/1lDaVeJc90TKbBrhxZKZbIfRTPv9VSsOg/view?usp=sharing), [pytorch_model.bin](https://drive.google.com/file/d/16SkLOsfja22kIwExs4NiU5pjrOV7SUdP/view?usp=sharing) |
| Tokenizer | [merges.txt](https://drive.google.com/file/d/1PrM9LMJ9Pmrc8yqKBT1OMRPXD1urkJ1r/view?usp=sharing), [vocab.json](https://drive.google.com/file/d/1i6L13u5P9HVzzmKsNZxe_wICteulIWY5/view?usp=sharing) |

El modelo utiliza un tokenizador Byte-level BPE con un vocabulario de 52000 tokens de subpalabras.
