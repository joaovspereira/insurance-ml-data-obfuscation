![Insurance ML & Data Obfuscation](assets/banner.svg)

[English](README.md) · **Português** · [Portfólio](https://github.com/joaovspereira)

# Insurance ML & Data Obfuscation

## Objetivo

Demonstrar a invariância da regressão linear e avaliar modelos de seguros sem usar o teste para escolher hiperparâmetros.

## O que foi implementado

Corrigi o cálculo de R², a ordem de execução, o escalonamento e a seleção de k. O pipeline ajusta o pré-processamento dentro da validação cruzada e mantém um conjunto de teste separado. A prova numérica usa 180 registros sintéticos de treino e 70 de teste, uma transformação invertível não ortogonal e tolerância de 1e-8 para comparar predições.

## Evidências e execução

[Notebook](notebooks/analysis.ipynb) · [Implementation](insurance.py) · [Synthetic output](reports/synthetic_demo.json)

Foram verificados **4 testes automatizados**. Eles validam a implementação com dados sintéticos; não substituem a avaliação nos dados reais. Veja [VALIDATION.md](VALIDATION.md) para o registro e os limites da verificação.

Execute a partir da raiz do repositório, com Python 3.12:

```bash
python -m pip install -r requirements.txt
python insurance.py
python -m unittest discover -s tests -v
python -m notebook notebooks/analysis.ipynb
```

As [instruções completas](README.md#run-locally) incluem a criação do ambiente virtual. Consulte [data/README.md](data/README.md) para os arquivos e campos esperados.

## Tecnologias

Python · NumPy · pandas · scikit-learn · linear algebra · kNN · OLS · cross-validation

## Limitações e próximos passos

A transformação é reversível e não oferece criptografia nem anonimização. A prova utiliza regressão por mínimos quadrados, intercepto preservado e matriz de treino com posto completo. As métricas antigas não foram usadas como resultado da implementação corrigida; a execução do CSV original segue pendente.

Projeto educacional derivado do Data Science Bootcamp da TripleTen, revisado para publicação em setembro de 2026. Datasets originais não são redistribuídos. O aprendizado central desta revisão é tornar explícitos os pressupostos, as unidades de medida e os limites das conclusões.
