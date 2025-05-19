# Determinantes da Pontuação Olímpica

Análise de contagem (Poisson, NB, ZIP, ZINB) sobre medalhas em Londres 2012 e Rio 2016.

## Instalação

```bash
git clone https://github.com/Jonathantakaki/olimpiadas.git
cd olimpiadas
pip install -r requirements.txt
```
## Estrutura do Projeto

- **src/** – scripts Python  
- **data/raw/** – dados brutos (não versionados)  
- **data/sample/** – exemplo de dados para testes  
- **notebooks/** – Jupyter Notebooks exploratórios  
- **docs/** – gráficos e slides exportados  

## Uso

```bash
python src/olimpiadas.py
```
## Resultados Principais

![LogLikelihood dos Modelos](docs/loglik_modelos.png)

> “O ZINB apresentou Loglik -265.551 e destacou `num_eventos` como único preditor significativo.”

## Licença

MIT © Jonathan Takaki
