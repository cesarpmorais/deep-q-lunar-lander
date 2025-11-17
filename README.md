# Deep Q-Learning LunarLander

Este projeto implementa um agente DQN para o ambiente LunarLander-v3 do Gymnasium, incluindo busca de hiperparâmetros automatizada.

## Requisitos

- Python 3.10+
- Microsoft C++ Build Tools
- GPU NVIDIA (opcional, recomendado)

### Pacotes necessários

Instale todos os pacotes com:
```bash
pip install -r requirements.txt
```

## Estrutura dos arquivos

- `dqn.py`: implementação do agente DQN e funções de treino/avaliação.
- `hyperparameter_search.py`: busca de hiperparâmetros (grid/random search).
- `hyperparameter_pipeline.py`: pipeline em duas fases (triagem rápida + refino top-K).
- `results/`: diretório onde os resultados e históricos são salvos.

## Como executar o agente DQN

Para treinar e avaliar o agente padrão:
```bash
python3 dqn.py
```
Isso irá treinar o agente, salvar o melhor modelo e gerar o gráfico `training_results.png`.

## Como executar a busca de hiperparâmetros

Para rodar a busca grid search:
```bash
python3 hyperparameter_search.py
```
Os resultados serão salvos em `hyperparameter_search_results.json` e os históricos de cada run em `results/hyper_search_histories/`.

## Como executar a pipeline completa (duas fases)

Para rodar a pipeline de triagem + refino:
```bash
python3 hyperparameter_pipeline.py
```
Isso irá:
- Rodar a fase A (triagem rápida, poucos episódios, 1 run por config)
- Selecionar os top-K configs
- Rodar a fase B (refino, mais episódios, múltiplos runs por config)
- Salvar os resultados em arquivos JSON com timestamp em `results/`
- Salvar históricos completos em `.npz` para cada run

## Observações
- O uso de GPU acelera bastante o treinamento.
- Os arquivos `.npz` permitem análise detalhada posterior sem inflar o JSON.
- Recomenda-se rodar a pipeline em ambiente com boa capacidade de processamento
