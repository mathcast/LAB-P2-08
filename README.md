# LAB P2-08: Alinhamento de Modelo de Linguagem com DPO

## Como baixar o projeto

```bash
git clone https://github.com/mathcast/LAB-P2-08.git
cd LAB-P2-08
```

## Conceito implementado

O projeto implementa o método DPO (Direct Preference Optimization) para alinhamento de modelos de linguagem.

A ideia central é treinar o modelo com base em preferências humanas, comparando duas respostas:

- chosen → resposta desejada (correta/segura)
- rejected → resposta indesejada (incorreta/insegura)

O objetivo é fazer com que o modelo atribua maior probabilidade à resposta chosen do que à rejected.

## Estrutura do dataset

Cada exemplo segue o formato:

```bash
{
  "prompt": "Pergunta do usuário",
  "chosen": "Resposta correta do assistente",
  "rejected": "Resposta incorreta ou indesejada"
}
```

Durante o treinamento, os dados são convertidos para o formato de conversa:

```bash
Usuário: <prompt>
Assistente: <resposta>
```

## Estrutura do repositório

```bash
LAB P2-08/
├── data/
│   └── dataset.jsonl        # Dataset de preferências (prompt, chosen, rejected)
│
├── src/
│   ├── train.py             # Treinamento do modelo com DPO
│   └── test.py              # Teste do modelo treinado
│
├── results/
│   └── final_model/         # Modelo treinado salvo
│
├── requirements.txt         # Dependências do projeto
└── README.md
```

## Como rodar
### 1. Ambiente virtual e dependências

No diretório do projeto:

**Windows (PowerShell):**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
cd src
python train.py
```

O script:

- Carrega o modelo base (GPT-2)
- Carrega o dataset de preferências
- Formata os dados como diálogo (Usuário / Assistente)S
- Aplica o treinamento com DPO
- Salva o modelo em ```results/final_model```

### 3. Testar o modelo

```bash
python test.py
```

O script:

- Recebe prompts de teste
- Gera respostas com o modelo treinado
- Exibe apenas a resposta do assistente

Exemplo de saída:

```bash
Prompt: O que é Python?
Resposta: Python é uma linguagem de programação versátil...
```

## Parâmetro β (beta)

O parâmetro β controla o nível de influência das preferências no treinamento.

- β baixo → modelo mais livre para se adaptar ao dataset
- β alto → modelo mais próximo do comportamento original

No projeto foi utilizado:

```bash
beta = 0.1
```

## Resultados

Após o treinamento:

- O modelo demonstra leve tendência a respostas mais seguras
- Ainda apresenta inconsistências e respostas fora de contexto

Isso ocorre devido a:

- Uso de modelo base pequeno (GPT-2)
- Dataset reduzido
- Limitações naturais do método em pequena escala

## Limitações
- O GPT-2 não é um modelo otimizado para conversação
- O DPO ajusta preferências, mas não ensina comportamento do zero
- O tamanho do dataset impacta diretamente a qualidade das respostas

## Conclusão

O experimento demonstra com sucesso o funcionamento do Direct Preference Optimization (DPO) para alinhamento de modelos de linguagem.

Mesmo com limitações práticas, foi possível observar o efeito do treinamento baseado em preferências, validando o pipeline proposto.

## Requisitos técnicos
- Linguagem: Python 3
- Bibliotecas:
    - transformers
    - datasets
    - trl
    - torch
    - accelerate
