# Explicação em ordem do código — TF_IA.ipynb

Este arquivo foi criado para complementar o notebook `TF_IA.ipynb`.

  
O notebook continua sendo o código principal; este arquivo apenas destaca as partes mais importantes, explica o papel de cada bloco e mostra a ordem lógica da implementação.

---

# 1. Objetivo geral do código

O código implementa um agente de **raciocínio espacial neuro-simbólico** inspirado em Logic Tensor Networks.

O objetivo não é apenas classificar formas ou cores.  
O foco é aprender e avaliar relações espaciais entre objetos, como:

```text
LeftOf(x, y)
RightOf(x, y)
Below(x, y)
Above(x, y)
CloseTo(x, y)
InBetween(x, y, z)
CanStack(x, y)
```

A partir de uma única cena com vários objetos, o código gera pares e trios de objetos para treinar predicados neurais e avaliar regras lógicas fuzzy.

---

# 2. Adaptações feitas em relação ao enunciado

Durante o desenvolvimento, foram incorporadas orientações posteriores do professor.

## 2.1 Quantidade de objetos

Em vez de apenas cinco objetos no total, foram usadas cinco classes de objetos, com cinco objetos por classe.

Assim, cada cena possui:

```text
5 círculos
5 quadrados
5 elipses
5 retângulos
5 triângulos
```

Total:

```text
5 classes × 5 objetos por classe = 25 objetos
```

## 2.2 Adaptação das formas para 2D

Como a cena é bidimensional, as formas tridimensionais do enunciado foram adaptadas:

```text
cilindro → elipse
cone → retângulo
```

Portanto, as formas finais usadas no código foram:

```python
formas = ["circle", "square", "ellipse", "rectangle", "triangle"]
```

## 2.3 Controle de overlapping

O código também implementa tratamento para evitar sobreposição excessiva entre os objetos.  
Isso é importante porque os objetos são desenhados em uma cena 2D e precisam ficar visualmente distinguíveis.

---

# 3. Bibliotecas utilizadas

O notebook começa importando bibliotecas para:

- manipulação numérica;
- criação de tabelas;
- visualização das cenas;
- redes neurais;
- lógica fuzzy e LTNtorch.

Principais bibliotecas:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import ltn
```

O `LTNtorch` é utilizado de forma explícita em uma seção final do notebook, para demonstrar o uso de `ltn.Predicate`, `ltn.Variable`, `ltn.Connective`, `ltn.Quantifier` e `SatAgg`.

---

# 4. Configurações gerais do experimento

O código define as configurações principais da cena:

```python
N_POR_CLASSE = 5

cores = ["red", "green", "blue"]

formas = ["circle", "square", "ellipse", "rectangle", "triangle"]

N_OBJETOS = len(formas) * N_POR_CLASSE

tamanhos = [1, 2, 3]

BASE_VISUAL = 0.018
```

Essas variáveis controlam:

- quantos objetos de cada classe serão criados;
- quais cores existem;
- quais formas existem;
- quais tamanhos são possíveis;
- o tamanho visual dos objetos no gráfico.

---

# 5. Representação vetorial dos objetos

Cada objeto é convertido em um vetor de 11 posições.

A estrutura do vetor é:

```text
[x, y, red, green, blue, circle, square, ellipse, rectangle, triangle, size]
```

A função responsável por isso é:

```python
def criar_vetor(objeto):
    x = objeto["x"]
    y = objeto["y"]

    cor_one_hot = [0, 0, 0]
    indice_cor = cores.index(objeto["cor"])
    cor_one_hot[indice_cor] = 1

    forma_one_hot = [0, 0, 0, 0, 0]
    indice_forma = formas.index(objeto["forma"])
    forma_one_hot[indice_forma] = 1

    tamanho_normalizado = objeto["tamanho"] / 3

    vetor = [x, y] + cor_one_hot + forma_one_hot + [tamanho_normalizado]

    return vetor
```

## Por que essa parte é importante?

Os modelos neurais não recebem diretamente objetos desenhados.  
Eles recebem vetores numéricos.

Exemplo conceitual:

```text
objeto vermelho, círculo, pequeno, na posição (0.2, 0.7)
```

é transformado em algo como:

```text
[0.2, 0.7, 1, 0, 0, 1, 0, 0, 0, 0, 0.333]
```

Essa representação permite que os predicados neurais aprendam relações espaciais a partir dos atributos dos objetos.

---

# 6. Dimensões dos objetos e tratamento de overlapping

O código define dimensões diferentes para cada forma:

```python
def dimensoes_objeto(forma, tamanho):
    escala = BASE_VISUAL * tamanho

    if forma == "circle":
        largura = 2 * escala
        altura = 2 * escala

    elif forma == "square":
        largura = 2 * escala
        altura = 2 * escala

    elif forma == "rectangle":
        largura = 2.8 * escala
        altura = 1.4 * escala

    elif forma == "ellipse":
        largura = 3.0 * escala
        altura = 1.5 * escala

    elif forma == "triangle":
        largura = 2.4 * escala
        altura = 2.3 * escala

    return largura, altura
```

Depois, antes de inserir um objeto na cena, o código verifica se ele ficaria muito sobreposto a outro objeto:

```python
def tem_sobreposicao(x, y, forma, tamanho, objetos, margem=0.015):
    largura, altura = dimensoes_objeto(forma, tamanho)

    for obj in objetos:
        largura2, altura2 = dimensoes_objeto(obj["forma"], obj["tamanho"])

        distancia_x = abs(x - obj["x"])
        distancia_y = abs(y - obj["y"])

        limite_x = (largura / 2) + (largura2 / 2) + margem
        limite_y = (altura / 2) + (altura2 / 2) + margem

        if distancia_x < limite_x and distancia_y < limite_y:
            return True

    return False
```

## Por que essa parte é importante?

Sem isso, vários objetos poderiam ser gerados em cima uns dos outros.  
Isso prejudicaria a visualização da cena e também poderia dificultar a interpretação das relações espaciais.

---

# 7. Geração da cena

A função `gerar_objetos` cria uma cena com 25 objetos.

Ela garante que existam cinco objetos de cada forma:

```python
def gerar_objetos(n_por_classe=5):
    objetos = []

    formas_da_cena = []

    for forma in formas:
        for _ in range(n_por_classe):
            formas_da_cena.append(forma)

    np.random.shuffle(formas_da_cena)
```

Depois, para cada forma, o código sorteia:

- cor;
- tamanho;
- posição `x`;
- posição `y`.

Só adiciona o objeto se ele não gerar sobreposição excessiva.

Cada objeto fica no formato:

```python
objeto = {
    "id": len(objetos),
    "x": x,
    "y": y,
    "cor": cor,
    "forma": forma,
    "tamanho": tamanho,
    "tamanho_nome": f"{tamanho}x{tamanho}"
}
```

E depois recebe o vetor de características:

```python
objeto["vetor"] = criar_vetor(objeto)
```

---

# 8. Visualização da cena

A função `plotar_cenario` desenha a cena 2D.

Ela representa:

- círculos com `Circle`;
- quadrados e retângulos com `Rectangle`;
- elipses com `Ellipse`;
- triângulos com `Polygon`.

Cada objeto recebe também seu `id` no gráfico.

Essa parte é importante porque permite visualizar as cenas salvas em `figures/`, como:

```text
figures/cena_treino.png
figures/cena_teste_1.png
figures/cena_teste_2.png
...
```

---

# 9. Cena de treino e cenas de teste

O experimento usa:

```text
1 cena de treino
5 cenas de teste
```

A cena de treino é usada para ajustar os predicados neurais e as regras.

As cinco cenas de teste servem para avaliar se o raciocínio generaliza para novas disposições aleatórias de objetos.

Exemplo:

```python
cena_treino = gerar_objetos(n_por_classe=N_POR_CLASSE)
```

E para teste:

```python
cenas_teste = []

for i in range(5):
    cena = gerar_objetos(n_por_classe=N_POR_CLASSE)
    cenas_teste.append(cena)
```

---

# 10. Relações espaciais verdadeiras

O código define funções que funcionam como gabarito.

Exemplo para relações horizontais:

```python
def left_of(a, b):
    return a["x"] < b["x"]


def right_of(a, b):
    return a["x"] > b["x"]
```

Relações verticais:

```python
def below(a, b):
    return a["y"] < b["y"]


def above(a, b):
    return a["y"] > b["y"]
```

Proximidade:

```python
def close_to(a, b, threshold=0.20):
    distancia = np.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)
    return distancia < threshold
```

Mesmo tamanho:

```python
def same_size(a, b):
    return a["tamanho"] == b["tamanho"]
```

Relação ternária:

```python
def in_between(x, y, z):
    return in_between_horizontal(x, y, z) or in_between_vertical(x, y, z)
```

## Por que essa parte é importante?

Essas funções são usadas para gerar os rótulos verdadeiros dos pares e trios de objetos.  
Elas são o "gabarito" usado para treinar e avaliar os modelos.

---

# 11. Geração de pares e trios

Com 25 objetos, o código gera:

```text
600 pares ordenados
13800 trios ordenados
```

## Pares

Os pares são usados para relações binárias:

```text
LeftOf(x, y)
RightOf(x, y)
Below(x, y)
Above(x, y)
CloseTo(x, y)
SameSize(x, y)
CanStack(x, y)
```

## Trios

Os trios são usados para:

```text
InBetween(x, y, z)
```

Essa etapa transforma uma única cena em muitos exemplos de raciocínio.

---

# 12. Métricas de avaliação

A função `calcular_metricas` calcula:

```text
TP
TN
FP
FN
Accuracy
Precision
Recall
F1
```

Essas métricas são usadas para avaliar os predicados nas cinco cenas de teste.

A análise principal deve considerar as cenas de teste, e não apenas a cena de treino, porque o objetivo é avaliar generalização.

---

# 13. Modelo neural para relações binárias

O modelo binário recebe dois objetos, concatena seus vetores e retorna um valor entre 0 e 1.

```python
class RelacaoBinariaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        entrada = torch.cat([x, y], dim=1)
        return self.net(entrada)
```

Como cada objeto tem 11 características, dois objetos juntos têm:

```text
11 + 11 = 22 entradas
```

Esse modelo é usado para predicados como:

```text
LeftOf
RightOf
Below
Above
CloseTo
SameSize
CanStack
```

---

# 14. Modelo neural para relação ternária

Para `InBetween(x, y, z)`, são usados três objetos.

Como cada objeto tem 11 características:

```text
11 + 11 + 11 = 33 entradas
```

O modelo ternário é:

```python
class RelacaoTernariaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(33, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y, z):
        entrada = torch.cat([x, y, z], dim=1)
        return self.net(entrada)
```

---

# 15. Operadores fuzzy

O código implementa operadores fuzzy diferenciáveis:

```python
def fuzzy_not(p):
    return 1 - p


def fuzzy_and(p, q):
    return p * q


def fuzzy_or(p, q):
    return p + q - p * q


def fuzzy_implies(p, q):
    return 1 - p + p * q


def fuzzy_iff(p, q):
    return fuzzy_and(
        fuzzy_implies(p, q),
        fuzzy_implies(q, p)
    )
```

Esses operadores permitem transformar regras lógicas em valores numéricos diferenciáveis.

Exemplo:

```text
LeftOf(x,y) → ¬LeftOf(y,x)
```

vira uma expressão matemática com valores entre 0 e 1.

---

# 16. Raciocínio horizontal: LeftOf e RightOf

O código treina modelos para:

```text
LeftOf(x, y)
RightOf(x, y)
```

As regras usadas são:

```text
¬LeftOf(x,x)

LeftOf(x,y) → ¬LeftOf(y,x)

LeftOf(x,y) ↔ RightOf(y,x)

LeftOf(x,y) ∧ LeftOf(y,z) → LeftOf(x,z)
```

A satisfação dessas regras é agregada em um valor médio chamado `satAgg`.

A perda combina:

```text
erro supervisionado + erro das regras
```

Conceitualmente:

```python
loss_total = loss_dados + lambda_regras * loss_regras
```

onde:

```python
loss_regras = 1 - satAgg
```

Assim, o treinamento tenta:

- acertar os dados;
- aumentar a satisfação das regras lógicas.

---

# 17. Raciocínio vertical: Below e Above

A mesma ideia é aplicada às relações verticais:

```text
Below(x, y)
Above(x, y)
```

Regras usadas:

```text
¬Below(x,x)

Below(x,y) → ¬Below(y,x)

Below(x,y) ↔ Above(y,x)

Below(x,y) ∧ Below(y,z) → Below(x,z)
```

Assim como na parte horizontal, o objetivo é treinar predicados neurais que respeitem regras lógicas.

---

# 18. CloseTo

A relação `CloseTo(x, y)` identifica objetos próximos pela distância euclidiana.

A regra lógica principal é a simetria:

```text
CloseTo(x,y) ↔ CloseTo(y,x)
```

Como `CloseTo` pode ter menos exemplos positivos do que negativos, o código usa ponderação para lidar melhor com esse desbalanceamento.

---

# 19. SameSize

A relação `SameSize(x, y)` verifica se dois objetos possuem o mesmo tamanho.

Ela é importante principalmente para a Tarefa 4, porque a regra dos triângulos usa:

```text
CloseTo(x,y) → SameSize(x,y)
```

para pares de triângulos.

---

# 20. InBetween

A relação ternária `InBetween(x, y, z)` avalia se o objeto `x` está entre `y` e `z`.

Ela pode ser verdadeira se `x` estiver entre os outros dois:

- horizontalmente;
- verticalmente.

A definição usa consistência com:

```text
LeftOf
RightOf
Below
Above
```

Exemplo conceitual:

```text
x está entre y e z se y está à esquerda de x e z está à direita de x,
ou o contrário.
```

Também é avaliada a simetria dos extremos:

```text
InBetween(x,y,z) ↔ InBetween(x,z,y)
```

---

# 21. CanStack

A relação `CanStack(x, y)` verifica se o objeto `x` pode ser empilhado sobre o objeto `y`.

A melhor formulação usada no notebook é a regra composta:

```text
CanStack(x, y) =
Above(x, y)
∧ BaseEmpilhavel(y)
∧ (SameSize(x, y) ∨ CentroideEstavel(x, y))
```

Como `cone` foi adaptado para `rectangle`, as bases não empilháveis são:

```text
triangle
rectangle
```

Essa abordagem representa melhor o reasoning do que tratar `CanStack` apenas como classificação direta.

---

# 22. Tarefa 4: raciocínio composto

A Tarefa 4 combina atributos e relações espaciais.

## 22.1 Objeto pequeno abaixo de uma elipse e à esquerda de um quadrado

Adaptação do enunciado original com cilindro:

```text
∃x (
    IsSmall(x)
    ∧ ∃y (IsEllipse(y) ∧ Below(x,y))
    ∧ ∃z (IsSquare(z) ∧ LeftOf(x,z))
)
```

## 22.2 Retângulo verde entre dois objetos

Adaptação do enunciado original com cone:

```text
∃x,y,z (
    IsRectangle(x)
    ∧ IsGreen(x)
    ∧ InBetween(x,y,z)
)
```

## 22.3 Triângulos próximos devem ter o mesmo tamanho

```text
∀x,y (
    IsTriangle(x)
    ∧ IsTriangle(y)
    ∧ CloseTo(x,y)
    → SameSize(x,y)
)
```

Essas consultas mostram o uso de raciocínio composto, pois combinam:

- cor;
- forma;
- tamanho;
- relação espacial;
- regra lógica.

---

# 23. Uso explícito do LTNtorch

Para deixar claro o uso da API oficial do LTNtorch, o notebook inclui uma seção específica com:

```python
ltn.Predicate
ltn.Variable
ltn.Connective
ltn.Quantifier
ltn.fuzzy_ops.SatAgg
```

Nessa seção, os modelos treinados de `LeftOf` e `RightOf` são encapsulados como predicados LTN.

Exemplo conceitual:

```python
LeftOf_LTN = ltn.Predicate(model=RelacaoBinariaLTN(modelo_left_rule))
RightOf_LTN = ltn.Predicate(model=RelacaoBinariaLTN(modelo_right_rule))
```

Depois são criadas variáveis:

```python
x_ltn = ltn.Variable("x_ltn", X_ltn_treino)
y_ltn = ltn.Variable("y_ltn", X_ltn_treino)
z_ltn = ltn.Variable("z_ltn", X_ltn_treino)
```

E regras como:

```text
∀x ¬LeftOf(x,x)

∀x,y LeftOf(x,y) → ¬LeftOf(y,x)

∀x,y LeftOf(x,y) ↔ RightOf(y,x)

∀x,y,z (LeftOf(x,y) ∧ LeftOf(y,z)) → LeftOf(x,z)
```

são avaliadas com `SatAgg_LTNtorch`.

Essa seção demonstra formalmente a relação do trabalho com Logic Tensor Networks.

---

# 24. Salvamento dos resultados

Os resultados finais são salvos na pasta `results/`.

Arquivos esperados:

```text
results/resultados_left_right.csv
results/resultados_below_above.csv
results/resultados_close.csv
results/resultados_inbetween.csv
results/resultados_canstack.csv
results/resultados_tarefa4.csv
results/resumo_resultados.csv
```

Esses arquivos permitem que o professor veja os resultados sem precisar rodar o notebook inteiro.

---

# 25. Salvamento das figuras

As figuras são salvas na pasta `figures/`.

Arquivos esperados:

```text
figures/cena_treino.png
figures/cena_teste_1.png
figures/cena_teste_2.png
figures/cena_teste_3.png
figures/cena_teste_4.png
figures/cena_teste_5.png
```

Essas figuras mostram visualmente as cenas usadas para treino e teste.

---

# 26. Observação sobre métricas de 100%

Algumas métricas podem aparecer como 100% no notebook.

Esses casos geralmente representam:

```text
avaliação na própria cena de treino
verificação interna
teste de funcionamento da função de métricas
```

A avaliação mais importante é a das cinco cenas de teste, porque elas indicam se o modelo generaliza para novas cenas.

---

# 27. Ordem recomendada para leitura do notebook

A ordem recomendada para acompanhar o notebook é:

```text
1. Imports e configurações
2. Representação dos objetos
3. Geração da cena
4. Visualização
5. Relações verdadeiras
6. Pares e trios
7. Métricas
8. Modelos neurais
9. Operadores fuzzy
10. LeftOf / RightOf
11. Below / Above
12. CloseTo
13. SameSize
14. InBetween
15. CanStack
16. Tarefa 4
17. LTNtorch explícito
18. Salvamento dos resultados
19. Salvamento das figuras
20. Conclusão
```

---

# 28. Conclusão

O código implementa uma abordagem neuro-simbólica para raciocínio espacial em uma cena 2D com 25 objetos.

A partir de uma única cena de treino, são gerados múltiplos exemplos de pares e trios de objetos.  
Os predicados neurais são treinados com dados e regras fuzzy, permitindo avaliar relações espaciais simples e compostas.

A implementação cobre:

```text
relações horizontais
relações verticais
proximidade
relação ternária
empilhamento
raciocínio composto
uso explícito de LTNtorch
avaliação com métricas e satAgg
```

Este arquivo serve como guia de leitura para o notebook `TF_IA.ipynb`.
