# Trabalho Final de FIA — Raciocínio Espacial Neuro-Simbólico com LTN

Este repositório contém a implementação do trabalho final da disciplina **Fundamentos de Inteligência Artificial (FIA)**, cujo objetivo é construir um agente neuro-simbólico capaz de realizar **raciocínio espacial** sobre objetos em um ambiente 2D inspirado no dataset CLEVR.

# Alunos: #
Marcele Azevedo de Paula Oliveira - 22353160

A proposta utiliza a ideia de **Logic Tensor Networks (LTN)**, combinando redes neurais com regras lógicas diferenciáveis para avaliar relações espaciais entre objetos, como:

* `LeftOf(x, y)`
* `RightOf(x, y)`
* `Below(x, y)`
* `Above(x, y)`
* `CloseTo(x, y)`
* `InBetween(x, y, z)`
* `CanStack(x, y)`

O foco do trabalho não é apenas classificar objetos, mas testar se regras de **reasoning** conseguem ser aprendidas a partir de poucos dados e generalizadas para novas cenas.

---

## 1. Objetivo do trabalho

O objetivo é gerar uma cena 2D com múltiplos objetos e treinar predicados neurais com regras lógicas para realizar raciocínio espacial.

Diferentemente de um problema tradicional de classificação, em que o modelo aprende com muitas imagens para identificar categorias, neste trabalho a ideia é:

1. Treinar usando uma única cena, ou seja, uma única “imagem” contendo vários objetos.
2. Gerar muitos casos de raciocínio a partir dos pares e trios de objetos dessa cena.
3. Testar a generalização das regras em novas cenas aleatórias.
4. Avaliar os resultados usando `satAgg`, acurácia, precisão, recall e F1 Score.

---

## 2. Alterações posteriores ao enunciado

Durante as orientações do professor, algumas adaptações foram consideradas para adequar melhor o trabalho ao cenário 2D.

### 2.1 Cinco classes de objetos

O professor esclareceu que não são apenas cinco objetos no total, mas sim **cinco classes de objetos**, com **cinco objetos de cada classe**.

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

### 2.2 Substituição de formas

O enunciado original mencionava formas como cilindro e cone. Como o trabalho foi desenvolvido em um espaço 2D, essas formas foram adaptadas:

```text
cilindro → elipse
cone → retângulo
```

Portanto, as cinco formas usadas foram:

```text
circle, square, ellipse, rectangle, triangle
```

### 2.3 Cena única para treino

Também foi esclarecido que o treinamento deve ocorrer a partir de uma única cena com 25 objetos, e não a partir de um grande conjunto de imagens.

A partir dessa única cena, são gerados vários casos de reasoning:

```text
25 objetos → 600 pares ordenados
25 objetos → 13800 trios ordenados
```

Depois, o modelo é testado em cinco novas cenas aleatórias.

### 2.4 Tratamento de overlapping

Como os objetos ocupam o mesmo espaço 2D, foi implementado um controle para evitar sobreposição visual excessiva entre eles. A geração da cena considera largura, altura e tamanho dos objetos para reduzir casos de overlapping.

---

## 3. Representação dos objetos

Cada objeto é representado por um vetor de 11 características:

```text
[x, y, red, green, blue, circle, square, ellipse, rectangle, triangle, size]
```

A estrutura é:

| Índice | Significado         |
| ------ | ------------------- |
| 0      | posição x           |
| 1      | posição y           |
| 2      | cor vermelha        |
| 3      | cor verde           |
| 4      | cor azul            |
| 5      | círculo             |
| 6      | quadrado            |
| 7      | elipse              |
| 8      | retângulo           |
| 9      | triângulo           |
| 10     | tamanho normalizado |

As cores são representadas em one-hot:

```text
red   → [1, 0, 0]
green → [0, 1, 0]
blue  → [0, 0, 1]
```

As formas também são representadas em one-hot:

```text
circle    → [1, 0, 0, 0, 0]
square    → [0, 1, 0, 0, 0]
ellipse   → [0, 0, 1, 0, 0]
rectangle → [0, 0, 0, 1, 0]
triangle  → [0, 0, 0, 0, 1]
```

Os tamanhos usados foram:

```text
1x1 → pequeno
2x2 → médio
3x3 → grande
```

No vetor, o tamanho é normalizado:

```text
1x1 → 1/3
2x2 → 2/3
3x3 → 1
```

---

## 4. Relações espaciais implementadas

Foram implementadas relações espaciais entre objetos usando suas posições no plano 2D.

### 4.1 Relações horizontais

```text
LeftOf(x, y)
RightOf(x, y)
```

Regras utilizadas:

```text
¬LeftOf(x, x)

LeftOf(x, y) → ¬LeftOf(y, x)

LeftOf(x, y) ↔ RightOf(y, x)

LeftOf(x, y) ∧ LeftOf(y, z) → LeftOf(x, z)
```

### 4.2 Relações verticais

```text
Below(x, y)
Above(x, y)
```

Regras utilizadas:

```text
¬Below(x, x)

Below(x, y) → ¬Below(y, x)

Below(x, y) ↔ Above(y, x)

Below(x, y) ∧ Below(y, z) → Below(x, z)
```

### 4.3 Proximidade

```text
CloseTo(x, y)
```

A relação `CloseTo` é baseada na distância euclidiana entre dois objetos.

Também foi usada a regra de simetria:

```text
CloseTo(x, y) ↔ CloseTo(y, x)
```

### 4.4 Relação entre três objetos

```text
InBetween(x, y, z)
```

A relação `InBetween` verifica se um objeto está entre outros dois objetos no eixo horizontal ou vertical.

Foram usadas regras de consistência com os predicados:

```text
LeftOf
RightOf
Below
Above
```

### 4.5 Empilhamento

```text
CanStack(x, y)
```

A relação `CanStack(x, y)` verifica se o objeto `x` pode ser empilhado sobre o objeto `y`.

Ela foi tratada como uma regra composta, em vez de apenas como uma classificação direta:

```text
CanStack(x, y) =
Above(x, y)
∧ BaseEmpilhavel(y)
∧ (SameSize(x, y) ∨ CentroideEstavel(x, y))
```

Como o cone foi adaptado para retângulo, a base não empilhável foi considerada como:

```text
triangle ou rectangle
```

---

## 5. Raciocínio composto

Além das relações espaciais simples, foram implementadas consultas compostas.

### 5.1 Objeto pequeno abaixo de uma elipse e à esquerda de um quadrado

Adaptação da consulta original com cilindro:

```text
∃x (
    IsSmall(x)
    ∧ ∃y (IsEllipse(y) ∧ Below(x, y))
    ∧ ∃z (IsSquare(z) ∧ LeftOf(x, z))
)
```

### 5.2 Retângulo verde entre dois objetos

Adaptação da consulta original com cone:

```text
∃x, y, z (
    IsRectangle(x)
    ∧ IsGreen(x)
    ∧ InBetween(x, y, z)
)
```

### 5.3 Triângulos próximos devem ter o mesmo tamanho

```text
∀x, y (
    IsTriangle(x)
    ∧ IsTriangle(y)
    ∧ CloseTo(x, y)
    → SameSize(x, y)
)
```

---
### 6. Uso de LTN e LTNtorch

A implementação principal utiliza predicados neurais em PyTorch e operadores de lógica fuzzy diferenciável para representar regras do tipo LTN.

Foram usados operadores como:

NOT
AND
OR
IMPLIES
IFF
satAgg

Além disso, foi adicionada uma seção explícita usando a API oficial do LTNtorch, com:

ltn.Predicate
ltn.Variable
ltn.Connective
ltn.Quantifier
ltn.fuzzy_ops.SatAgg

Essa seção demonstra a formulação das regras LeftOf e RightOf usando diretamente a estrutura do LTNtorch.

As regras avaliadas nessa seção são:

∀x ¬LeftOf(x,x)

∀x,y LeftOf(x,y) → ¬LeftOf(y,x)

∀x,y LeftOf(x,y) ↔ RightOf(y,x)

∀x,y,z (LeftOf(x,y) ∧ LeftOf(y,z)) → LeftOf(x,z)

O valor SatAgg_LTNtorch representa o grau de satisfação da base lógica usando a API oficial do LTNtorch.

### 7. Tecnologias utilizadas

O projeto foi desenvolvido em Python, usando o Google Colab.

Bibliotecas principais:

numpy
pandas
matplotlib
torch
LTNtorch
### 8. Como executar
### 8.1 Executando no Google Colab

A forma recomendada de execução é pelo Google Colab.

Passos:

Abrir o notebook TF_IA.ipynb.
Selecionar:
Ambiente de execução → Executar tudo
Caso necessário, instalar o LTNtorch executando:
!pip install LTNtorch
Rodar as células na ordem em que aparecem no notebook.
### 8.2 Executando localmente

Também é possível executar localmente usando Jupyter Notebook.

Primeiro, instale as dependências:

pip install -r requirements.txt

Depois, abra o notebook:

jupyter notebook TF_IA.ipynb

Ou, se estiver usando Jupyter Lab:

jupyter lab TF_IA.ipynb
### 9. Dependências

O arquivo requirements.txt deve conter:

numpy
pandas
matplotlib
torch
LTNtorch
notebook
11. Resultados

Os resultados são salvos na pasta results/.

Arquivos esperados:

Arquivo	Conteúdo
resultados_left_right.csv	Métricas e satisfação das regras para LeftOf e RightOf
resultados_below_above.csv	Métricas e satisfação das regras para Below e Above
resultados_canstack.csv	Métricas da regra composta CanStack
resultados_inbetween.csv	Métricas e satisfação lógica de InBetween
resultados_tarefa4.csv	Resultados das consultas compostas
resumo_resultados.csv	Média geral dos principais resultados

As métricas usadas são:

Accuracy
Precision
Recall
F1 Score
satAgg

A avaliação é feita em:

1 cena de treino
5 cenas de teste aleatórias

Os resultados detalhados e exemplos de execução serão adicionados posteriormente.

### 10. Figuras

As figuras geradas pelo notebook devem ser salvas na pasta figures/.

Exemplos:

figures/cena_treino.png
figures/cena_teste_1.png
figures/cena_teste_2.png

Essas imagens representam as cenas 2D geradas, contendo 25 objetos com formas, cores, tamanhos e posições distintas.
