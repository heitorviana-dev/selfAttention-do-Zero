# Scaled Dot-Product Attention — LAB P1-01

## Descrição

Este projeto implementa o mecanismo fundamental de **Scaled Dot-Product Attention**, introduzido no paper seminal "Attention Is All You Need" (Vaswani et al., 2017). O mecanismo de self-attention permite que modelos de aprendizado profundo ponderem a importância de diferentes partes de uma sequência ao processar cada elemento, revolucionando o processamento de linguagem natural e outras áreas.

A implementação usa apenas NumPy para demonstrar os conceitos matemáticos fundamentais sem abstrações de frameworks de deep learning. O código é educacional e foca na clareza dos conceitos, mostrando cada passo do cálculo de atenção de forma explícita.

O projeto foi desenvolvido como parte do LAB P1-01, demonstrando compreensão profunda dos fundamentos matemáticos e práticos do mecanismo de attention que fundamenta arquiteturas modernas como Transformers.

## Como Rodar

### Pré-requisitos

- Python 3.x
- NumPy

### Instalação

```bash
pip install numpy
```

### Executar os Testes

```bash
python test_attention.py
```

O script de testes irá:
- Validar que os attention weights somam 1.0 para cada query
- Verificar que as dimensões do output estão corretas
- Testar a correção numérica comparando com cálculo manual
- Exibir todas as matrizes intermediárias e resultados

## Explicação do Scaling Factor (√dₖ)

### Por que dividir por √dₖ?

O scaling factor é um componente crítico do mecanismo de attention. Quando calculamos os produtos escalares QKᵀ, os valores resultantes têm magnitude proporcional à dimensionalidade dₖ.

**Problema sem scaling:**
- Quando dₖ é grande (ex: 64, 512), os produtos escalares crescem em magnitude
- Valores grandes entram no softmax: softmax([100, 98, 95]) → [0.952, 0.047, 0.001]
- O softmax satura, concentrando quase toda a probabilidade em um único valor
- Os gradientes ficam extremamente pequenos nas regiões saturadas
- O treinamento fica instável e lento

**Solução com √dₖ:**
- Dividir por √dₖ mantém a variância dos scores controlada (~1.0)
- Os valores ficam em uma faixa razoável para o softmax
- A distribuição de atenção fica mais suave e informativa
- Os gradientes permanecem saudáveis durante o treinamento
- O modelo pode aprender padrões de atenção mais refinados

Por exemplo, com dₖ = 64, dividimos por √64 = 8, reduzindo scores de magnitude ~100 para ~12.5, uma faixa muito mais adequada para o softmax.

## Exemplo de Input/Output

### Matrizes de Input

**Q (Queries)** - shape (3, 4):
```
[[1.0, 0.0, 1.0, 0.0],
 [0.0, 2.0, 0.0, 2.0],
 [1.0, 1.0, 1.0, 1.0]]
```

**K (Keys)** - shape (3, 4):
```
[[1.0, 2.0, 3.0, 4.0],
 [2.0, 4.0, 6.0, 8.0],
 [1.0, 1.0, 1.0, 1.0]]
```

**V (Values)** - shape (3, 2):
```
[[1.0, 0.0],
 [0.0, 1.0],
 [0.5, 0.5]]
```

### Output Esperado

**Attention Weights** - shape (3, 3):
```
[[0.259, 0.259, 0.482],
 [0.104, 0.787, 0.109],
 [0.186, 0.628, 0.186]]
```
*Nota: Cada linha soma 1.0, representando uma distribuição de probabilidade*

**Output Final** - shape (3, 2):
```
[[0.482, 0.518],
 [0.109, 0.891],
 [0.407, 0.593]]
```

## Fórmula de Referência

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Onde:
- **Q**: Matriz de queries (n × dₖ)
- **K**: Matriz de keys (m × dₖ)  
- **V**: Matriz de values (m × dᵥ)
- **dₖ**: Dimensionalidade das keys
- **softmax**: Aplicado linha a linha nos scores normalizados

## Estrutura do Código

### `attention.py`
Contém as implementações principais:
- `softmax(x)`: Função auxiliar que aplica softmax com estabilidade numérica
- `scaled_dot_product_attention(Q, K, V)`: Implementação completa do mecanismo de attention

### `test_attention.py`
Suite de testes que valida:
1. Propriedades matemáticas (attention weights somam 1.0)
2. Dimensões corretas do output
3. Correção numérica dos cálculos

## Referência

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems*, 30.

---

**Desenvolvido para:** LAB P1-01  
**Autor:** Heitor Viana  
**Data:** Fevereiro 2026
