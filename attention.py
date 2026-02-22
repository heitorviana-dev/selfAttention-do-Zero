import numpy as np


def softmax(x):
    """
    Aplica a função softmax linha a linha em uma matriz 2D.
    
    A função softmax converte valores numéricos em probabilidades que somam 1.
    Usa o truque de estabilidade numérica (subtrair o máximo de cada linha)
    para evitar overflow/underflow durante a exponenciação.
    
    Fórmula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Parâmetros:
        x: Matriz 2D NumPy contendo os valores de entrada.
    
    Retorna:
        Matriz 2D com softmax aplicado linha a linha, onde cada linha soma 1.0.
    """
    # Subtrai o máximo de cada linha para estabilidade numérica
    shifted_values = x - np.max(x, axis=1, keepdims=True)
    
    # Calcula exponenciais dos valores deslocados
    exponentials = np.exp(shifted_values)
    
    # Calcula a soma de cada linha
    row_sums = np.sum(exponentials, axis=1, keepdims=True)
    
    # Normaliza para obter probabilidades
    return exponentials / row_sums


def scaled_dot_product_attention(Q, K, V):
    """
    Implementa o mecanismo de Scaled Dot-Product Attention.
    
    Fórmula: Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
    
    O scaling factor (√dₖ) é crucial para prevenir que os produtos escalares
    cresçam muito em magnitude quando a dimensionalidade é alta, o que
    causaria saturação do softmax e gradientes muito pequenos.
    
    Parâmetros:
        Q: Matriz de queries (n × dₖ)
        K: Matriz de keys (m × dₖ)
        V: Matriz de values (m × dᵥ)
    
    Retorna:
        Tupla contendo:
        - output: Resultado da attention (n × dᵥ)
        - attention_weights: Pesos de atenção após softmax (n × m)
    """
    # Extrai a dimensionalidade das keys (dₖ)
    d_k = K.shape[1]
    
    # Calcula o fator de escalonamento
    scaling_factor = np.sqrt(d_k)
    
    # Calcula os scores: QKᵀ
    scores = np.matmul(Q, K.T)
    
    # Normaliza os scores dividindo por √dₖ
    scaled_scores = scores / scaling_factor
    
    # Aplica softmax linha a linha para obter os pesos de atenção
    attention_weights = softmax(scaled_scores)
    
    # Multiplica os pesos pelas values para obter o output final
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
