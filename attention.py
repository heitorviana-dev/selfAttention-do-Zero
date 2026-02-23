import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Aplica a função softmax linha a linha em uma matriz 2D.
    
    A função softmax converte valores numéricos em probabilidades que somam 1.
    Usa o truque de estabilidade numérica (subtrair o máximo de cada linha)
    para evitar overflow/underflow durante a exponenciação.
    
    Fórmula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Args:
        x (np.ndarray): Matriz 2D NumPy contendo os valores de entrada.
    
    Returns:
        np.ndarray: Matriz 2D com softmax aplicado linha a linha, onde cada linha soma 1.0.
    
    Raises:
        ValueError: Se x não for uma matriz 2D.
    """
    # Valida que a entrada é uma matriz 2D
    if x.ndim != 2:
        raise ValueError(f"Esperado array 2D, mas recebeu array com {x.ndim} dimensões")
    # Subtrai o máximo de cada linha para estabilidade numérica
    shifted_values = x - np.max(x, axis=1, keepdims=True)
    
    # Calcula exponenciais dos valores deslocados
    exponentials = np.exp(shifted_values)
    
    # Calcula a soma de cada linha
    row_sums = np.sum(exponentials, axis=1, keepdims=True)
    
    # Normaliza para obter probabilidades
    return exponentials / row_sums

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Implementa o mecanismo de Scaled Dot-Product Attention.
    
    Fórmula: Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
    
    O scaling factor (√dₖ) é crucial para prevenir que os produtos escalares
    cresçam muito em magnitude quando a dimensionalidade é alta, o que
    causaria saturação do softmax e gradientes muito pequenos.
    
    Args:
        Q (np.ndarray): Matriz de queries (n × dₖ)
        K (np.ndarray): Matriz de keys (m × dₖ)
        V (np.ndarray): Matriz de values (m × dᵥ)
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Tupla contendo:
            - output: Resultado da attention (n × dᵥ)
            - attention_weights: Pesos de atenção após softmax (n × m)
    
    Raises:
        ValueError: Se Q, K ou V não forem matrizes 2D.
        ValueError: Se as dimensões de Q e K não forem compatíveis (Q.shape[1] != K.shape[1]).
        ValueError: Se as dimensões de K e V não forem compatíveis (K.shape[0] != V.shape[0]).
    """
    # Valida que todas as entradas são matrizes 2D
    if Q.ndim != 2:
        raise ValueError(f"Q deve ser uma matriz 2D, mas tem {Q.ndim} dimensões")
    if K.ndim != 2:
        raise ValueError(f"K deve ser uma matriz 2D, mas tem {K.ndim} dimensões")
    if V.ndim != 2:
        raise ValueError(f"V deve ser uma matriz 2D, mas tem {V.ndim} dimensões")
    
    # Valida compatibilidade de dimensões: Q e K devem ter a mesma dimensionalidade (dₖ)
    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Dimensões incompatíveis entre Q e K: "
            f"Q.shape[1]={Q.shape[1]} != K.shape[1]={K.shape[1]}. "
            f"Q e K devem ter a mesma dimensionalidade (dₖ)."
        )
    
    # Valida que o número de keys corresponde ao número de values
    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis entre K e V: "
            f"K.shape[0]={K.shape[0]} != V.shape[0]={V.shape[0]}. "
            f"O número de keys deve ser igual ao número de values."
        )
    # Extrai a dimensionalidade das keys (dₖ)
    d_k = K.shape[1]
    
    # Calcula o fator de escalonamento
    scaling_factor = np.sqrt(d_k)
    
    # Calcula os scores: QK
    scores = np.matmul(Q, K.T)
    
    # Normaliza os scores dividindo por √dₖ
    scaled_scores = scores / scaling_factor
    
    # Aplica softmax linha a linha para obter os pesos de atenção
    attention_weights = softmax(scaled_scores)
    
    # Multiplica os pesos pelas values para obter o output final
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
