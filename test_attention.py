import numpy as np
from attention import scaled_dot_product_attention


def test_weights_sum_to_one():
    """
    Testa se cada linha dos attention weights soma 1.0.
    """
    print("\n" + "="*60)
    print("TESTE 1: Verificar se attention weights somam 1.0")
    print("="*60)
    
    # Define matrizes de teste pequenas
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    
    K = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    
    V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], dtype=np.float64)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    # Verifica se cada linha soma 1.0
    row_sums = np.sum(attention_weights, axis=1)
    tolerance = 1e-6
    
    test_passed = np.all(np.abs(row_sums - 1.0) < tolerance)
    
    print(f"Somas das linhas: {row_sums}")
    print(f"Todas as linhas somam 1.0? {test_passed}")
    print(f"Status: {'✓ PASSED' if test_passed else '✗ FAILED'}")
    
    return test_passed, Q, K, V, output, attention_weights


def test_output_shape():
    """
    Testa se o output tem a shape correta: (num_queries, dim_values).
    """
    print("\n" + "="*60)
    print("TESTE 2: Verificar shape do output")
    print("="*60)
    
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    
    K = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    
    V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], dtype=np.float64)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    expected_shape = (Q.shape[0], V.shape[1])
    test_passed = output.shape == expected_shape
    
    print(f"Shape esperada: {expected_shape}")
    print(f"Shape obtida: {output.shape}")
    print(f"Status: {'✓ PASSED' if test_passed else '✗ FAILED'}")
    
    return test_passed


def test_numerical_correctness():
    """
    Testa a correção numérica calculando manualmente o resultado esperado.
    """
    print("\n" + "="*60)
    print("TESTE 3: Verificar correção numérica")
    print("="*60)
    
    # Define matrizes simples para facilitar o cálculo manual
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    
    K = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    
    V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ], dtype=np.float64)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    # Cálculo manual passo a passo
    d_k = K.shape[1]  # 4
    scaling_factor = np.sqrt(d_k)  # 2.0
    
    # QKᵀ
    scores = np.matmul(Q, K.T)
    
    # QKᵀ / √dₖ
    scaled_scores = scores / scaling_factor
    
    # softmax linha a linha
    shifted = scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    expected_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Multiplica por V
    expected_output = np.matmul(expected_weights, V)
    
    # Compara com tolerância
    try:
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)
        test_passed = True
    except AssertionError:
        test_passed = False
    
    print(f"Output esperado:\n{expected_output}")
    print(f"\nOutput obtido:\n{output}")
    print(f"\nStatus: {'✓ PASSED' if test_passed else '✗ FAILED'}")
    
    return test_passed


def main():
    """
    Executa todos os testes e imprime resultados.
    """
    print("\n")
    print("="*60)
    print(" TESTES DE SCALED DOT-PRODUCT ATTENTION")
    print("="*60)
    
    # Executa o primeiro teste e captura as matrizes
    test1_passed, Q, K, V, output, attention_weights = test_weights_sum_to_one()
    
    # Imprime as matrizes de input
    print("\n" + "="*60)
    print("MATRIZES DE INPUT")
    print("="*60)
    print(f"\nQ (queries) - shape {Q.shape}:")
    print(Q)
    print(f"\nK (keys) - shape {K.shape}:")
    print(K)
    print(f"\nV (values) - shape {V.shape}:")
    print(V)
    
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"\nAttention Weights - shape {attention_weights.shape}:")
    print(attention_weights)
    print(f"\nOutput - shape {output.shape}:")
    print(output)
    
    # Executa os outros testes
    test2_passed = test_output_shape()
    test3_passed = test_numerical_correctness()
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    print(f"Teste 1 (Weights somam 1.0): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Teste 2 (Shape do output): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Teste 3 (Correção numérica): {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL: {'✓ TODOS OS TESTES PASSARAM' if all_passed else '✗ ALGUNS TESTES FALHARAM'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
