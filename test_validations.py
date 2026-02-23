"""
Script para testar as validações de entrada adicionadas no Commit 5.
"""
import numpy as np
from attention import scaled_dot_product_attention, softmax


def test_validations():
    """
    Testa as validações de entrada.
    """
    print("\n" + "="*60)
    print("TESTES DE VALIDAÇÃO")
    print("="*60)
    
    # Teste 1: softmax com array 1D (deve falhar)
    print("\n[Teste 1] softmax com array 1D:")
    try:
        softmax(np.array([1, 2, 3]))
        print("✗ FAILED - Deveria ter levantado ValueError")
    except ValueError as e:
        print(f"✓ PASSED - Erro esperado: {e}")
    
    # Teste 2: Q com dimensões incompatíveis com K
    print("\n[Teste 2] Q e K com dimensões incompatíveis:")
    try:
        Q = np.array([[1, 2], [3, 4]])  # 2x2
        K = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        V = np.array([[1], [2]])
        scaled_dot_product_attention(Q, K, V)
        print("✗ FAILED - Deveria ter levantado ValueError")
    except ValueError as e:
        print(f"✓ PASSED - Erro esperado: {e}")
    
    # Teste 3: K e V com número de linhas diferentes
    print("\n[Teste 3] K e V com número de linhas incompatível:")
    try:
        Q = np.array([[1, 2], [3, 4]])
        K = np.array([[1, 2], [3, 4]])  # 2x2
        V = np.array([[1], [2], [3]])  # 3x1
        scaled_dot_product_attention(Q, K, V)
        print("✗ FAILED - Deveria ter levantado ValueError")
    except ValueError as e:
        print(f"✓ PASSED - Erro esperado: {e}")
    
    # Teste 4: Entrada válida (deve funcionar)
    print("\n[Teste 4] Entrada válida:")
    try:
        Q = np.array([[1.0, 2.0], [3.0, 4.0]])
        K = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        V = np.array([[1.0], [2.0], [3.0]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        print(f"✓ PASSED - Output shape: {output.shape}, Weights shape: {weights.shape}")
    except Exception as e:
        print(f"✗ FAILED - Erro inesperado: {e}")
    
    print("\n" + "="*60)
    print("TESTES DE VALIDAÇÃO CONCLUÍDOS")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_validations()
