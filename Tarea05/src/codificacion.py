def codifica_dec(n, nBits):
    binArr = [0] * nBits

    for i in range(nBits-1, -1, -1):
        binArr[i] = n%2
        n //= 2

    return binArr

def decodifica_dec(bits, nBits):
    if len(bits) > nBits:
        raise ValueError(f"Imposible convertir {bits} con {nBits}")
    bits.reverse()              # Se asume orden correcto de los bits
    n = 0
    base = 1
    for i in range(0, nBits-1):
        if i >= len(bits):
            break
        n += base * bits[i]
        base = base * 2

    return n

def codifica(x, n_bits, a, b):
    max_val = (1 << n_bits) - 1  # 2^n_bits - 1
    k = round((x - a) * max_val / (b - a))
    return codifica_dec(k, n_bits)


def decodifica(x_cod, n_bits, a, b):
    k = 0
    for i in range(n_bits):
        k = (k << 1) | x_cod[i]

    max_val = (1 << n_bits) - 1
    return a + k * (b - a) / max_val


def codifica_array(x, dim_x, n_bits, a, b):
    res = [0] * (dim_x * n_bits)
    for i in range(dim_x):
        bits = codifica(x[i], n_bits, a, b)
        for j in range(n_bits):
            res[i * n_bits + j] = bits[j]

    return res


def decodifica_array(x_cod, dim_x, n_bits, a, b):
    res = [0.0] * dim_x
    for i in range(dim_x):
        bits = [0] * n_bits
        for j in range(n_bits):
            bits[j] = x_cod[i * n_bits + j]
        res[i] = decodifica(bits, n_bits, a, b)

    return res


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo con un solo valor
    valor = 3.5
    n_bits = 8
    a = 0.0
    b = 10.0
    i = 135
    
    print(f"Valor original: {valor}")
    print(f"Valor original decimal: {i}")

    binario = codifica_dec(i, 12)
    print(f"Valor binario: {binario}" )
    print(f"Volver: {decodifica_dec(binario, 12)}")

    # Codificar
    bits_codificados = codifica(valor, n_bits, a, b)
    print(f"Bits codificados: {bits_codificados}")
    
    # Decodificar
    valor_decodificado = decodifica(bits_codificados, n_bits, a, b)
    print(f"Valor decodificado: {valor_decodificado}")
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo con array
    valores = [1.2, 5.7, 8.3]
    dim_x = len(valores)
    
    print(f"Valores originales: {valores}")
    
    # Codificar array
    bits_array = codifica_array(valores, dim_x, n_bits, a, b)
    print(f"Bits del array: {bits_array}")
    
    # Decodificar array
    valores_decodificados = decodifica_array(bits_array, dim_x, n_bits, a, b)
    print(f"Valores decodificados: {valores_decodificados}")
