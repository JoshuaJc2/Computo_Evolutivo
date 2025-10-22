import numpy as np

def sphere(x : np.array):
    '''
    Funcion Sphere
    '''
    if np.any(x < -5.12) or np.any(x>5.12):
        raise ValueError("Advertencia: algunos valores están fuera del rango [-5.12, 5.12]")
    return np.sum(x**2.0)

def ackley(X : np.array, a=20, b=0.2, c=2*np.pi):
    '''
    Funcion Ackley definida:
        f(x) = a + e - a*exp(-b*sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i)))
    '''
    if np.any(X < -30) or np.any(X > 30):
        raise ValueError("Advertencia: algunos valores están fuera del rango [-30, 30]")

    n = len(X)

    #Suma de cuadrados
    sum_squares = np.sum(X**2.0)

    # Suma de cosenos
    sum_cos = np.sum(np.cos(c * X))
    term_1 = -a*np.exp(-b * np.sqrt(sum_squares/n))
    term_2 = -np.exp(sum_cos/n)
    return a + np.e + term_1 + term_2

def griewank(X : np.array):
    if np.any(X < -600) or np.any(X > 600):
        raise ValueError("Advertencia: algunos valores están fuera del rango [-600, 600]")

    n = len(X)
    sum_term = np.sum(X**2.0) / 4000.0

    # Producto
    indices = np.arange(1, n+1)
    cos_terms = np.cos(X / np.sqrt(indices))
    prod_term = np.prod(cos_terms)

    # Definición correcta: 1 + sum(x_i^2/4000) - prod(cos(x_i/sqrt(i))) ent chance es  return 1 + sum_term - prod_term
    return 1 + sum_term + prod_term

def rastrigin(X : np.array):
    if np.any(X < -5.12) or np.any(X > 5.12):
        raise ValueError("Advertencia: algunos valores están fuera del rango [-5.12, 5.12]")

    n = len(X)
    sum_term = np.sum(X**2.0 - 10.0*np.cos(2.0 * np.pi * X))

    return 10*n + sum_term

def rosenbrock(X : np.array):
    if np.any(X < -2.048) or np.any(X > 2.048):
        raise ValueError("Advertencia: algunos valores están fuera del rango [-2.048, 2.048]")
    sum_term = np.sum(100.0 * (X[1:] - X[:-1]**2.0)**2.0 + (1 - X[:-1])**2.0)
    return sum_term
