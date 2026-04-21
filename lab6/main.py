import numpy as np

def generate_data(n=100, x_val=2.5):
    # Генерація матриці A
    A = np.random.uniform(-100, 100, (n, n))
    x_true = np.full(n, x_val)
    # вектор вільних членів
    b = A @ x_true
    
    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)
    print(f"Дані згенеровані: матриця {n}x{n} та вектор B збережені у файли.")
    return x_true

def load_data():
    A = np.loadtxt('matrix_A.txt')
    b = np.loadtxt('vector_B.txt')
    return A, b

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    return L, U

def solve_lu(L, U, b):
    n = len(L)
    # Прямий хід
    z = np.zeros(n)
    for i in range(n):
        z[i] = b[i] - np.dot(L[i, :i], z[:i])
    # Зворотний хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def iterative_refinement(A, b, x0, eps0=1e-14):
    x = x0.copy()
    L, U = lu_decomposition(A)
    iterations = 0
    max_iters = 50 
    
    while iterations < max_iters:
        # нев'язка
        r = b - (A @ x)
        error = np.max(np.abs(r))
        
        if error < eps0:
            break
            
        # Розв'язання системи для уточнення
        d = solve_lu(L, U, r)
        x = x + d
        iterations += 1
        
    return x, iterations, error


#дані
x_target = generate_data(100, 2.5)
A, b = load_data()

# LU-розклад 
L, U = lu_decomposition(A)
np.savetxt('matrix_L.txt', L)
np.savetxt('matrix_U.txt', U)

# Початковий розв'язок
x_initial = solve_lu(L, U, b)

#Уточнення розв'язку
x_refined, iters, final_res = iterative_refinement(A, b, x_initial)

# вивід результатів
print("\n" + "="*75)
print(f"{'№':<4} | {'Початкове (x_i)':<12} | {'Уточнений результат':<22} | {'Різниця':<10}")
print("-" * 75)

for i in range(len(x_refined)):
    diff = abs(x_refined[i] - x_target[i])
    print(f"{i:<4} | {x_target[i]:<12.1f} | {x_refined[i]:<22.18f} | {diff:.2e}")

print("="*75)
print(f"Статистика:")
print(f"- Кількість ітерацій уточнення: {iters}")
print(f"- Фінальна норма нев'язки (eps): {final_res:.2e}")
print("="*75)