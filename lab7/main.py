import numpy as np
import os


def generate_and_save_data(n=100, x_val=2.5, file_a="matrix_A.txt", file_b="vector_B.txt"):
    # Випадкова матриця
    A = np.random.uniform(1, 10, (n, n))
    
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + 5.0  
    
    # Точний розв'язок x_i = 2.5
    x_true = np.full(n, x_val)
    
    b = A @ x_true
    
    np.savetxt(file_a, A)
    np.savetxt(file_b, b)
    print(f"Файли {file_a} та {file_b} успішно створено.")

def load_data(file_a, file_b):
    A = np.loadtxt(file_a)
    b = np.loadtxt(file_b)
    return A, b


def vector_norm(v):
    #Норма вектора
    return np.max(np.abs(v))

def simple_iteration(A, b, eps=1e-14, max_iter=20000):
    #Метод простої ітерації.
    n = len(b)
    tau = 1.0 / np.max(np.sum(np.abs(A), axis=1))
    x = np.ones(n) # Початкове наближення x0 = 1.0
    
    for k in range(1, max_iter + 1):
        x_new = x - tau * (A @ x - b)
        if vector_norm(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter

def jacobi_method(A, b, eps=1e-14, max_iter=20000):
   # Метод Якобі
    n = len(b)
    x = np.ones(n)
    D = np.diag(A)
    R = A - np.diag(D)
    
    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        if vector_norm(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter

def gauss_seidel_method(A, b, eps=1e-14, max_iter=20000):
    #Метод Гаусса-Зейделя
    n = len(b)
    x = np.ones(n)
    
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sum_j = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = sum_j / A[i, i]
        
        if vector_norm(x - x_old) < eps:
            return x, k
    return x, max_iter


if __name__ == "__main__":
    N = 100
    EPS = 1e-14
    FILE_A = "matrix_A.txt"
    FILE_B = "vector_B.txt"

    generate_and_save_data(n=N, file_a=FILE_A, file_b=FILE_B)
    A_mat, b_vec = load_data(FILE_A, FILE_B)

    results = []
    results.append(("Simple Iteration", simple_iteration(A_mat, b_vec, EPS)))
    results.append(("Jacobi Method", jacobi_method(A_mat, b_vec, EPS)))
    results.append(("Gauss-Seidel", gauss_seidel_method(A_mat, b_vec, EPS)))

    print("\n" + "="*95)
    print(f"{'Метод':<20} | {'Ітер.':<7} | {'Похибка':<10} | {'Перші 5 значень x'}")
    print("-" * 95)

    for name, (sol, iters) in results:
        # повний вектор у файл
        filename = f"solution_{name.replace(' ', '_').lower()}.txt"
        np.savetxt(filename, sol)
        
        # нев'язка
        residual = vector_norm(A_mat @ sol - b_vec)
        
        # рядок з першими 5 елементами
        sol_preview = ", ".join([f"{x:.6f}" for x in sol[:5]])
        
        # 4. Вивід у таблицю
        print(f"{name:<20} | {iters:<7} | {residual:.2e} | [{sol_preview}...]")
        print(f"{'':<20} | {'':<7} | {'':<10} |")
        print("-" * 95)