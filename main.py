import numpy as np
import matplotlib.pyplot as plt
import csv

# 1. Зчитування даних
def load_data(filename):
    x, y = [], []
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Temp'] and row['Month']:
                    x.append(float(row['Month']))
                    y.append(float(row['Temp'])) 
        return np.array(x), np.array(y)
    except FileNotFoundError:
        print(f"Помилка: Файл {filename} не знайдено.")
        return np.array([]), np.array([])

# 2. Математичні функції
def form_matrix(x, m):
    n = m + 1
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i][j] = np.sum(x ** (i + j))
    return A

def form_vector(x, y, m):
    n = m + 1
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.sum(y * (x ** i))
    return b

def gauss_solve(A, b):
    n = len(b)
    Ab = np.column_stack((A, b.astype(float)))
    for k in range(n):
        max_row = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, max_row]] = Ab[[max_row, k]]
        for i in range(k + 1, n):
            factor = Ab[i][k] / Ab[k][k]
            Ab[i] -= factor * Ab[k]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (Ab[i][-1] - np.dot(Ab[i][i + 1:n], x_sol[i + 1:n])) / Ab[i][i]
    return x_sol

def calculate_polynomial(x, coeffs):
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        y_poly += c * (x ** i)
    return y_poly

# ГОЛОВНА ЧАСТИНА
data_file = 'temp_data.csv'
x, y = load_data(data_file)

if len(x) > 0:
    max_degree = 4
    results = {}
    m_values = []
    d_values = []

    # Створюємо одне велике вікно для трьох графіків
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Обчислення
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vector = form_vector(x, y, m)
        coefs = gauss_solve(A, b_vector)
        y_approx = calculate_polynomial(x, coefs)

        dispersion = np.sum((y - y_approx) ** 2) / (len(y) - m - 1)
        
        results[m] = {'coefs': coefs, 'disp': dispersion, 'y_app': y_approx}
        m_values.append(m)
        d_values.append(dispersion)

        # Малюємо лінію апроксимації на першому графіку
        ax1.plot(x, y_approx, label=f'Степінь {m} (D={dispersion:.2f})')

    optimal_m = min(results, key=lambda m: results[m]['disp'])

    # ГРАФІК 1: Апроксимація 
    ax1.scatter(x, y, color='black', label='Фактичні дані', zorder=5)
    x_future = np.array([25, 26, 27])
    y_future = calculate_polynomial(x_future, results[optimal_m]['coefs'])
    ax1.scatter(x_future, y_future, color='red', marker='x', s=100, label='Прогноз')
    ax1.set_title('Апроксимація та екстраполяція')
    ax1.legend()
    ax1.grid(True)

    # ГРАФІК 2: Похибки для оптимального m 
    error = y - results[optimal_m]['y_app']
    ax2.bar(x, error, color='red', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title(f'Залишки (похибки) для оптимального m={optimal_m}')
    ax2.grid(True)

    #  ГРАФІК 3: Дисперсія ДЛЯ ВСІХ СТЕПЕНІВ 
    ax3.plot(m_values, d_values, marker='o', linestyle='-', color='blue', linewidth=2, label='Значення D')
    # Виділяємо кожну точку окремо, щоб було видно всі 4
    ax3.scatter(m_values, d_values, color='blue', s=80)
    # Виділяємо оптимальну точку червоним
    ax3.scatter(optimal_m, results[optimal_m]['disp'], color='red', s=150, zorder=5, label='Мінімум (Оптимально)')
    
    ax3.set_title('Залежність дисперсії від степеня полінома (m=1..4)')
    ax3.set_xlabel('Степінь (m)')
    ax3.set_ylabel('Дисперсія (D)')
    ax3.set_xticks(m_values) # Фіксуємо мітки 1, 2, 3, 4
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Вивід результатів у консоль
    print(f"\nОптимальний степінь: {optimal_m}")
    print(f"Прогноз на 25-27 місяці: {y_future}")