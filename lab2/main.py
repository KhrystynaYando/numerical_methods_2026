import numpy as np
import matplotlib.pyplot as plt
import csv


# 1. Зчитування даних з CSV-файлу

# Файл fps_data.csv містить:
# 1-й стовпець — кількість об'єктів
# 2-й стовпець — відповідне значення FPS

x = []  # масив кількості об'єктів
y = []  # масив значень FPS

with open("fps_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # пропускаємо заголовок таблиці
    for row in reader:
        x.append(float(row[0]))  # кількість об'єктів
        y.append(float(row[1]))  # FPS

# Перетворюємо списки у масиви NumPy для зручності обчислень
x = np.array(x)
y = np.array(y)


# 2. Таблиця розділених різниць (метод Ньютона)

# Створюємо функцію для обчислення таблиці розділених різниць.
# Вона використовується для побудови інтерполяційного многочлена Ньютона.

def divided_differences(x, y):
    n = len(y)  # кількість вузлів
    table = np.zeros((n, n))  # створюємо квадратну таблицю n×n

    # Перший стовпець таблиці — це значення функції
    table[:, 0] = y

    # Обчислення розділених різниць вищих порядків
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])

    return table


# Обчислюємо таблицю для наших даних
table = divided_differences(x, y)


# 3. Інтерполяція методом Ньютона

# Обчислення значення інтерполяційного многочлена Ньютона у довільній точці value.

def newton(x, table, value):
    n = len(x)
    result = table[0][0]  # початкове значення (y0)
    product = 1           # накопичувальний добуток (x - x0)(x - x1)...

    for i in range(1, n):
        product *= (value - x[i-1])
        result += table[0][i] * product

    return result


# 4. Метод Лагранжа
# Альтернативна форма інтерполяційного многочлена.
# Використовується для перевірки правильності результатів.

def lagrange(x_points, y_points, value):
    n = len(x_points)
    result = 0

    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (value - x_points[j]) / (x_points[i] - x_points[j])
        result += term

    return result



# 5. Прогноз FPS для 1000 об'єктів
# Обчислюємо значення FPS при 1000 об'єктах двома методами (Ньютон і Лагранж)

fps_1000_newton = newton(x, table, 1000)
fps_1000_lagrange = lagrange(x, y, 1000)

print("FPS для 1000 (Ньютон) =", round(fps_1000_newton, 2))
print("FPS для 1000 (Лагранж) =", round(fps_1000_lagrange, 2))


# 6. Пошук максимальної кількості об'єктів при FPS ≥ 60

# Перебираємо значення кількості об'єктів і визначаємо найбільше, при якому FPS не менше 60.

limit_objects = None

for n_value in range(100, 2000):
    if newton(x, table, n_value) >= 60:
        limit_objects = n_value

print("Максимальна кількість об'єктів при FPS ≥ 60 =", limit_objects)


# 7. Побудова графіка інтерполяції
# Будуємо графік:
# - початкових експериментальних точок
# - інтерполяційної кривої Ньютона

x_dense = np.linspace(min(x), max(x), 500)  # щільна сітка
y_dense = np.array([newton(x, table, val) for val in x_dense])

plt.figure()
plt.scatter(x, y)        # експериментальні точки
plt.plot(x_dense, y_dense)  # інтерполяційна крива
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.title("Інтерполяція методом Ньютона")
plt.grid(True)
plt.show()


# 8. Дослідження впливу кількості вузлів на похибку

nodes_list = [5, 10, 20]  # кількість вузлів
errors = []               # список максимальних похибок
error_distributions = []  # похибки по всій сітці для кожного набору вузлів

# Правильні значення для порівняння беремо з інтерполяції всіх експериментальних точок
true_vals = np.interp(x_dense, x, y)

plt.figure(figsize=(10,6))  # великий графік для наочності

for nodes in nodes_list:
    # Вибираємо вузли рівномірно по інтервалу
    x_temp = np.linspace(min(x), max(x), nodes)
    y_temp = np.interp(x_temp, x, y)  # значення FPS у вузлах

    # Створюємо таблицю розділених різниць для цих вузлів
    table_temp = divided_differences(x_temp, y_temp)

    # Інтерполяція на щільній сітці
    interp_vals = np.array([newton(x_temp, table_temp, val) for val in x_dense])

    # Абсолютна похибка на кожному значенні сітки
    abs_error = np.abs(true_vals - interp_vals)
    error_distributions.append(abs_error)

    # Максимальна похибка для цього набору вузлів
    max_error = np.max(abs_error)
    errors.append(max_error)

    # Додаємо криву похибки на графік
    plt.plot(x_dense, abs_error, label=f"{nodes} вузлів (max={max_error:.2f})")

# Налаштування графіка
plt.xlabel("Кількість об'єктів")
plt.ylabel("Абсолютна похибка")
plt.title("Похибка інтерполяції для різної кількості вузлів")
plt.grid(True)
plt.legend()
plt.show()

# Виводимо максимальні похибки для кожного набору вузлів
print("Максимальні похибки для вузлів [5, 10, 20]:", errors)


# 9. Демонстрація ефекту Рунге
# При великій кількості рівновіддалених вузлів многочлен високого степеня починає коливатись на краях інтервалу (ефект Рунге).

x_large = np.linspace(min(x), max(x), 20)
y_large = np.interp(x_large, x, y)

table_large = divided_differences(x_large, y_large)
y_interp_large = np.array([newton(x_large, table_large, val) for val in x_dense])

errors_runge = np.abs(true_vals - y_interp_large)

plt.figure()
plt.plot(x_dense, errors_runge)
plt.xlabel("Кількість об'єктів")
plt.ylabel("Похибка")
plt.title("Ефект Рунге (20 вузлів)")
plt.grid(True)
plt.show()


# 10. Розподіл похибки по всьому інтервалу
# Показуємо, як змінюється абсолютна похибка вздовж усього досліджуваного інтервалу.

y_interp = np.array([newton(x, table, val) for val in x_dense])
error_distribution = np.abs(true_vals - y_interp)

plt.figure()
plt.plot(x_dense, error_distribution)
plt.xlabel("Кількість об'єктів")
plt.ylabel("Абсолютна похибка")
plt.title("Розподіл похибки по інтервалу")
plt.grid(True)
plt.show()
