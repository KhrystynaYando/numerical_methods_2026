import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)


x_plot = np.linspace(0, 24, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='$f(x)$') 
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show() 


# Точне значення інтеграла 
a, b = 0, 24
I0, _ = quad(f, a, b)
print("-" * 30)
print(f"2. Точне значення I0: {I0:.12f}")

# Функція для методу Сімпсона
def simpson(f, a, b, n):
    if n % 2 != 0: n += 1  
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    I = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return I

#  N_opt
target_eps = 1e-12
n_range = np.arange(10, 1001, 10)
eps_values = []
N_opt = 1000 

for n in n_range:
    current_eps = abs(simpson(f, a, b, n) - I0)
    eps_values.append(current_eps)
    if current_eps <= target_eps and N_opt == 1000:
        N_opt = n

print(f"4. Оптимальне N_opt (для eps=1e-12): {N_opt}")

# Графік залежності похибки
plt.figure(figsize=(8, 5))
plt.semilogy(n_range, eps_values, label='|I(N) - I0|')
plt.axhline(y=target_eps, color='r', linestyle='--', label='Задана точність 1e-12')
plt.title('Графік залежності похибки від N')
plt.xlabel('N (кількість розбиттів)')
plt.ylabel('Похибка (log scale)')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()

N0 = int((N_opt / 10) // 8 * 8) 
if N0 < 8: N0 = 8
eps0 = abs(simpson(f, a, b, N0) - I0)
print(f"5. Похибка eps0 при N0={N0}: {eps0:.2e}")

#  Метод Рунге-Ромберга
I_N0 = simpson(f, a, b, N0)
I_N0_2 = simpson(f, a, b, N0 // 2)
IR = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(IR - I0)
print(f"6. Метод Рунге-Ромберга IR: {IR:.12f}, Похибка epsR: {epsR:.2e}")

# Метод Ейткена та порядок точності p
I_N0_4 = simpson(f, a, b, N0 // 4)
# Розрахунок IE
denominator_aitken = (2 * I_N0_2 - (I_N0 + I_N0_4))
if denominator_aitken != 0:
    IE = (I_N0_2**2 - I_N0 * I_N0_4) / denominator_aitken
else:
    IE = I_N0

# p
p = (1 / np.log(2)) * np.log(abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))
epsE = abs(IE - I0)

print(f"7. Метод Ейткена IE: {IE:.12f}")
print(f"   Похибка epsE: {epsE:.2e}")
print(f"   Оцінений порядок точності p: {p:.4f}")

# Адаптивний алгоритм
def adaptive_simpson(f, a, b, eps, whole):
    c = (a + b) / 2
    left = simpson(f, a, c, 2)
    right = simpson(f, c, b, 2)
    if abs(left + right - whole) <= 15 * eps:
        return left + right + (left + right - whole) / 15
    return adaptive_simpson(f, a, c, eps/2, left) + \
           adaptive_simpson(f, c, b, eps/2, right)

I_adapt = adaptive_simpson(f, a, b, target_eps, simpson(f, a, b, 2))
print(f"9. Адаптивний алгоритм: {I_adapt:.12f}")
print(f"   Похибка адаптивного методу: {abs(I_adapt - I0):.2e}")
print("-" * 30)