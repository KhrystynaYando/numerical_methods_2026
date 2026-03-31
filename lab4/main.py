import math
import matplotlib.pyplot as plt

# 1.Визначення функції та її точної похідної
def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def M_prime_exact(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

# Функція для чисельного диференціювання (центральна різниця)
def numerical_deriv(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

def main():
    x0 = 2.0
    exact_val = M_prime_exact(x0)
    
    print(f"Точне значення M'({x0}): {exact_val:.10f}\n")

    best_h = 1.0
    min_error = float('inf')
    
    for p in range(-20, 4):
        h_test = 10**p
        current_error = abs(numerical_deriv(x0, h_test) - exact_val)
        if current_error < min_error:
            min_error = current_error
            best_h = h_test
            
    print(f"Найкраща точність досягнута при h0 = {best_h:.1e}")
    print(f"Мінімальна похибка R0: {min_error:.2e}\n")

    h = 1e-3
    y_h = numerical_deriv(x0, h)
    y_2h = numerical_deriv(x0, 2 * h)
    R1 = abs(y_h - exact_val)
    
    print(f"Чисельна похідна y'(h):  {y_h:.10f}")
    print(f"Похибка R1:              {R1:.2e}\n")

    y_R = y_h + (y_h - y_2h) / 3
    R2 = abs(y_R - exact_val)
    
    print(f" (Рунге-Ромберг) ")
    print(f"Уточнене значення y'_R:  {y_R:.10f}")
    print(f"Похибка R2:              {R2:.2e}\n")


    y_4h = numerical_deriv(x0, 4 * h)
    
    numerator = (y_2h**2) - (y_4h * y_h)
    denominator = 2 * y_2h - (y_4h + y_h)
    y_E = numerator / denominator
    
    p_val = (1 / math.log(2)) * math.log(abs((y_4h - y_2h) / (y_2h - y_h)))
    R3 = abs(y_E - exact_val)
    
    print(f"(Ейткен)")
    print(f"Уточнене значення y'_E:  {y_E:.10f}")
    print(f"Порядок точності p:      {p_val:.4f}")
    print(f"Похибка R3:              {R3:.2e}")

    # Графік похибки
    h_values = []
    errors = []

    for p in range(-20, 4):
        h_val = 10**p
        err = abs(numerical_deriv(x0, h_val) - exact_val)
        h_values.append(h_val)
        errors.append(err)

    plt.figure()
    plt.loglog(h_values, errors, marker='o')  # <-- додали marker
    plt.xlabel("h")
    plt.ylabel("Похибка")
    plt.title("Графік похибки чисельного диференціювання")
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()