import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. Отримання даних про висоту з API Open-Elevation
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

try:
    response = requests.get(url, timeout=15)
    data = response.json()
    results = data["results"]
except Exception as e:
    print(f"Помилка API: {e}. Використовуються тестові дані.")
    results = [{"latitude": 0, "longitude": 0, "elevation": 1200 + i*50} for i in range(21)]

# Функція для обчислення відстані між координатами (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Формування масивів відстані та висоти
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
distances = [0]
for i in range(1, len(results)):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)
distances = np.array(distances)

# 2. Математична модель: Кубічний сплайн для апроксимації профілю висоти
def solve_spline_coeffs(x, y):
    n = len(x) - 1
    h = np.diff(x)
    
    # СЛАР для коефіцієнтів c (другі похідні)
    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)
    
    # Природні умови (Natural Spline): c0 = 0, cn = 0
    A[0, 0] = 1
    A[n, n] = 1
    
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    
    # Метод прогонки (Tridiagonal Matrix Algorithm)
    def tridiagonal_solve(A, B):
        size = len(B)
        alpha = np.zeros(size)
        beta = np.zeros(size)
        x_res = np.zeros(size)
        
        # Пряма прогонка
        alpha[1] = -A[0, 1] / A[0, 0]
        beta[1] = B[0] / A[0, 0]
        for i in range(1, size - 1):
            m = A[i, i] + A[i, i-1] * alpha[i]
            alpha[i+1] = -A[i, i+1] / m
            beta[i+1] = (B[i] - A[i, i-1] * beta[i]) / m
            
        # Зворотна прогонка
        denom = (A[size-1, size-1] + A[size-1, size-2] * alpha[size-1])
        x_res[size-1] = (B[size-1] - A[size-1, size-2] * beta[size-1]) / denom
        for i in range(size-2, -1, -1):
            x_res[i] = alpha[i+1] * x_res[i+1] + beta[i+1]
        return x_res

    c = tridiagonal_solve(A, B)
    
    # Обчислення a, b, d
    a = y[:-1]
    d = np.diff(c) / (3 * h)
    b = (np.diff(y) / h) - (h * (2 * c[:-1] + c[1:]) / 3)
    
    return a, b, c[:-1], d

# 3. Аналіз маршруту та побудова графіків
def analyze_route(n_nodes):
    # Вибір підмножини вузлів (10, 15 або 20)
    idx = np.linspace(0, len(distances)-1, n_nodes, dtype=int)
    x_n, y_n = distances[idx], elevations[idx]
    
    a, b, c, d = solve_spline_coeffs(x_n, y_n)
    
    # сітка для графіка та градієнта
    x_fine = np.linspace(x_n[0], x_n[-1], 500)
    y_fine = []
    slopes = []

    for val in x_fine:
        # Пошук відповідного інтервалу i
        i = np.searchsorted(x_n, val) - 1
        i = max(0, min(i, len(a)-1))
        dx = val - x_n[i]
        y_fine.append(a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3)
        slopes.append((b[i] + 2*c[i]*dx + 3*d[i]*dx**2) * 100) # Градієнт у %

    return x_fine, y_fine, slopes, x_n, y_n

# Вивід характеристик маршруту
total_ascent = sum(max(elevations[i]-elevations[i-1], 0) for i in range(1, len(elevations)))
energy_kcal = (80 * 9.81 * total_ascent) / 4184

print(f"МАРШРУТ ЗАРОСЛЯК — ГОВЕРЛА:")
print(f"Довжина: {distances[-1]:.1f} м")
print(f"Набір висоти: {total_ascent:.1f} м")
print(f"Енергія (80кг): {energy_kcal:.1f} ккал")

# Побудова графіків
plt.figure(figsize=(12, 8))

for nodes in [10, 15, 20]:
    xf, yf, slp, xn, yn = analyze_route(nodes)
    
    plt.subplot(2, 1, 1)
    plt.plot(xf, yf, label=f'Сплайн ({nodes} вузлів)')
    plt.scatter(xn, yn, s=20)
    
    plt.subplot(2, 1, 2)
    plt.plot(xf, slp, label=f'Градієнт ({nodes} вузлів)')

# Оформлення графіків
plt.subplot(2, 1, 1)
plt.title("Профіль висоти (Кубічний сплайн)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Крутизна маршруту (Градієнт %)")
plt.ylabel("Ухил (%)")
plt.xlabel("Відстань (м)")
plt.axhline(15, color='red', linestyle='--', alpha=0.5, label='Поріг 15%')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()