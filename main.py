import numpy as np
import matplotlib.pyplot as plt

# Параметры для нового варианта
T = np.pi  # период


def f(x):
    """Функция для нового варианта"""
    result = np.zeros_like(x)

    # Маски для разных интервалов
    mask1 = (x >= -np.pi / 2) & (x <= 0)  # cos(x) на [-π/2, 0]
    mask2 = (x > 0) & (x <= np.pi / 2)  # 1 - (2/π)*x на (0, π/2]

    result[mask1] = np.cos(x[mask1])
    result[mask2] = 1 - (2 / np.pi) * x[mask2]

    return result


def get_interpolation_nodes(n):
    """Генерация узлов интерполирования"""
    h = T / (2 * n)  # шаг
    k_values = np.arange(-n + 1, n + 1)
    x_k = k_values * h
    y_k = f(x_k)
    return x_k, y_k


def compute_fourier_coefficients(x_k, y_k, n):
    """Вычисление коэффициентов Фурье"""
    h = T / (2 * n)
    omega = 2 * np.pi / T

    # Инициализация коэффициентов
    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)

    # Вычисление a_l и b_l
    for l in range(n + 1):
        a_l = (h / T) * np.sum(y_k * np.cos(l * omega * x_k))
        b_l = (h / T) * np.sum(y_k * np.sin(l * omega * x_k))

        if l == 0:
            alpha[0] = a_l
        elif l < n:
            alpha[l] = 2 * a_l
            beta[l] = 2 * b_l
        else:  # l == n
            alpha[n] = a_l
            beta[n] = 0  # b_n = 0

    return alpha, beta


def trigonometric_polynomial(x, alpha, beta, n):
    """Тригонометрический интерполяционный многочлен"""
    omega = 2 * np.pi / T
    result = alpha[0] * np.ones_like(x)

    for l in range(1, n + 1):
        result += alpha[l] * np.cos(l * omega * x) + beta[l] * np.sin(l * omega * x)

    return result


def main():
    n_values = [4, 8, 16]

    # Создаем таблицы для разных n
    for n in n_values:
        print(f"\n{'=' * 50}")
        print(f"n = {n}")
        print(f"{'=' * 50}")

        # Узлы интерполирования
        x_k, y_k = get_interpolation_nodes(n)
        print(f"\nУзлы интерполирования:")
        print("k\tx_k\t\ty_k")
        print("-" * 30)
        for i, (x, y) in enumerate(zip(x_k, y_k)):
            print(f"{i - n + 1}\t{x:.3f}\t\t{y:.4f}")

        # Коэффициенты Фурье
        alpha, beta = compute_fourier_coefficients(x_k, y_k, n)
        print(f"\nКоэффициенты:")
        print(f"α₀ = {alpha[0]:.4f}")
        for l in range(1, n + 1):
            print(f"α_{l} = {alpha[l]:.4f}, β_{l} = {beta[l]:.4f}")

    # Построение графиков
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Точки для построения графиков
    x_plot = np.linspace(-np.pi / 2, np.pi / 2, 500)
    y_true = f(x_plot)

    for idx, n in enumerate(n_values):
        # Узлы и коэффициенты
        x_k, y_k = get_interpolation_nodes(n)
        alpha, beta = compute_fourier_coefficients(x_k, y_k, n)

        # Тригонометрический многочлен
        y_poly = trigonometric_polynomial(x_plot, alpha, beta, n)

        # Построение графика
        ax = axes[idx]
        ax.plot(x_plot, y_true, 'b-', linewidth=2, label='f(x)')
        ax.plot(x_plot, y_poly, 'r--', linewidth=2, label=f'φ(x), n={n}')
        ax.scatter(x_k, y_k, color='green', s=30, zorder=5, label='Узлы интерполяции')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Тригонометрический многочлен, n={n}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-np.pi / 2, np.pi / 2)

    plt.tight_layout()
    plt.show()

    # Таблица значений функции и многочленов
    print(f"\n{'=' * 100}")
    print("ТАБЛИЦА ЗНАЧЕНИЙ ФУНКЦИИ И ТРИГОНОМЕТРИЧЕСКИХ МНОГОЧЛЕНОВ")
    print(f"{'=' * 100}")

    x_table = np.linspace(-np.pi / 2, np.pi / 2, 41)  # 40 интервалов + 1 = 41 точка
    y_table = f(x_table)

    print(f"{'x':>8} {'f(x)':>10} ", end="")
    for n in n_values:
        print(f"{'φ(x), n=' + str(n):>15} ", end="")
    print()
    print("-" * 100)

    # Вычисляем значения многочленов для всех n
    poly_values = {}
    for n in n_values:
        x_k, y_k = get_interpolation_nodes(n)
        alpha, beta = compute_fourier_coefficients(x_k, y_k, n)
        poly_values[n] = trigonometric_polynomial(x_table, alpha, beta, n)

    # Выводим таблицу (ВСЕ точки)
    for i in range(len(x_table)):
        x = x_table[i]
        print(f"{x:>8.2f} {y_table[i]:>10.4f} ", end="")
        for n in n_values:
            print(f"{poly_values[n][i]:>15.4f} ", end="")
        print()


if __name__ == "__main__":
    main()