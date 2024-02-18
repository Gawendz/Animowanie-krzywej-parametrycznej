import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Funkcja do obliczania hipotrochoidy
def hipotrochoida(r, R, h, t):
    x = (R - r) * np.cos(t) + h * np.cos((R - r) / r * t)
    y = (R - r) * np.sin(t) - h * np.sin((R - r) / r * t)
    return x, y

# Funkcja do obliczania epitrochoidy
def epitrochoida(r, R, h, t):
    x = (R + r) * np.cos(t) - h * np.cos((R + r) / r * t)
    y = (R + r) * np.sin(t) - h * np.sin((R + r) / r * t)
    return x, y

# Funkcja do aktualizacji ramek animacji hipotrochoidy i epitrochoidy
def update_hipo_epitro(frame, hipo, epitro, male_kolo, wektor, user_choice):
    r = 3
    R = 5
    h = 5
    time = np.linspace(0, 6 * np.pi, 300)

    x_curve, y_curve = [], []

    if frame > 0:
        if user_choice == '1':
            # Rysuj hipotrochoidę
            x_curve, y_curve = hipotrochoida(r, R, h, time[:frame])
            epitro.set_data([], [])  # Wyczyść dane epitrochoidy
            hipo.set_data(x_curve, y_curve)
            male_kolo.set_center(((R - r) * np.cos(time[frame]), (R - r) * np.sin(time[frame])))
        elif user_choice == '2':
            # Rysuj epitrochoidę
            x_curve, y_curve = epitrochoida(r, R, h, time[:frame])
            hipo.set_data([], [])  # Wyczyść dane hipotrochoidy
            male_kolo.set_center(((R + r) * np.cos(time[frame]), (R + r) * np.sin(time[frame])))
            epitro.set_data(x_curve, y_curve)

        wektor.set_data(
            [male_kolo.center[0], x_curve[-1] if len(x_curve) > 0 else male_kolo.center[0]],
            [male_kolo.center[1], y_curve[-1] if len(y_curve) > 0 else male_kolo.center[1]]
        )

    return hipo, epitro, male_kolo, wektor

# Funkcja do ewaluacji funkcji matematycznej
def eval_func(x, input_func):
    return input_func.subs(sp.symbols('x'), x)

# Funkcja do obliczania prostej prostopadłej
def calc_perpendicular_line(a, xs, ys):
    new_a = -1 / a
    new_b = ys - new_a * xs
    return new_a, new_b

# Funkcja do rozwiązania układu równań
def find_intersection_point(a_perp, b_perp, r, a_tan, b_tan):
    x, y = sp.symbols('x y')
    gen_func = a_tan * x - y + b_tan
    eq1 = sp.Eq(-gen_func / np.sqrt(float((a_tan * a_tan) + (-1 * -1))), r)
    eq2 = sp.Eq(a_perp * x - y, -b_perp)
    solution = sp.solve((eq1, eq2), (x, y))

    if solution:
        x_val = solution[x].evalf()
        y_val = solution[y].evalf()
        return float(x_val), float(y_val)
    else:
        return None

# Funkcja do obliczania okręgu w punkcie styczności
def calc_circle_at_tangent(x, r, func):
    f_prime = func.diff('x')
    f_value = func.subs('x', x)
    f_prime_value = f_prime.subs('x', x)
    tangent_eq = f_prime_value * (x - x) + f_value
    a_perp, b_perp = calc_perpendicular_line(f_prime_value, x, f_value)
    x_o, y_o = find_intersection_point(a_perp, b_perp, r, f_prime_value, (-f_prime_value * x + f_value))
    return x_o, y_o


# Funkcja do inicjalizacji wykresu
def initialization(user_choice):
    r = 3
    R = 5
    h = 5
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    duze_kolo = plt.Circle((0, 0), R, fill=False)
    male_kolo = plt.Circle((R - r, 0), r, fill=False)
    ax.add_patch(duze_kolo)
    ax.add_patch(male_kolo)

    hipo, = ax.plot([], [])
    epitro, = ax.plot([], [])
    wektor, = ax.plot([], [])

    time = np.linspace(0, 6 * np.pi, 300)

    ani = animation.FuncAnimation(fig, update_hipo_epitro, frames=len(time), init_func=None,
                                  fargs=(hipo, epitro, male_kolo, wektor, user_choice))

    plt.show()

user_choice = input("Opcje programu:\n1: hipocykloida\n2: epicykloida\n3: cykloida po krzywej (11*x + 12*x**2 + 5*x**3) / (7 + 4*x)\nTwój wybór:")

if user_choice == '1' or user_choice == '2':
    initialization(user_choice)
elif user_choice == '3':
    
    input_func_str = "(11*x + 12*x**2 + 5*x**3) / (7 + 4*x)"
    interval_start = -1
    interval_end = 9
    r = 0.5
    time = np.linspace(interval_start, interval_end, 300)
    input_func = sp.sympify(input_func_str)
    x_values = np.linspace(interval_start, interval_end, 300)
    y_values = np.vectorize(lambda x: eval_func(x, input_func))(x_values)
    xs_circle, ys_circle = calc_circle_at_tangent(x_values[0], r, input_func)

    plot, ax = plt.subplots(figsize=[10, 10])
    plt.title(f'Cykloida na {input_func}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_values, y_values, label=input_func_str)
    plt.grid(True)
    plt.axis('equal')

    circle_patch = plt.Circle((xs_circle, ys_circle), r, fill=False)
    ax.add_patch(circle_patch)

    curve, = ax.plot([], [])
    vector, = ax.plot([], [])

    def initialize():
        curve.set_data([], [])
        vector.set_data([], [])
        return curve, vector

    def distance(x1, x2, y1, y2):
        return np.sqrt(float((x1 - x2) ** 2 + (y1 - y2) ** 2))

    list_centers_x = []
    list_centers_y = []
    list_x = []
    list_y = []
    distances = []

    def update(frame):
        if frame == len(time) - 1:
        # Zeruj dane po dojechaniu do końca przedziału
            del list_centers_x[:]
            del list_centers_y[:]
            del list_x[:]
            del list_y[:]
            del distances[:]

        xc, yc = calc_circle_at_tangent(time[frame], r, input_func)
        circle_patch.center = (xc, yc)

        list_x.append(x_values[frame])
        list_y.append(y_values[frame])
        list_centers_x.append(xc)
        list_centers_y.append(yc)

        if len(list_x) > 1:
            distances.append(distance(list_x[-1], list_x[-2], list_y[-1], list_y[-2]))

        angle_sum = np.sum(distances)

        new_vector_x = xc + r * np.sin(angle_sum / r)
        new_vector_y = yc + r * np.cos(angle_sum / r)
        vector.set_data([new_vector_x, xc], [new_vector_y, yc])

        curve_x = list_centers_x + r * np.sin(np.cumsum([0] + distances) / r)
        curve_y = list_centers_y + r * np.cos(np.cumsum([0] + distances) / r)

        curve.set_data(curve_x, curve_y)

        return circle_patch, vector, curve


    animation_object = animation.FuncAnimation(plot, func=update, init_func=initialize, frames=len(time), interval=40)
    plt.show()
   
