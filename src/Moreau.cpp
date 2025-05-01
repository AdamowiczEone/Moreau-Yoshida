#include <Eigen>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <limits>
#include <functional>
#include <matplot/matplot.h>

using namespace Eigen;
using namespace std;
using namespace std;
using namespace matplot;

// Простейшая реализация аналогов
double f_abs(double x) {
    return abs(x);
}

double f_huber(double x, double delta = 1.0) {
    return (abs(x) <= delta) ? 0.5 * x * x : delta * (abs(x) - 0.5 * delta);
}

double f_logistic(double x) {
    return log(1 + exp(-x));
}

double f_square_threshold(double x, double threshold = 0.5) {
    return (abs(x) < threshold) ? x * x : 2 * threshold * abs(x) - threshold * threshold;
}

double f_ridge(double x, double epsilon = 0.1) {
    return (x * x) / (abs(x) + epsilon);
}

// Проксимальный оператор по сетке
double proximal_operator(double (*func)(double), double v, double lam = 0.5,
                         double x_first = -10, double x_last = 10, int x_count = 500) {
    double min_val = 1e12;
    double best_x = x_first;

    for (int i = 0; i < x_count; ++i) {
        double x = x_first + i * (x_last - x_first) / (x_count - 1);
        double val = func(x) + (1 / (2 * lam)) * pow(x - v, 2);
        if (val < min_val) {
            min_val = val;
            best_x = x;
        }
    }
    return best_x;
}

// Moreau-Yoshida регуляризация
double moreau_yoshida(double (*func)(double), double lam, const vector<double> &k_vals,
                      double x, bool return_prox, double *prox_out = nullptr) {
    double min_val = 1e12;
    double best_k = k_vals[0];

    for (auto k : k_vals) {
        double val = func(k) + (1 / (2 * lam)) * pow(x - k, 2);
        if (val < min_val) {
            min_val = val;
            best_k = k;
        }
    }

    if (return_prox && prox_out) {
        *prox_out = best_k;
    }

    return min_val;
}

double g(double x) {
    if (x < 0) {
        return x + 1; // для x < 0
    } else if (x >= 0 && x <= 2) {
        return pow(x, 2); // для 0 <= x <= 2
    } else {
        return sqrt(x); // для x > 2
    }
}


vector<double> get_funct_values(double (*func)(double), const vector<double> &x_vals) {
    vector<double> funct_vals;
    for (double x : x_vals) {
        funct_vals.push_back(func(x));
    }
    return funct_vals;
}

int main() {
    setenv("GNUPLOT_DEFAULT_TERMINAL", "pngcairo", 1);
    // Диапазон значений

    std::vector<double> x_vals = linspace(-5, 10, 250);

    auto funct_values = get_funct_values(g, x_vals);
    // Пример проксимального оператора
    vector<double> prox_vals;
    for (double x : x_vals) {
        prox_vals.push_back(proximal_operator(g, x, 0.4));
    }

    vector<double> moreau_vals;
    for (double x_plot : x_vals) {
        moreau_vals.push_back(moreau_yoshida(g, 0.5, x_vals, x_plot, false));
    }

    // // === График проксимального оператора ===
    // figure();
    // plot(x_vals, prox_vals);
    // title("Proximal operator of |x|");
    // xlabel("x");
    // ylabel("prox_{λ|·|}(x)");
    // grid(true);
    // show();

    // === График регуляризации Моро-Йошиды ===

    plot(x_vals, moreau_vals);
    hold(on);
    plot(x_vals, funct_values);
    title("Moreau-Yoshida regularization of g");
    xlabel("x");
    ylabel("φ_λ(x)");
    grid(true);
    save("../output/moreau.png");
    show();

    return 0;
}