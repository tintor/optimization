#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>

using namespace std;
using namespace boost::math::differentiation;

constexpr double PI = M_PI;

template<typename X>
auto sqr(const X& x) { return x * x; }

#include <random>

double clamp(double x, double min_x, double max_x) {
    if (x > max_x) return max_x;
    if (x < min_x) return min_x;
    return x;
}

// TODO adam, vector as input, benchmarks

template<typename Func>
void optimize(const char* name, double range, bool newton, const Func& func) {
    cout << name << endl;
    std::mt19937 rng(0);

    double best = 1e100;
    std::uniform_real_distribution<double> unif(-range, range);
    double best_wx = unif(rng), best_wy = unif(rng);
    double dev = range / 3;
    size_t evals = 100 * 1000 * 1000;
    const double alpha = newton ? 1 : 1e-2;
    size_t misses = 0;

    for (size_t j = 0; evals > 0; j++) {
        double wx = clamp(std::normal_distribution<>(best_wx, dev)(rng), -range, range);
        double wy = clamp(std::normal_distribution<>(best_wy, dev)(rng), -range, range);
        bool improved = false;
        for (size_t i = 0; i < 10000; i++) {
            double result, dx, dy, d2x = 1, d2y = 1;
            if (newton) {
                const auto in = make_ftuple<double, 2, 2>(wx, wy);
                const auto& f = func(std::get<0>(in), std::get<1>(in));
                result = f.derivative(0, 0);
                dx = f.derivative(1, 0);
                dy = f.derivative(0, 1);
                d2x = f.derivative(2, 0);
                d2y = f.derivative(0, 2);
            } else {
                const auto in = make_ftuple<double, 1, 1>(wx, wy);
                const auto& f = func(std::get<0>(in), std::get<1>(in));
                result = f.derivative(0, 0);
                dx = f.derivative(1, 0);
                dy = f.derivative(0, 1);
            }
            if (result < best) {
                best = result;
                best_wx = wx;
                best_wy = wy;
                improved = true;
            }
            if (isnan(dx) || isnan(dy) || isnan(d2x) || isnan(d2y)) {
                cout << "nan gradients at " << wx << " " << wy << endl;
                return;
            }
            wx -= alpha * dx / d2x;
            wy -= alpha * dy / d2y;
            if (abs(dx) < 1e-15 && abs(dy) < 1e-15) break;
            if (--evals == 0) break;
            if (wx < -range || wx > range) break;
            if (wy < -range || wy > range) break;
        }
        if (improved) {
            misses = 0;
            dev *= 0.95;
            cout << setprecision(15) << "x=" << best_wx << " y=" << best_wy << " f=" << best << " dev=" << dev << endl;
        } else {
            misses += 1;
            if (misses == 1000)
                break;
        }
    }
    cout << endl;
}

#define Optimize(A, B, C) \
    optimize(A, B, false, [](const auto& x, const auto& y){C;}); \
    optimize(A, B, true, [](const auto& x, const auto& y){C;})

int main() {
    Optimize("sphere", 10, return sqr(x) + sqr(y));
    Optimize("booth", 10, return sqr(x + 2*y - 7) + sqr(2*x + y - 5));
    Optimize("rastrigin", 5.12, return 20 + (x*x - 10*cos(2*PI*x)) + (y*y - 10*cos(2*PI*y)));
    Optimize("rosenbrock", 10, return 100*sqr(y - x*x) + sqr(1 - x));
    Optimize("easom", 100, return -cos(x)*cos(y)*exp(-sqr(x - PI) - sqr(y - PI)));
    Optimize("himmelblau", 5, return sqr(x*x + y - 11) + sqr(x + y*y - 7));
    Optimize("styblinski–tang", 5, return x*x*x*x - 16*x*x + 5*x + y*y*y*y - 16*y*y + 5*y);
    Optimize("schaffer n.2", 100, return 0.5 + (sqr(sin(x*x - y*y)) - 0.5) / sqr(1 + 0.001*(x*x + y*y)));
    Optimize("hölder table", 10, return -abs(sin(x)*cos(y)*exp(abs(1 - sqrt(x*x + y*y)/PI))));
}
