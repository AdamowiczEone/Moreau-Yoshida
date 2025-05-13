#include <iostream>
#include <Eigen>
using namespace std;

// https://eigen.tuxfamily.org/dox/namespaceEigen.html

// Matrix класс работает только с плотными (dense) матрицами
// Dense -матрицы хранят все элементы, включая нули
// Для разреженных (sparse) матриц нужен отдельный модуль Sparse
// Sparse-матрицы хранят только ненулевые элементы, экономя память.


/*
* Этот код решает систему линейных уравнений вида
A⋅x=b с использованием LDLT-разложения (разновидность разложения Холецкого для симметричных матриц)

LDLT-разложение работает только для симметричных матриц. Если A не симметрична, используйте другие методы:
 */
Eigen::Matrix2f LDLT(Eigen::Matrix2f A, Eigen::Matrix2f b){

std::cout << "Вот матрица A:\n" << A << std::endl;
std::cout << "Вот выходная матрица b:\n" << b << std::endl;
Eigen::Matrix2f x = A.ldlt().solve(b);
std::cout << "Решение:\n" << x << std::endl;
    return x;
}

void example_habr_1() {
    // Фиксированная матрица 2x2 (double)
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;

    // Фиксированный вектор из 2-х элементов
    Eigen::Vector2d vec(5, 6);

    // Матрично-векторное умножение
    Eigen::Vector2d result = mat * vec;
    std::cout << "Результат: " << result.transpose() << std::endl;
}


int main(){
  Eigen::Matrix2f A, B;
  A << 2, -1, -1, 3;
  B << 1, 2, 3, 1;
  auto x = LDLT(A, B);
    cout << "Сделаем проверку Ax = b"<< endl;
    cout << A*x << endl;

    Eigen::Matrix2f C = A + B;
    Eigen::Matrix2f result = (A * B + C).eval();
  return 0;
}