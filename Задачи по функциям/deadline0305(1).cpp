#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

double lp_norm(const VectorXd& v, double p)
{
    if (p <= 0)
    {
        throw invalid_argument("Parameter p must be a positive number.");
    }
    
    double sum_abs_p = v.array().abs().pow(p).sum();

    return pow(sum_abs_p, 1.0 / p);
}

double total_variation(const MatrixXd& u, double p)
{
    if (p < 1)
    {
        throw invalid_argument("Parameter p must be greater than or equal to 1.");
    }

    MatrixXd dx = u.rightCols(u.cols() - 1) - u.leftCols(u.cols() - 1);
    dx = dx.array().abs();

    MatrixXd dy = u.bottomRows(u.rows() - 1) - u.topRows(u.rows() - 1);
    dy = dy.array().abs();

    int min_rows = min(dx.rows(), dy.rows());
    int min_cols = min(dx.cols(), dy.cols());

    MatrixXd dx_trimmed = dx.topLeftCorner(min_rows, min_cols);
    MatrixXd dy_trimmed = dy.topLeftCorner(min_rows, min_cols);

    double tv_sum = dx_trimmed.array().pow(p).sum() + dy_trimmed.array().pow(p).sum();

    return pow(tv_sum, 1.0 / p);
}

int main()
{
    try
    {
        VectorXd vector(4);
        vector << 3.0, -1.0, 2.0, 1.0;

        MatrixXd image = MatrixXd::Random(10, 7);

        double l1_norm_val = lp_norm(vector, 1);
        cout << "L1-norm: " << l1_norm_val << endl;

        double l2_norm_val = lp_norm(vector, 2);
        cout << "L2-norm: " << l2_norm_val << endl;

        double tv_l1 = total_variation(image, 1);
        cout << "Total variation with L1-norm: " << tv_l1 << endl;

        double tv_l2 = total_variation(image, 2);
        cout << "Total variation with L2-norm: " << tv_l2 << endl;
    }

    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
    }

    char c; cin >> c;
    return 0;
}