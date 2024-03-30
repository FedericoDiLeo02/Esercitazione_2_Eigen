#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd A(2,2);
    Eigen::VectorXd b(2);
    Eigen::VectorXd x(2);
    Eigen::VectorXd true_x(2);
    true_x << -1.0, -1.0;

    // System 1 PALU
    A << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b << -5.169911863249772e-01, 1.672384680188350e-01;
    x = A.fullPivLu().solve(b);
    std::cout << "Solution System 1: " << x << std::endl;
    std::cout << "Relative error: " << (x - true_x).norm() / true_x.norm() << std::endl;

    // System 2 PALU
    A << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b << -6.394645785530173e-04, 4.259549612877223e-04;
    x = A.fullPivLu().solve(b);
    std::cout << "Solution System 2: " << x << std::endl;
    std::cout << "Relative error: " << (x - true_x).norm() / true_x.norm() << std::endl;

    // System 3 PALU
    A << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b << -6.400391328043042e-10, 4.266924591433963e-10;
    x = A.fullPivLu().solve(b);
    std::cout << "Solution System 3: " << x << std::endl;
    std::cout << "Relative error: " << (x - true_x).norm() / true_x.norm() << std::endl;

    // System 1 QR
    A << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b << -5.169911863249772e-01, 1.672384680188350e-01;
    x = A.householderQr().solve(b);
    std::cout << "Solution System 1: " << x << std::endl;
    std::cout << "Relative error: " << (x - true_x).norm() / true_x.norm() << std::endl;

    // System 2 QR
    A << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b << -6.394645785530173e-04, 4.259549612877223e-04;
    x = A.householderQr().solve(b);
    std::cout << "Solution System 2: " << x << std::endl;
    std::cout << "Relative error: " << (x - true_x).norm() / true_x.norm() << std::endl;

    // System 3 QR
    A << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b << -6.400391328043042e-10, 4.266924591433963e-10;
    x = A.householderQr().solve(b);
    std::cout << "Solution System 3: " << x << std::endl;
    std::cout << "Relative error: " << (x - true_x).norm() / true_x.norm() << std::endl;

    return 0;
}
