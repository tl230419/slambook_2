#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
	MatrixXf A = MatrixXf::Random(3, 2);
	cout << "Here is the matrix A:\n" << A << endl;
	VectorXf b = VectorXf::Random(3);
	cout << "Here is the right hand side b:\n" << b << endl;

	// SVD ??? Why not have no bdcSvd
	//cout << "The least-squares solution is:\n"
	//    << A.bdcSvd(ComputeThinU | ComputeThinV).solve(b) << endl;

	// QR
	cout << "The solution using the QR decomposition is:\n"
	     << A.colPivHouseholderQr().solve(b) << endl;
	
	// Normal Equations
	cout << "The solution using normal equations is:\n"
	     << (A.transpose() * A).ldlt().solve(A.transpose() * b) << endl;

	return 0;
}
