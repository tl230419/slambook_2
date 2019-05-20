#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;

int main(int argc, char** argv)
{
	Eigen::Matrix<double, 10, 10> originMat = Eigen::Matrix<double, 10, 10>::Random(10, 10);
	cout << "origin matrix = " << endl << originMat << endl;

	Eigen::Matrix<double, 3, 3> Mat = originMat.block<3, 3>(0, 0);
	cout << "old 3*3 matrix = " << endl << Mat << endl;

	Eigen::Matrix<double, 3, 3> eyeMat = Eigen::Matrix<double, 3, 3>::Identity(3, 3);
	cout << "new 3*3 matrix = " << endl << eyeMat << endl;

	originMat.block<3, 3>(0, 0) = eyeMat;
	cout << "new matrix = " << endl << originMat << endl;

	return 0;
}
