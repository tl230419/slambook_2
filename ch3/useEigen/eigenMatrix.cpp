#include <iostream>
#include <ctime>
using namespace std;

// Eigen部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆/特征值等）
#include <Eigen/Dense>

#define MATRIX_SIZE 50

/**********************************
 * 本程序演示了Eigen基本类型的使用
 *********************************/

int main(int argc, char** argv)
{
	// Eigen 以矩阵为基本数据单元。它是一个模板类。它的前三个参数为：数据类型，行，列
	// 声明一个2*3的float矩阵
	Eigen::Matrix<float, 2, 3> matrix_23;
	// 同时，Eigen通过typedef提供了许多内置类型，不过底层仍是Eigen::Matrix
	// 例如Vector3d实质上Eigen::Matrix<double, 3, 1>
	Eigen::Vector3d v_3d;
	// 还有Matrix3d实质上是Eigen::Matrix<double, 3, 3>
	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero(); // 初始化为零
	// 如果不确定矩阵大小，可以使用动态大小的矩阵
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dybnamic;
	// 更简单的
	Eigen::MatrixXd matrix_x;
	// 这种类型还有很多，就不一一列举
	
	// 下面是对矩阵的操作
	// 输入数据
	matrix_23 << 1, 2, 3, 4, 5, 6;
	// 输出
	cout << matrix_23 << endl;

	// 使用()访问矩阵中法的元素
	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			cout << matrix_23(i, j) << endl;
		}
	}

	v_3d << 3, 2, 1;
	// 矩阵和向量相乘（实际上还是矩阵和矩阵）
	// 但是在这里你不能混合两种不同类型的矩阵，像这样是错的
	//Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;

	// 应该是显式转换
	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << result << endl;


	// 同样你不能搞错矩阵的维度
	// 试着取消下面的注释，看看会报什么错
	//Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;
	
	// 一些矩阵运算
	// 四则运算就不演示了，直接用对应的运算符即可
	matrix_33 = Eigen::Matrix3d::Random();

#if 1
	Eigen::Matrix3d matrix_33_transpose = matrix_33;
	Eigen::Matrix3d matrix_33_inverse = matrix_33;
	cout << "matrix_33_transpose = \n" << endl;
	cout << matrix_33_transpose.transpose() << endl << endl;
	cout << "matrix_33_inverse = \n" << endl;
	cout << matrix_33_inverse.inverse() << endl << endl;
#endif
	cout << matrix_33 << endl << endl;

	cout << matrix_33.transpose() << endl;		// 转置
	cout << matrix_33.sum() << endl;		// 各元素和
	cout << matrix_33.trace() << endl;		// 迹
	cout << 10 * matrix_33 << endl;			// 数乘
	cout << matrix_33.inverse() << endl;		// 逆
	cout << matrix_33.determinant() << endl;	// 行列式

	// 特征值
	// 实对称矩阵可以保证对角化成功
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
	cout << "Eigen values = " << eigen_solver.eigenvalues() << endl;
	cout << "Eigen vectors = " << eigen_solver.eigenvectors() << endl;

	// 解方程
	// 我们求解matrix_NN * x = v_Nd这个方程
	// N的大小在前边的宏里定义，矩阵由随机数生成
	// 直接求逆自然是最直接的，但是求逆运算量大
	
	Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
	Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
	v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

	clock_t time_stt = clock(); // 计时
	// 直接求逆
	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
	cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

	// 通常用矩阵分解来求，例如QR分解，速度会快很多
	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << "time use in Qr composition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

	return 0;
}
