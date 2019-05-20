#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

Eigen::Matrix4d GetT(const Eigen::Quaterniond &q, const Eigen::Vector3d &t)
{
	Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
	Eigen::Matrix3d R = q.toRotationMatrix();
	T.block<3, 3>(0, 0) = R;
	T.block<3, 1>(0, 3) = t;

	return T;
}

Eigen::Matrix4d GetTInv(const Eigen::Quaterniond &q, const Eigen::Vector3d &t)
{
	Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
	Eigen::Matrix3d R = q.toRotationMatrix();
	T.block<3, 3>(0, 0) = R.transpose();
	T.block<3, 1>(0, 3) = -R.transpose() * t;

	return T;
}

int main(int argc, char** argv)
{
	// Point in robot1
	Eigen::Vector4d p1(0.5, 0, 0.2, 1);

	// Robot 1
	Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1);
	q1.normalize();
	Eigen::Vector3d t1(0.3, 0.1, 0.1);
	
	// Robot 2
	Eigen::Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
	q2.normalize();
	Eigen::Vector3d t2(-0.1, 0.5, 0.3);

	// Get T1 T2
	Eigen::Matrix4d T1_inv = GetTInv(q1, t1);
	Eigen::Matrix4d T2 = GetT(q2, t2);

	std::cout << T2 * T1_inv * p1 << std::endl;

	return 0;
}
