#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

int main(int argc, char** argv)
{
	Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1);
	Eigen::Vector3d t2(0.3, 0.1, 0.1);
	Eigen::Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
	Eigen::Vector3d t(-0.1, 0.5, 0.3);
	Eigen::Vector3d p1_c(0.5, 0, 0.2);
	Eigen::Vector3d p_w;
	Eigen::Vector3d p2_c;

	p_w = q1.normalized().inverse() * (p1_c - t2);
	p2_c = q2.normalized() * p_w + t;
	cout << "p2_c = " << endl << p2_c << endl;

	return 0;
}
