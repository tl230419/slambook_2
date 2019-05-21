#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;

void find_feature_matches(
	const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1,
	std::vector<KeyPoint>& keypoints_2,
	std::vector<DMatch>& matches);

// 像素坐标归一化坐标
Point2d pixel2cam(const Point2d& p, const Mat& K);

void pose_estimation_3d3d(
	const vector<Point3f>& pts1,
	const vector<Point3f>& pts2,
	Mat& R, Mat& t
);

void bundleAdjustment(
	const vector<Point3f>& pts1,
	const vector<Point3f>& pts2,
	Mat& R, Mat& t, Mat& T
);

struct icp_problem
{
	icp_problem(double x1, double y1, double z1, double x2, double y2, double z2)
	: _x1(x1), _y1(y1), _z1(z1), _x2(x2), _y2(y2), _z2(z2)
	{
	}

	template <typename T>
	bool operator()(
		const T* const pose, T* residual) const
	{
		T pts_2[3];
		pts_2[0] = T(_x2);
		pts_2[1] = T(_y2);
		pts_2[2] = T(_z2);

		T r[3];
		r[0] = pose[0];
		r[1] = pose[1];
		r[2] = pose[2];

		T R[3];
		ceres::AngleAxisRotatePoint(r, pts_2, R);

		R[0] += pose[3];
		R[1] += pose[4];
		R[2] += pose[5];

		// 3d-3d
		residual[0] = T(_x1) - R[0];
		residual[1] = T(_x1) - R[1];
		residual[2] = T(_x1) - R[2];

		return true;
	}

	double _x1, _y1, _z1;
	double _x2, _y2, _z2;
};

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point) : _point(point) {}

	virtual void computeError()
	{
		const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
		// measurement is p, point is p'
		_error = _measurement - pose->estimate().map(_point);
	}

	virtual void linearizeOplus()
	{
		g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
		g2o::SE3Quat T(pose->estimate());
		Eigen::Vector3d xyz_trans = T.map(_point);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double z = xyz_trans[2];

		_jacobianOplusXi(0, 0) = 0;
		_jacobianOplusXi(0, 1) = -z;
		_jacobianOplusXi(0, 2) = y;
		_jacobianOplusXi(0, 3) = -1;
		_jacobianOplusXi(0, 4) = 0;
		_jacobianOplusXi(0, 5) = 0;

		_jacobianOplusXi(1, 0) = z;
		_jacobianOplusXi(1, 1) = 0;
		_jacobianOplusXi(1, 2) = -x;
		_jacobianOplusXi(1, 3) = 0;
		_jacobianOplusXi(1, 4) = -1;
		_jacobianOplusXi(1, 5) = 0;

		_jacobianOplusXi(2, 0) = -y;
		_jacobianOplusXi(2, 1) = x;
		_jacobianOplusXi(2, 2) = 0;
		_jacobianOplusXi(2, 3) = 0;
		_jacobianOplusXi(2, 4) = 0;
		_jacobianOplusXi(2, 5) = -1;
	}

	bool read(istream& in) {}
	bool write(ostream& out) const {}
protected:
	Eigen::Vector3d _point;
};

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
		return 1;
	}

	// -- 读取图像
	Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "一共找到了" << matches.size() << "组匹配点" << endl;

	// 建立3D点
	Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
	Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	vector<Point3f> pts1, pts2;

	for (DMatch m: matches)
	{
		ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
		ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
		if (d1 == 0 || d2 == 0) // bad depth
			continue;
		float dd1 = float(d1) / 5000.0;
		float dd2 = float(d2) / 5000.0;
		Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
		Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
		pts1.push_back(Point3f(p1.x*dd1, p1.y*dd1, dd1));
		pts2.push_back(Point3f(p2.x*dd2, p2.y*dd2, dd2));
	}	

	cout << "3d-2d pairs: " << pts1.size() << endl;

	Mat R, t;
	pose_estimation_3d3d(pts1, pts2, R, t);	
	cout << "ICP via SVD results: " << endl;
	cout << "R=" << R << endl;
	cout << "t=" << t << endl;
	cout << "R_inv=" << R.t() << endl;
	cout << "t_inv=" << -R.t() * t << endl;

	cout << "calling bundle adjustment" << endl;
	Mat T;
	bundleAdjustment(pts1, pts2, R, t, T);

	return 0;
}

void find_feature_matches(
	const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1,
	std::vector<KeyPoint>& keypoints_2,
	std::vector<DMatch>& matches)
{
	// -- 初始化
	Mat descriptors_1, descriptors_2;
	// used in OpenCV3
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	// use this if you are in OpenCV2
	// Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("ORB");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	// -- 第一步：检测Oriented FAST 角点位置
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	// -- 第二步：根据角点位置计算BRIEF描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	// -- 第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
	vector<DMatch> match;
	// BFMatcher matcher(NORM_HAMMING);
	matcher->match(descriptors_1, descriptors_2, match);

	// -- 第四步：匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	// 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	// 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离会非常小，设置一个经验值30作为下限。
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (match[i].distance <= max(2 * min_dist, 30.0))
		{
			matches.push_back(match[i]);
		}
	}
}

// 像素坐标归一化坐标
Point2d pixel2cam(const Point2d& p, const Mat& K)
{
	return Point2d
		(
		 	(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
			(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
		);
}

void pose_estimation_3d3d(
	const vector<Point3f>& pts1,
	const vector<Point3f>& pts2,
	Mat& R, Mat& t
)
{
	Point3f p1, p2;		// center of mass
	int N = pts1.size();
	for (int i = 0; i < N; i++)
	{
		p1 += pts1[i];
		p2 += pts2[i];
	}
	p1 = Point3f(Vec3f(p1) / N);
	p2 = Point3f(Vec3f(p2) / N);
	vector<Point3f> q1(N), q2(N); // remove the center
	for (int i = 0; i < N; i++)
	{
		q1[i] = pts1[i] - p1;
		q2[i] = pts2[i] - p2;
	}

	// compute q1*q2^T
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i < N; i++)
	{
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * 
			Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
	}
	cout << "W=" << W << endl;

	// SVD on W
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	cout << "U=" << U << endl;
	cout << "V=" << V << endl;

	Eigen::Matrix3d R_ = U * (V.transpose());
	Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

	// convert to cv::Mat
	R = (Mat_<double>(3, 3) << 
		R_(0, 0), R_(0, 1), R_(0, 2),
		R_(1, 0), R_(1, 1), R_(1, 2),
		R_(2, 0), R_(2, 1), R_(2, 2)
	     );
	t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(
	const vector<Point3f>& pts_1,
	const vector<Point3f>& pts_2,
	Mat& R, Mat& t, Mat& T
)
{
	ceres::Problem problem;

	Mat rotate_vector;
	Rodrigues(R, rotate_vector);

	double pose[6];
	pose[0] = rotate_vector.at<double>(0);
	pose[1] = rotate_vector.at<double>(1);
	pose[2] = rotate_vector.at<double>(2);
	pose[3] = t.at<double>(0);
	pose[4] = t.at<double>(1);
	pose[5] = t.at<double>(2);

	for (size_t i = 0; i < pts_1.size(); i++)
	{
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<icp_problem, 3, 6>(new icp_problem(
				pts_1[i].x, pts_1[i].y, pts_1[i].z,
				pts_2[i].x, pts_2[i].y, pts_2[i].z
			)),
			nullptr,
			pose
		);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t1 - t2);
	cout << "solve time cost = " << time_used.count() << " seconds." << endl;
	cout << summary.BriefReport() << endl;
	
	rotate_vector.at<double>(0) = pose[3];
	rotate_vector.at<double>(1) = pose[4];
	rotate_vector.at<double>(2) = pose[5];
	Rodrigues(rotate_vector, R);

	T = (Mat_<double>(4, 4) << 
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2),
		0,		    0,		       0,		   1
	);
	cout << "T = " << endl << T << endl;
}
