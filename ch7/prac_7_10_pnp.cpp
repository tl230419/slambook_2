#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
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

void bundleAdjustment(
	const vector<Point3f>& points_3d,
	const vector<Point2f>& points_2d,
	const Mat& K,
	Mat& R, Mat& t, Mat& T
);

struct pnp_problem
{
	// 2d point, 3d point, camera intrinsics
	pnp_problem(double x, double y, double X, double Y, double Z, 
		double cx, double cy, double fx, double fy) : 
		_x(x), _y(y), _X(X), _Y(Y), _Z(Z), _cx(cx), _cy(cy), _fx(fx), _fy(fy)
	{
	}	

	template <typename T>
	bool operator() (
		const T* const pose, 
		T* residual) const
	{
		T pts_3d[3];
		pts_3d[0] = T(_X);
		pts_3d[1] = T(_Y);
		pts_3d[2] = T(_Z);

		T r[3];
		r[0] = pose[0];
		r[1] = pose[1];
		r[2] = pose[2];
	
		T R[3];
		ceres::AngleAxisRotatePoint(r, pts_3d, R);

		// t
		R[0] += pose[3];
		R[1] += pose[4];
		R[2] += pose[5];

		// pixel frame map to camera frame
		T projectX = _fx * R[0] / R[2] + _cx;
		T projectY = _fy * R[1] / R[2] + _cy;

		residual[0] = T(_x) - projectX;
		residual[1] = T(_y) - projectY;

		return true;
	}

	double _x, _y;
	double _X, _Y, _Z;
	double _cx, _cy;
	double _fx, _fy;
};

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		cout << "usage: prac_7_10_pnp img1 img2 depth1 depth2" << endl;
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
	Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	vector<Point3f> pts_3d;
	vector<Point2f> pts_2d;
	for (DMatch m: matches)
	{
		ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
		if (d == 0) // bad depth
			continue;
		float dd = d / 5000.0;
		Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
		pts_3d.push_back(Point3f(p1.x*dd, p1.y*dd, dd));
		pts_2d.push_back(keypoints_2[m.trainIdx].pt);
	}	

	cout << "3d-2d pairs: " << pts_3d.size() << endl;

	Mat r, t;
#if 1
	solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP); // 调用OpenCV的Pnp求解，可选择EPNP，DLS等方法
#else
	solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV的Pnp求解，可选择EPNP，DLS等方法
#endif
	Mat R;
	cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

	cout << "r=" << endl << r << endl;
	cout << "R=" << endl << R << endl;
	cout << "t=" << endl << t << endl;

	cout << "calling bundle adjustment" << endl;

	Mat T;
	bundleAdjustment(pts_3d, pts_2d, K, R, t, T);

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

void bundleAdjustment(
	const vector<Point3f>& points_3d,
	const vector<Point2f>& points_2d,
	const Mat& K,
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

	double fx = K.at<double>(0, 0);
	double fy = K.at<double>(1, 1);
	double cx = K.at<double>(0, 2);
	double cy = K.at<double>(1, 2);

	for (size_t i = 0; i < points_3d.size(); i++)
	{
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<pnp_problem, 2, 6>(new pnp_problem(
				points_2d[i].x, points_2d[i].y,
				points_3d[i].x, points_3d[i].y, points_3d[i].z,
				cx, cy,
				fx, fy
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
