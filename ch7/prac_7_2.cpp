#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage: prac_7_1 img" << endl;
		return 1;
	}

	Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	std::vector<KeyPoint> keypoints_sift, keypoints_surf;
	Mat descriptors_sift, descriptors_surf;

	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	Ptr<SIFT> sift = SIFT::create(1000);
	sift->detectAndCompute(img, noArray(), keypoints_sift, descriptors_sift);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> sift_time_used = chrono::duration_cast<chrono::duration<double>>(t1 - t2);
	cout << "sift costs time: " << sift_time_used.count() << " seconds." << endl;
	Mat out_img_sift;
	drawKeypoints(img, keypoints_sift, out_img_sift, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("SIFT特征点", out_img_sift);

	chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
	Ptr<SURF> surf = SURF::create(1000);
	surf->detectAndCompute(img, noArray(), keypoints_surf, descriptors_surf);
	chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
	chrono::duration<double> surf_time_used = chrono::duration_cast<chrono::duration<double>>(t3 - t4);
	cout << "surf costs time: " << surf_time_used.count() << " seconds." << endl;
	Mat out_img_surf;
	drawKeypoints(img, keypoints_surf, out_img_surf, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("SURF特征点", out_img_surf);
	waitKey(0);

	return 0;
}
