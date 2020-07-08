// NAVER LABS PLACE-RECOGNITION Competition 2020
// IRiS LAB. KAIST CEE.
// KAIST, W16, #510
// Daejeon, Yuseoung-gu, Rep. of KOREA

// Lee Sang Min
// iismn@kaist.ac.kr

// 헤더 호출
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"							// OPENCV LIB
#include "opencv2/xfeatures2d.hpp"				// FEATURES2D LIB (Non-Free Func.)
#include "opencv2/features2d.hpp"					// FEATURED2D LIB (Free Func.)
#include "opencv2/highgui.hpp"
#include <numeric>
#include <iostream>
// 호출자
using namespace cv;
using namespace std;

// 함수 선언
void keypoint_matching();
void find_homography();

// MAIN 함수
int main(void)
{
	keypoint_matching();
	// find_homography();
	return 0;
}

// FEATURE MATCHING
void keypoint_matching()
{
	// Img Reading
	char filename1[200];
  char filename2[200];
	sprintf(filename1, "DB/%06d.png", 1);
	// Image Reading - Origin
	Mat src1 = imread(filename1, IMREAD_GRAYSCALE);
	Mat src2 = imread(filename2, IMREAD_GRAYSCALE);

	// Parameter Setting
	Ptr<Feature2D> feature = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints1, keypoints_indx;
	vector<DMatch> good_matched_indx;
	float Distance_DESC = 0;
	float Distance_Desc_Check = 10000;
	int indx;
	Mat src_indx;

	// START
  for (int i=2; i<10; i++){
		// Initialize
		vector<KeyPoint> keypoints2;
		vector<DMatch> good_matches;
		// Image Reading - Target
	  sprintf(filename2, "DB/%06d.png", i);
		src2 = imread(filename2, IMREAD_GRAYSCALE);

		// !--Exception--!
		if (src1.empty() || src2.empty()) {
			cerr << "Image load failed!" << endl;
			return;
		}

		// SIFT Detector - Descriptor
		Mat desc1, desc2;																						// Descriptor Announce
		feature->detectAndCompute(src1, Mat(), keypoints1, desc1);	// First Image Feature Points
		feature->detectAndCompute(src2, Mat(), keypoints2, desc2);	// Second Image Feature Points

		// !--CALCULATION SPD CHECK--!
		double duration;
		duration = static_cast<double>(getTickCount());
		//

		// FLANN BASED MATCHER
		Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
		vector<vector<DMatch>> matches;
		matcher->knnMatch(desc1, desc2, matches, 2);
		Distance_DESC = 0;
		const float ratio_thresh = 0.5f;

		for (size_t i = 0; i < matches.size(); i++){
				if (matches[i][0].distance < ratio_thresh * matches[i][1].distance){
						good_matches.push_back(matches[i][0]);

						if (isnan(good_matches[i].distance)){
							good_matches[i].distance = 0;
						}
						// cout << good_matches[i].distance << endl;
						Distance_DESC = Distance_DESC + good_matches[i].distance;
				}
		}

		// !--CALCULATION SPD END--!
		duration = static_cast<double>(getTickCount())-duration;
		duration /= getTickFrequency();
		cout << "ETA 실행시간 : IMG" << i << " / "<< duration << endl;
		//
		cout << "DESCRIPTOR DISTANCE : " << Distance_DESC << endl << endl;

		if (Distance_DESC > Distance_Desc_Check){
			Distance_Desc_Check = Distance_DESC;
			indx = i;
			src_indx = src2;
			keypoints_indx = keypoints2;
			good_matched_indx = good_matches;
		}
	}
	cout << "BEST MATCH " << indx << endl << endl;
	// DRAW IMAGE
	Mat dst;
	drawMatches(src1, keypoints1, src_indx, keypoints_indx, good_matched_indx, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

// HOMOGRAPHY
void find_homography()
{
	Mat src1 = imread("box.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("box_in_scene.png", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	std::sort(matches.begin(), matches.end());
	vector<DMatch> good_matches(matches.begin(), matches.begin() + 50);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<Point2f> pts1, pts2;
	for (size_t i = 0; i < good_matches.size(); i++) {
		pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(pts1, pts2, RANSAC);

	vector<Point2f> corners1, corners2;
	corners1.push_back(Point2f(0, 0));
	corners1.push_back(Point2f(src1.cols - 1.f, 0));
	corners1.push_back(Point2f(src1.cols - 1.f, src1.rows - 1.f));
	corners1.push_back(Point2f(0, src1.rows - 1.f));
	perspectiveTransform(corners1, corners2, H);

	vector<Point> corners_dst;
	for (Point2f pt : corners2) {
		corners_dst.push_back(Point(cvRound(pt.x + src1.cols), cvRound(pt.y)));
	}

	polylines(dst, corners_dst, true, Scalar(0, 255, 0), 2, LINE_AA);

	imshow("dst", dst);
	cout << "Homography = " << endl << " "  << H << endl << endl;
	waitKey();
	destroyAllWindows();
}


//// BRUTE FOCRCE MATCHER
// Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
//
// vector<DMatch> matches;
// matcher->match(desc1, desc2, matches);
// std::sort(matches.begin(), matches.end());
// cout << "BF Matcher 호출 - 일반 방법 " << endl;
// vector<DMatch> matches;
// matcher->Match(desc1, desc2, matches, 2);
// std::sort(matches.begin(), matches.end());
// vector<DMatch> good_matches(matches.begin(), matches.begin() + 50);
// for (int i = 0; i < 51; i++){
// 	cout << "MATCHED POINT DISTACNE : " << matches[i].distance << endl;
// }
////
