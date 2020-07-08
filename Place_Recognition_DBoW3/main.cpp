/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"							// OPENCV LIB
#include "opencv2/xfeatures2d.hpp"				// FEATURES2D LIB (Non-Free Func.)
#include "opencv2/features2d.hpp"					// FEATURED2D LIB (Free Func.)
#include "opencv2/highgui.hpp"
#include "DescManip.h"
// Visual C++
#include <numeric>
#include <iostream>
#include <cmath>
// Name Space
using namespace cv;
using namespace std;
using namespace DBoW3;
// Function CALL
vector<Mat> ExtractFeature();
void DBoW_Creation(const vector<Mat> &features);
void DBoW_Validation(const vector<Mat > &features);
void RootSIFT(Mat &desc,  Mat &RootSIFTdesc);
// PARAMETER
const bool EXTENDED_SURF = true;					 // Extended SURF 128-D Vector

// MAIN
int main(void)
{
	vector<Mat> features=ExtractFeature();
	DBoW_Creation(features);
	DBoW_Validation(features);
	return 0;
}

/// --------- OPENCV 4.2
// FEATURE MATCHING_Test
vector<Mat> ExtractFeature()
{
	// Img Reading
	char filename[200];

	vector<String> filelist;
	glob("DB2/*.jpg", filelist, false);
	cout << filelist.size() << endl;
	// Parameter Setting
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
	vector<Mat>    features;

	// START
	cout << "Extracting Extended SIFT Features" << endl;

  for (int i=1; i<filelist.size(); i++){
		// Initialize
		vector<KeyPoint> keypoints;
		Mat descriptors;

		// Image Reading - Target
		String filename = filelist[i-1];
		cout << filename << " / " << i << endl;
		Mat src = imread(filename, IMREAD_GRAYSCALE);
		resize(src,src,Size(),1,1);

		// !--Exception--!
		if (src.empty()) {
			cerr << "Image load failed!" << endl;
		}
		//

		// ExSURF Detector - Descriptor
		Mat desc;																									// Descriptor Announce
		Mat RootSIFTdesc;
		detector->detectAndCompute(src, Mat(), keypoints, desc);	// Image Feature Points
		// RootSIFT(desc, RootSIFTdesc);
		features.push_back(desc);
	}
	return features;
}
// FEATURE MATCHING_NAVERLABS
// vector<Mat> ExtractFeature()
// {
// 	// Img Reading
// 	char filename[200];
//
// 	vector<String> filelist;
// 	glob("/home/iismn/Pictures/temp/*.jpg", filelist, false);
// 	cout << filelist.size() << endl;
// 	// Parameter Setting
// 	Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
// 	vector<Mat>    features;
//
// 	// START
// 	cout << "Extracting Extended SIFT Features" << endl;
//
//   for (int i=1; i<filelist.size(); i++){
// 		// Initialize
// 		vector<KeyPoint> keypoints;
// 		Mat descriptors;
//
// 		// Image Reading - Target
// 		String filename = filelist[i];
// 		cout << filename << " / " << i << endl;
// 		Mat src = imread(filename, IMREAD_GRAYSCALE);
// 		resize(src,src,Size(),1,1);
//
// 		// !--Exception--!
// 		if (src.empty()) {
// 			cerr << "Image load failed!" << endl;
// 		}
// 		//
//
// 		// ExSURF Detector - Descriptor
// 		Mat desc;																									// Descriptor Announce
// 		Mat RootSIFTdesc;
// 		detector->detectAndCompute(src, Mat(), keypoints, desc);	// Image Feature Points
// 		// RootSIFT(desc, RootSIFTdesc);
// 		features.push_back(desc);
// 	}
// 	return features;
// }

/// --------- DBOW3
// DBoW Creation
void DBoW_Creation(const vector<Mat> &features)
{
    // Branching Factor / Depth Levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);
		cout << "Create Vocabulary" << endl;
    voc.create(features);

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // cout << "Matching Images(SCORE 0~1): " << endl;
    // BowVector v1, v2;
		//
		// // Compare Input Image to Validation DB
    // voc.transform(features[0], v1);
    // for(size_t j = 0; j < features.size(); j++)
    // {
    //     voc.transform(features[j], v2);
		//
    //     double score = voc.score(v1, v2);
    //     cout << "Origin Image " << " - Target Image " << j << ": " << score << endl;
    // }


    // Save DBoW DB
    cout << endl << "Saving vocabulary" << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}
// DBoW Validation
void DBoW_Validation(const vector<Mat> &features)
{
		Mat src1;
		Mat src2;
		char Original[200];
		char Target[200];
		// FLANN BASED MATCHER
		vector<DMatch> good_matches;
		Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
		sprintf(Original, "DB2/000001.jpg");
		src1 = imread(Original, IMREAD_GRAYSCALE);
		// resize(src1,src1,Size(), 0.5, 0.5);
		sprintf(Target, "DB2/000002.jpg");

		src2 = imread(Target, IMREAD_GRAYSCALE);
		// resize(src2,src2,Size(), 0.5, 0.5);
		// SIFT Detector - Descriptor
		Mat desc1, desc2,	Rdesc1, Rdesc2;																					// Descriptor Announce
		vector<KeyPoint> keypoints1, keypoints2;
		detector->detectAndCompute(src1, Mat(), keypoints1, desc1);	// First Image Feature Points
		detector->detectAndCompute(src2, Mat(), keypoints2, desc2);	// Second Image Feature Points
		RootSIFT(desc1, Rdesc1);
		RootSIFT(desc2, Rdesc2);
		Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
		vector<vector<DMatch>> matches;

		matcher->knnMatch(Rdesc1, Rdesc2, matches, 2);
		// matcher->knnMatch(desc1, desc2, matches, 2);
		const float ratio_thresh = 0.7f;

		for (size_t i = 0; i < matches.size(); i++){
				if (matches[i][0].distance < ratio_thresh * matches[i][1].distance){
						good_matches.push_back(matches[i][0]);

						if (isnan(good_matches[i].distance)){
							good_matches[i].distance = 0;
						}
				}
		}

		// DRAW IMAGE - Match Points
		Mat dst;
		drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
			Scalar::all(-1), Scalar::all(-1), vector<char>(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow("dst", dst);

		//
		vector<Point2f> matched_points1, matched_points2; // these are your points that match

		for (int i=0;i<good_matches.size();i++)
		{
		    // this is how the DMatch structure stores the matching information
		    int idx1=good_matches[i].trainIdx;
		    int idx2=good_matches[i].queryIdx;

		    //now use those match indices to get the keypoints, add to your two lists of points
		    matched_points1.push_back(keypoints1[idx2].pt);
		    matched_points2.push_back(keypoints2[idx1].pt);
		}

		cout << matched_points1 << endl << endl;
		cout << matched_points2 << endl;

		waitKey();
		destroyAllWindows();
}

/// --------- RootSIFT Descriptor
void RootSIFT(Mat &desc, Mat &RootSIFTdesc)
{

		// // For each row
		// for (int i = 0; i < desc.rows; ++i) {
		//   // Perform L1 normalization
		//   normalize(desc.row(i), desc.row(i), 1.0, 0.0, cv::NORM_L1);
		// }
		// // Perform sqrt on the whole descriptor matrix
		// sqrt(desc, desc);
		// cout << desc.rows << endl;
		for(int i = 0; i < desc.rows; i ++){
	    // Conver to float type
	    Mat f;
	    desc.row(i).convertTo(f,CV_32FC1);

	    normalize(f,f,1,0,NORM_L1); // l1 normalize
	    sqrt(f,f); // sqrt-root  root-sift
	    RootSIFTdesc.push_back(f);
			// cout << i << endl;
		}

}
