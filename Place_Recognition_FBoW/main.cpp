/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */
#include <fstream>
#include <iostream>
#include <vector>

// DBoW3
#include "vocabulary_creator.h"
// OpenCV
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"							// OPENCV LIB
#include "opencv2/xfeatures2d.hpp"				// FEATURES2D LIB (Non-Free Func.)
#include "opencv2/features2d.hpp"					// FEATURED2D LIB (Free Func.)
#include "opencv2/highgui.hpp"
// Visual C++
#include <numeric>
#include <iostream>
#include <cmath>
// Name Space
using namespace cv;
using namespace std;
using namespace fbow;
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
	// DBoW_Validation(features);
	return 0;
}

/// --------- OPENCV 4.2
// FEATURE MATCHING_NAVERLABS
vector<Mat> ExtractFeature()
{
	// Img Reading
	char filename[200];

	vector<String> filelist;
	glob("/home/iismn/Pictures/temp/*.jpg", filelist, false);
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
		String filename = filelist[i];
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

/// --------- FBoW
// FBoWCreation
void DBoW_Creation(const vector<Mat> &features)
{
    // Save DBoW DB - FBoW
		string desc_name;
		desc_name = "SIFT";
		VocabularyCreator::Params params;
		params.k = 10;
		params.L = 9;
		params.nthreads=1;
		params.maxIters=0;
		srand(0);
		VocabularyCreator voc_creator;
		Vocabulary voc;
		cout << "Creating a " << params.k << "^" << params.L << " vocabulary..." << endl;
		auto t_start=chrono::high_resolution_clock::now();
		voc_creator.create(voc,features,desc_name, params);
		auto t_end=chrono::high_resolution_clock::now();
		cout<<"time="<<double(chrono::duration_cast<chrono::milliseconds>(t_end-t_start).count())<<" msecs"<<endl;
		cout<<"nblocks="<<voc.size()<<endl;
		voc.saveToFile("NVR_voc.yml.gz");


}
// DBoW Validation
// void DBoW_Validation(const vector<Mat> &features)
// {
//     cout << "LOAD DBoW3 DB" << endl;
//     Vocabulary voc("small_voc.yml.gz");
//     Database db(voc, false, 0);
//
//     // Add Img from DB
//     for(size_t i = 0; i < features.size(); i++)
//         db.add(features[i]);
//
//     QueryResults ret;
//     db.query(features[0], ret, 4);
//     // Best Matching Image Out
//     cout << "Searching for Original Image" << ". " << ret << endl;
//     cout << "Best Match Index : " << ret[1].Id << " / Score : " << ret[1].Score << endl;
//
// 		////// Img MATCHER
// 		Mat src1;
// 		Mat src2;
// 		char Original[200];
// 		char Target[200];
// 		// FLANN BASED MATCHER
// 		vector<DMatch> good_matches;
// 		Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
// 		sprintf(Original, "DB/%06d.jpg", 1);
// 		src1 = imread(Original, IMREAD_GRAYSCALE);
// 		resize(src1,src1,Size(), 0.5, 0.5);
// 		sprintf(Target, "DB/%06d.jpg", ret[1].Id);
//
// 		src2 = imread(Target, IMREAD_GRAYSCALE);
// 		resize(src2,src2,Size(), 0.5, 0.5);
// 		// SIFT Detector - Descriptor
// 		Mat desc1, desc2,	Rdesc1, Rdesc2;																					// Descriptor Announce
// 		vector<KeyPoint> keypoints1, keypoints2;
// 		detector->detectAndCompute(src1, Mat(), keypoints1, desc1);	// First Image Feature Points
// 		detector->detectAndCompute(src2, Mat(), keypoints2, desc2);	// Second Image Feature Points
// 		RootSIFT(desc1, Rdesc1);
// 		RootSIFT(desc2, Rdesc2);
// 		Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
// 		vector<vector<DMatch>> matches;
//
// 		matcher->knnMatch(Rdesc1, Rdesc2, matches, 2);
// 		// matcher->knnMatch(desc1, desc2, matches, 2);
// 		const float ratio_thresh = 0.5f;
//
// 		for (size_t i = 0; i < matches.size(); i++){
// 				if (matches[i][0].distance < ratio_thresh * matches[i][1].distance){
// 						good_matches.push_back(matches[i][0]);
//
// 						if (isnan(good_matches[i].distance)){
// 							good_matches[i].distance = 0;
// 						}
// 				}
// 		}
//
// 		// DRAW IMAGE - Match Points
// 		Mat dst;
// 		drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
// 			Scalar::all(-1), Scalar::all(-1), vector<char>(),
// 			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
// 		imshow("dst", dst);
//
// 		//
// 		vector<Point2f> matched_points1, matched_points2; // these are your points that match
//
// 		for (int i=0;i<good_matches.size();i++)
// 		{
// 		    // this is how the DMatch structure stores the matching information
// 		    int idx1=good_matches[i].trainIdx;
// 		    int idx2=good_matches[i].queryIdx;
//
// 		    //now use those match indices to get the keypoints, add to your two lists of points
// 		    matched_points1.push_back(keypoints1[idx2].pt);
// 		    matched_points2.push_back(keypoints2[idx1].pt);
// 		}
//
// 		cout << matched_points1 << endl << endl;
// 		cout << matched_points2 << endl;
//
// 		waitKey();
// 		destroyAllWindows();
// }

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
