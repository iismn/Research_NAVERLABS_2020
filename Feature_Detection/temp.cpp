// NAVER LABS PLACE-RECOGNITION Competition 2020
// IRiS LAB. KAIST CEE.
// KAIST, W16, #510
// Daejeon, Yuseoung-gu, Rep. of KOREA

// Lee Sang Min
// iismn@kaist.ac.kr

// DBoW3
#include "DBoW3.h"
#include "DescManip.h"
// OPENCV 4.2
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"							// OPENCV LIB
#include "opencv2/xfeatures2d.hpp"				// FEATURES2D LIB (Non-Free Func.)
#include "opencv2/features2d.hpp"					// FEATURED2D LIB (Free Func.)
#include "opencv2/highgui.hpp"
// Visual C++
#include <numeric>
#include <iostream>
// Name Space
using namespace cv;
using namespace std;
using namespace DBoW3;
// Function CALL
vector<Mat> ExtractFeature();
void DBoW_Creation(const vector<Mat> &features);
void DBoW_Validation(const vector<Mat > &features);
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
// FEATURE MATCHING
vector<Mat> ExtractFeature()
{
	// Img Reading
	char filename[200];

	// Parameter Setting
	Ptr<Feature2D> detector = xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
	vector<Mat>    features;

	// START
	cout << "Extracting Extended SURF Features" << endl;

  for (int i=1; i<10; i++){
		// Initialize
		vector<KeyPoint> keypoints;
		Mat descriptors;
		// Image Reading - Target
	  sprintf(filename, "DB/%06d.png", i);
		Mat src = imread(filename, IMREAD_GRAYSCALE);

		// !--Exception--!
		if (src.empty()) {
			cerr << "Image load failed!" << endl;
		}
		//
		// ExSURF Detector - Descriptor
		Mat desc;																									// Descriptor Announce
		detector->detectAndCompute(src, Mat(), keypoints, desc);	// Image Feature Points
		features.push_back(desc);
		cout << "SURF Detector End : ITER(" << i << ")" << endl;
	}
	return features;
}

/// --------- DBOW3
// DBoW Creation
void DBoW_Creation(const vector<Mat> &features)
{
    // Branching Factor / Depth Levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    voc.create(features);

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    cout << "Matching Images(SCORE 0~1): " << endl;
    BowVector v1, v2;

		// Compare Input Image to Validation DB
    voc.transform(features[0], v1);
    for(size_t j = 0; j < features.size(); j++)
    {
        voc.transform(features[j], v2);

        double score = voc.score(v1, v2);
        cout << "Origin Image " << " - Target Image " << j << ": " << score << endl;
    }


    // Save DBoW DB
    cout << endl << "Saving vocabulary" << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}
// DBoW Validation
// DBoW Validation
void DBoW_Validation(const vector<Mat > &features)
{
    cout << "LOAD DBoW3 DB" << endl;
    Vocabulary voc("small_voc.yml.gz");
    Database db(voc, false, 0);

    // Add Img from DB
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    QueryResults ret;
    db.query(features[0], ret, 4);
    // Best Matching Image Out
    cout << "Searching for Original Image" << ". " << ret << endl;
    cout << "Best Match Index : " << ret[1].Id << " / Score : " << ret[1].Score << endl;

		////// Img MATCHER
		Mat src1;
		Mat src2;
		char Original[200];
		char Target[200];
		// FLANN BASED MATCHER
		vector<DMatch> good_matches;
		Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
		sprintf(Original, "DB/%06d.jpg", 1);
		src1 = imread(Original, IMREAD_GRAYSCALE);
		sprintf(Target, "DB/%06d.jpg", ret[1].Id+1);
		src2 = imread(Target, IMREAD_GRAYSCALE);

		// SIFT Detector - Descriptor
		Mat desc1, desc2;																						// Descriptor Announce
		vector<KeyPoint> keypoints1, keypoints2;
		detector->detectAndCompute(src1, Mat(), keypoints1, desc1);	// First Image Feature Points
		detector->detectAndCompute(src2, Mat(), keypoints2, desc2);	// Second Image Feature Points

		Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
		vector<vector<DMatch>> matches;

		matcher->knnMatch(desc1, desc2, matches, 2);
		const float ratio_thresh = 0.6f;

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

		waitKey();
		destroyAllWindows();

    db.save("small_db.yml.gz");
    Database db2("small_db.yml.gz");
}
