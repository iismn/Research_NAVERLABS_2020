#include "gms_matcher.h"
#include <fstream>
#include <sstream>
//#define USE_GPU
#ifdef USE_GPU
#include <opencv2/cudafeatures2d.hpp>
using cuda::GpuMat;
#endif

int GmsMatch(Mat &img1, Mat &img2);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

void runImagePair() {

    ifstream QFile("../data/Result_B1_Q_0711.txt");
    ofstream FilteredFile;
    FilteredFile.open ("FilteredResult_B1_DB_0711.txt");
    int iteration_Q = 0;
    int iteration_Compare = 0;
    if (QFile.is_open()) {

        string DBLine;
        string QLine;
        string Result_Img;

        while (getline(QFile, QLine)) {
            iteration_Q++;
            string QPath = "/home/iris_dl/IRiS_WS/SangMinLee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_TestDataset/Test/B1/images/";
            QPath += QLine.c_str();
            vector<string> DBPath;
            vector<string> DBFileName;
            int iteration = 1;
            ifstream DBFile("../data/Result_B1_DB_0711.txt");
            int i = 0;
            while (getline(DBFile, DBLine)) {
                if (iteration < iteration_Compare+6 && iteration > iteration_Compare){
                    string DBPath_Temp = "/home/iris_dl/IRiS_WS/SangMinLee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_TestDataset/images/";
                    DBPath_Temp += DBLine.c_str();
                    DBPath.push_back(DBPath_Temp);
                    DBFileName.push_back(DBLine.c_str());
                    i++;
                }
                iteration++;
            }
            DBFile.close();
            iteration_Compare += 5;

            Mat imgQuery = imread(QPath);
            int featureSize = 0;
            int featureSize_Prv = 0;
            for (int i = 0; i < 5; i++){
                Mat imgDataBase = imread(DBPath[i]);
                featureSize = GmsMatch(imgQuery, imgDataBase);
                if (featureSize > featureSize_Prv){
                    featureSize_Prv = featureSize;
                    Result_Img = DBFileName[i];
                }

            }
            cout << "Result Image : " << Result_Img << " / Result Feature Size : " << featureSize_Prv << endl;
            FilteredFile << Result_Img << endl;
            cout << "Iteration of Query Image : " << iteration_Q << endl << endl;
        }
//        imshow("show",img1);
//        waitKey();
        QFile.close();
        FilteredFile.close();

    }
    // GMS+ORB : Loop in Maximize Feature Correspondence

//	featureSize = GmsMatch(img1, img2);

//	cout << featureSize << endl;

}



int main()
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair();

	return 0;
}

int GmsMatch(Mat &img1, Mat &img2) {
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	Ptr<ORB> orb = cv::ORB::create(10000);
	orb->setFastThreshold(0);

	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif

	// GMS filter
	vector<cv::Point2f> train_feature;
	vector<cv::Point2f> test_feature;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
	cout << "Get total " << num_inliers << " matches." << endl;

	// collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	for (int i=0;i<matches_gms.size();i++)
	{
		// this is how the DMatch structure stores the matching information
		int idx1=matches_gms[i].trainIdx;
		int idx2=matches_gms[i].queryIdx;

		//now use those match indices to get the keypoints, add to your two lists of points
		train_feature.push_back(kp1[idx2].pt);
		test_feature.push_back(kp2[idx1].pt);

	}
	// draw matching
	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
//	imshow("show", show);
//	waitKey();

	return train_feature.size();
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}

	return output;
}
