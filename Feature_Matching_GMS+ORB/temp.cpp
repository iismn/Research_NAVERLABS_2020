//
// Created by joongku on 20. 6. 22..
//

//
// Created by joongku on 20. 6. 12..
//

#include <ros/ros.h>
#include <test/hdf5_reader.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_broadcaster.h>
#include "json/json.h"
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Int8.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>


using namespace std;

Json::Value Pose_list; // List for Saving Overall Pose

struct XYZ{
		float x;
		float y;
		float z;
};

//// Get Camera Matrix and Saves it to cam_matrix
void get_cam_parameters(string sensor_name, cv::Mat& cam_matrix, cv::Mat& dist_coef, int& cwidth, int& cheight)
{
	string path = "/run/user/1000/gvfs/smb-share:server=irislab-nas.local,share=irislab_nas/SLAM_DATASET/NAVERLABS/INDOOR/dataset/camera_parameters_merged.txt";
	ifstream fileinput(path);
	string line;
	string search = sensor_name;
	int found = 0;

	while (getline(fileinput, line))
	{
		if (line.find(search,0) != string::npos)
		{
			string data[12];
			int i = 0;
			char linechar[150];
			strcpy(linechar, line.c_str()); // string에는 strtok 못써서 char로 만들어줌
			char *tok = strtok(linechar, " ");
			while (tok != NULL) {
				data[i] = tok;
				tok = strtok(NULL, " ");
				i++;
			}

			cam_matrix = (cv::Mat_<double>(3, 3) << stod(data[3]), 0, stod(data[5]), 0, stod(data[4]), stod(
							data[6]), 0, 0, 1);
			dist_coef = (cv::Mat_<double>(5,1) << stod(data[7]),stod(data[8]),stod(data[9]),stod(data[10]),stod(data[11]));
			cwidth = stoi(data[1]);
			cheight = stoi(data[2]);

			found = 1;
		}
	}
	if (!found)
		cout << "Check Your Sensor Name" << endl;
}

vector<float> XYZtoPixel(const Eigen::Matrix3d& K, const Eigen::Matrix4d& m2c, Eigen::Vector3d pose){
	Eigen::Matrix<double, 3, 4> m2c_bar;
	m2c_bar << m2c(0,0), m2c(0,1), m2c(0,2), m2c(0,3),
					m2c(1,0), m2c(1,1), m2c(1,2), m2c(1,3),
					m2c(2,0), m2c(2,1), m2c(2,2), m2c(2,3);
	Eigen::Matrix<double, 3, 4> KT = K*m2c_bar;
	Eigen::Vector4d p(pose[0], pose[1], pose[2], 1);
	Eigen::Vector3d pixel = KT*p;

	vector<float> px;
	px.push_back(pixel[0]/pixel[2]);
	px.push_back(pixel[1]/pixel[2]);

	return px;
}

HDF5Reader::StampPose find_image_gt(string train_image, HDF5Reader ground_truth)
{
	int string_idx = train_image.find("_");
	string sensor_name = train_image.substr(0, string_idx);
	int64_t camera_stamp = stoll(train_image.substr(string_idx+1, 16));

	HDF5Reader::StampPose cStampPose;
	auto image_gt = ground_truth.data_map[sensor_name];
	for(int i=0; i<image_gt.size(); ++i)
	{
		if(image_gt[i].stamp == camera_stamp)
		{
			cStampPose = image_gt[i];
			break;
		}
	}
	return cStampPose;
}

void pcd_read(string path, string image_filename, HDF5Reader ground_truth, HDF5Reader::StampPose cStampPose, cv::Mat cam_matrix, int cwidth, int cheight)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr local_map_ds(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_localmap_pcl(new pcl::PointCloud<pcl::PointXYZ>);

	auto lidar0_gt = ground_truth.data_map["lidar0"];
	auto lidar1_gt = ground_truth.data_map["lidar1"];

	//// Find nearest data of camera_idx
	int lidar_idx = 0;
	for(int i=0; i<lidar0_gt.size(); ++i){
		if(cStampPose.stamp - lidar0_gt[i].stamp < 0){
			lidar_idx = i;
//			cout << "lidar_idx : " << i << endl;
			break;
		}
	}

	//// Build Local Point Cloud Map
	cout << "Build Local Map ...";
	fflush(stdout);
	int localmap_size = 100;
	for(int i=lidar_idx-localmap_size; i<lidar_idx+localmap_size; ++i) {
		if(i<0 || i>=lidar0_gt.size())
			continue;
		auto stamp = lidar0_gt[i].stamp;

		string pcd_file = path + "pointclouds_data/" + "lidar0_"+ to_string(stamp)+".pcd";
		Eigen::Vector3f t(lidar0_gt[i].x,lidar0_gt[i].y,lidar0_gt[i].z);
		Eigen::Quaternionf q(lidar0_gt[i].qw,lidar0_gt[i].qx,lidar0_gt[i].qy,lidar0_gt[i].qz);
		pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file.c_str(), *cloud);
		pcl::transformPointCloud(*cloud, *transformed_cloud, t,q);

		*local_map += *transformed_cloud;
	}
	for(int i=lidar_idx-localmap_size; i<lidar_idx+localmap_size; ++i) {
		if(i<0 || i>=lidar1_gt.size())
			continue;
		auto stamp = lidar1_gt[i].stamp;

		string pcd_file = path + "pointclouds_data/" + "lidar1_"+ to_string(stamp)+".pcd";
		Eigen::Vector3f t(lidar1_gt[i].x,lidar1_gt[i].y,lidar1_gt[i].z);
		Eigen::Quaternionf q(lidar1_gt[i].qw,lidar1_gt[i].qx,lidar1_gt[i].qy,lidar1_gt[i].qz);

		pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file.c_str(), *cloud);
		pcl::transformPointCloud(*cloud, *transformed_cloud, t,q);

		*local_map += *transformed_cloud;
	}

	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(local_map);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(1.35, 20.0);
	pass.filter(*local_map_ds);
	cout << "Done!" << endl;

	//// project local map to image frame
	Eigen::Translation3d t(cStampPose.x, cStampPose.y, cStampPose.z);
	Eigen::Quaterniond quat(cStampPose.qw, cStampPose.qx, cStampPose.qy, cStampPose.qz);
	Eigen::Matrix4d m2c = (t*quat.normalized().toRotationMatrix()).matrix();

	cout << "transformed local map .. ";
	fflush(stdout);
	pcl::transformPointCloud(*local_map_ds,*transformed_localmap_pcl, m2c.inverse().cast<float>());
	cout << "Done!" << endl;
	Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
	double fx, fy, cx, cy;
	fx = cam_matrix.at<double>(0,0);
	fy = cam_matrix.at<double>(1,1);
	cx = cam_matrix.at<double>(0,2);
	cy = cam_matrix.at<double>(1,2);
	K(0,0)=fx;
	K(0,2)=cx;
	K(1,1)=fy;
	K(1,2)=cy;

	//// Downsampling with Camera FOV
	cout << "Local map pass through filtering... ";
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
	// filtering -z points
	pass.setInputCloud(transformed_localmap_pcl);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0, 20.0);
	pass.filter(*local_map_ds);
	// rotate -45deg at Y axis and filtering
	Eigen::Vector3f t1(0, 0, 0);
	Eigen::Quaternionf q1(0.9238795, 0, -0.3826834, 0);
	pcl::transformPointCloud(*local_map_ds, *transformed_cloud1, t1,q1);
	pass.setInputCloud(transformed_cloud1);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(-200, 0);
	pass.filter(*temp);
	// rotate 90deg at Y axis and filtering
	q1.w() = 0.7071068;
	q1.y() = 0.7071068;
	pcl::transformPointCloud(*temp, *transformed_cloud1, t1,q1);
	pass.setInputCloud(transformed_cloud1);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(0, 200);
	pass.filter(*temp);
	// rotate -45deg at Y axis
	q1.w() = 0.9238795;
	q1.y() = -0.3826834;
	pcl::transformPointCloud(*temp, *local_map_ds, t1,q1);
	cout << "Done!" << endl;

	//// read pixel and make rgb local map
	cout << "Mapping RGB value in Local Map ...";
	fflush(stdout);
	Eigen::Vector3d pose;
	map<int, XYZ> image_xyz_map;
//	cout << "local map size : " << local_map_ds->size();

	int downsampling = 4;
	for(auto iter=local_map_ds->points.begin(); iter!=local_map_ds->points.end(); ++iter){
		if(iter->z > 0) {
			pose[0] = iter->x;
			pose[1] = iter->y;
			pose[2] = iter->z;
			vector<float> px = XYZtoPixel(K, Eigen::Matrix4d::Identity(),pose);
			if(round(px[0])>=0 && round(px[0])<cwidth
			   && round(px[1])>=0 && round(px[1])<cheight){

				int idx = round(px[1]/downsampling)*(cwidth/downsampling)+round(px[0]/downsampling);
				if(image_xyz_map.find(idx) != image_xyz_map.end()){
					float dist1 = sqrtf(powf(image_xyz_map[idx].x, 2)+powf(image_xyz_map[idx].y, 2)+powf(image_xyz_map[idx].z, 2));
					float dist2 = sqrtf(powf(iter->x, 2)+powf(iter->y, 2)+powf(iter->z, 2));
					if(dist1<dist2)
						continue;
				}

				image_xyz_map[idx].x = iter->x;
				image_xyz_map[idx].y = iter->y;
				image_xyz_map[idx].z = iter->z;
			}
		}
	}

	float *data;
	data = new float[image_xyz_map.size()*4];

	int i=0;
	for(auto iter=image_xyz_map.begin(); iter!=image_xyz_map.end(); ++iter){

		//// transformed to Map coordinate system
		float tx = m2c(0,0)*iter->second.x+m2c(0,1)*iter->second.y+m2c(0,2)*iter->second.z+m2c(0,3);
		float ty = m2c(1,0)*iter->second.x+m2c(1,1)*iter->second.y+m2c(1,2)*iter->second.z+m2c(1,3);
		float tz = m2c(2,0)*iter->second.x+m2c(2,1)*iter->second.y+m2c(2,2)*iter->second.z+m2c(2,3);

		data[i*4] = iter->first;
		data[i*4+1] = tx;
		data[i*4+2] = ty;
		data[i*4+3] = tz;

		i++;
	}
	cout << " Done!" << endl;
	printf("Image XYZ filled (%d/%d)\n", cwidth*cheight/downsampling/downsampling, image_xyz_map.size());

	//// Write image map HDF5 file
	float camera_pose[7] = {cStampPose.x, cStampPose.y, cStampPose.z, cStampPose.qw, cStampPose.qx, cStampPose.qy, cStampPose.qz};
	int string_idx = image_filename.find(".");
	string image_name = image_filename.substr(0, string_idx);
	ground_truth.WriteImageMapData("/home/joongku/Desktop/"+image_name+".hdf5", image_xyz_map.size(), camera_pose, data);

	delete[] data;
}

void depth_estimation(string path, string image_filename, HDF5Reader ground_truth, HDF5Reader::StampPose cStampPose, cv::Mat cam_matrix, int cwidth, int cheight){
	int string_idx = image_filename.find("_");
	string sensor_name = image_filename.substr(0, string_idx);
	int64_t camera_stamp = stoll(image_filename.substr(string_idx+1, 16));

	auto image_gt = ground_truth.data_map[sensor_name];
	for(int i=0; i<image_gt.size(); ++i){
		if(image_gt[i].stamp == camera_stamp){
			cStampPose = image_gt[i];
			break;
		}
	}

	pcd_read(path, image_filename, ground_truth, cStampPose, cam_matrix, cwidth, cheight);
}

string search_image_path(string image)
{
	string path = "/run/user/1000/gvfs/smb-share:server=irislab-nas.local,share=irislab_nas/SLAM_DATASET/NAVERLABS/INDOOR/dataset/image_and_path.txt";
	ifstream fileinput(path);
	string line;
	string search = image;
	int found = 0;
	string final_path;

	while (getline(fileinput, line))
	{
		if (line.find(search,0) != string::npos)
		{

			string data[6];
			int i = 0;
			char linechar[250];
			strcpy(linechar, line.c_str());
			char *tok = strtok(linechar,"'");
			while (tok != NULL) {
				data[i] = tok;
				tok = strtok(NULL, "'");
				i++;
			}
//			cout<<data[0]<<endl; // Contains image name
//			cout<<data[2]<<endl; // Contains Path
			int string_idx = data[2].find("images");
			final_path = data[2].substr(0,string_idx);
//			cout<<final_path<<endl;
			found = 1;
		}
	}
	if (found)
		return final_path;
	else
		cout << "Check Your Sensor Name" << endl;
}

string image_to_sensor_name(string image_name)
{
	int string_idx = image_name.find("_");
	string sensor_name = image_name.substr(0,string_idx);

	return sensor_name;
}

//// Match Scaling Coefficient!!!!!!!!!
vector<cv::Point3d> map_to_vec(vector<cv::Point2f> train_feature, HDF5Reader point_3d, cv::Mat cam_matrix, vector<int>& feature_idx)
{
	vector<cv::Point3d> matched_points_3d;
	int width = cam_matrix.at<double>(0,2) * 2;
	int scaling = 4; //// Match Scaling Coefficient!!!!!!!!!
	for (int i = 0; i<train_feature.size(); ++i)
	{
		int px = round(train_feature[i].x);
		int py = round(train_feature[i].y);
		int num = round(py/scaling)*width/scaling + round(px/scaling);

		auto iter = point_3d.point3d_map.find(num);
		if(iter != point_3d.point3d_map.end()) // key == num found!
		{
			feature_idx.push_back(i);
			cv::Point3d points;
			points.x = point_3d.point3d_map[num].x;
			points.y = point_3d.point3d_map[num].y;
			points.z = point_3d.point3d_map[num].z;
			matched_points_3d.push_back(points);
		}
//		else // if key == num not found!
//		{
//			//// Check if there exist 3d points for pixels around
//			for (int j = 0; j<9;++j)
//			{
//				if (j == 4) // Original location
//					continue;
//
//				float k = (float)j;
//				int new_num = num + floor((k-3)/3)*width/scaling + (j%3)-1;
//				auto new_iter = point_3d.point3d_map.find(new_num);
//
//				if(iter != point_3d.point3d_map.end()) // key == new_num found!
//				{
//					feature_idx.push_back(i);
//					cv::Point3d points;
//					points.x = point_3d.point3d_map[new_num].x;
//					points.y = point_3d.point3d_map[new_num].y;
//					points.z = point_3d.point3d_map[new_num].z;
//					matched_points_3d.push_back(points);
//					break;
//				}
//			}
//		}
	}
	return matched_points_3d;
}

vector<cv::Point2f> get_matched_feature(vector<cv::Point2f> test_feature, vector<int> feature_idx)
{
	vector<cv::Point2f> matched_feature;
	for (int i=0; i < feature_idx.size(); ++i)
	{
		int idx = feature_idx[i];
		matched_feature.push_back(test_feature[idx]);
	}
	return matched_feature;
}

HDF5Reader::StampPose get_train_gt(string train_image, HDF5Reader ground_truth)
{
	int string_idx = train_image.find("_");
	string sensor_name = train_image.substr(0,string_idx);
	int64_t camera_stamp = stoll(train_image.substr(string_idx+1,16));
	HDF5Reader::StampPose cStampPose;

	auto image_gt = ground_truth.data_map[sensor_name];
	for (int i=0; i<image_gt.size(); ++i)
	{
		if (image_gt[i].stamp == camera_stamp)
		{
			cStampPose = image_gt[i];
			break;
		}
	}
	return cStampPose;
}

void save_Json(Json::Value Pose)
{
	Pose_list.append(Pose);
	cout<<"Stacked number of Poses: "<<Pose_list.size()<<endl;
	ofstream json_file;
	string out_file = "/home/joongku/Desktop/0624_1f.json";
	json_file.open(out_file,ios_base::out|ios_base::trunc); // Remove Old data when updating

	Json::StreamWriterBuilder builder;
	builder["commentStyle"] = "None";
	builder["indentation"] = "    ";
	unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

//	writer->write(Pose_list, &cout); // Prints lists to command line
	writer->write(Pose_list, &json_file);
	cout << endl;

	json_file.close();
}

void save_pnp_unsolved(string train_image, string test_image)
{
	ofstream writeFile;
	string out_txt = "/home/joongku/Desktop/0624_1f_unsolved.txt";
	writeFile.open(out_txt, ios_base::app);

	if (writeFile.is_open())
	{
		writeFile << train_image << ".jpg ";
		writeFile << test_image << ".jpg\n";
		writeFile.close();
	}
}

Json::Value solve_campose(vector<cv::Point2f> image_points, vector<cv::Point3d> space_points, cv::Mat cam_matrix, cv::Mat dist_coef, string train_image, string test_image, HDF5Reader ground_truth)
{
	HDF5Reader::StampPose cStampPose = get_train_gt(train_image, ground_truth);
	cv::Mat rotation_vector;
	cv::Mat translation_vector = (cv::Mat_<double>(3,1) << cStampPose.x, cStampPose.y, cStampPose.z);

	Eigen::Quaterniond q;
	q.w() = cStampPose.qw;
	q.x() = cStampPose.qx;
	q.y() = cStampPose.qy;
	q.z() = cStampPose.qz;
	Eigen::Matrix3d q_mat(q);
	eigen2cv(q_mat,rotation_vector);

	// Solve R, t using solvePnP (Returns R,t from Camera's Coordinate to World Coordinate)
	//  cv::solvePnP(space_points, image_points, cam_matrix, dist_coef, rotation_vector, translation_vector);
	bool success;
	// Solve R, t using solvePnPRansac
	try
	{
		success = cv::solvePnPRansac(space_points, image_points, cam_matrix, dist_coef, rotation_vector, translation_vector, true);
	}
	catch (cv::Exception& exception)
	{
		const char* err_msg = exception.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		success = false;
	}

	if (success) // If PnP Solve Success
	{
		// Extract Rotation & Translation Matrix from Rotation & Translation Vector
		cv::Mat R;
		cv::Rodrigues(rotation_vector, R);
		cv::Mat R_inv = R.t(); // SO(3)니까 Inverse 랑 Transpose랑 같긴 할텐데 혹시 모르니 나중에 확인해보자
		cv::Mat P = -R_inv * translation_vector; // Calculated Translation Vector

		// H는 World 기준 Cam의 Pose (in Homogeneous Transform Matrix Form)
		cv::Mat H(4, 4, R_inv.type());
		H(cv::Range(0,3),cv::Range(0,3)) = R_inv * 1;
		H(cv::Range(0,3), cv::Range(3,4)) = P * 1;

		Eigen::Matrix3d eigenR; // Eigen 3x3 행렬 선언
		cv2eigen(R_inv,eigenR); // cv Matrix를 Eigen Matrix로 변환.
		Eigen::Quaterniond q(eigenR);

		cout<<"Calculated Pose for Test Image: "<<endl;
		cout<<"x : "<<H.at<double>(0,3)<<endl;
		cout<<"y : "<<H.at<double>(1,3)<<endl;
		cout<<"z : "<<H.at<double>(2,3)<<endl;
		cout<<"qw : "<<q.w()<<endl;
		cout<<"qx : "<<q.x()<<endl;
		cout<<"qy : "<<q.y()<<endl;
		cout<<"qz : "<<q.z()<<endl;

		Json::Value Pose;
		Pose["floor"] = "1f"; // Have to Change Manually
		Pose["name"] = test_image + ".jpg";
		Pose["qw"] = q.w();
		Pose["qx"] = q.x();
		Pose["qy"] = q.y();
		Pose["qz"] = q.z();
		Pose["x"] = H.at<double>(0,3);
		Pose["y"] = H.at<double>(1,3);
		Pose["z"] = H.at<double>(2,3);

		return Pose;
	}

	else
	{
		cout<<"PnP Solve failed!"<<endl;
		// Saves unsolved image list
		save_pnp_unsolved(train_image, test_image);
		cout<<"Calculated Pose for Test Image: "<<endl;
		cout<<"x : "<<cStampPose.x<<endl;
		cout<<"y : "<<cStampPose.y<<endl;
		cout<<"z : "<<cStampPose.z<<endl;
		cout<<"qw : "<<cStampPose.qw<<endl;
		cout<<"qx : "<<cStampPose.qx<<endl;
		cout<<"qy : "<<cStampPose.qy<<endl;
		cout<<"qz : "<<cStampPose.qz<<endl;

		Json::Value Pose;
		Pose["floor"] = "1f"; // Have to Change Manually
		Pose["name"] = test_image + ".jpg";
		Pose["qw"] = cStampPose.qw;
		Pose["qx"] = cStampPose.qx;
		Pose["qy"] = cStampPose.qy;
		Pose["qz"] = cStampPose.qz;
		Pose["x"] = cStampPose.x;
		Pose["y"] = cStampPose.y;
		Pose["z"] = cStampPose.z;

		return Pose;
	}
}

void solve(string train_image, string test_image, vector<cv::Point2f> train_feature, vector<cv::Point2f> test_feature)
{
	cout<< "Train Image : " << train_image << endl;
	cout<< "Test Image : " << test_image << endl;

	//// -- Extract Camera Parameters for train image and test image
	cv::Mat train_cam_matrix, train_dist_coef, test_cam_matrix, test_dist_coef;
	int train_cwidth, train_cheight, test_cwidth, test_cheight;

	string train_sensor = image_to_sensor_name(train_image);
	string test_sensor = image_to_sensor_name(test_image);

	get_cam_parameters(train_sensor, train_cam_matrix, train_dist_coef, train_cwidth, train_cheight);
	get_cam_parameters(test_sensor, test_cam_matrix, test_dist_coef, test_cwidth, test_cheight);

	//// -- Make Map for that train image
	string train_path = search_image_path(train_image);
	HDF5Reader ground_truth;
	ground_truth.ReadData(train_path + "groundtruth.hdf5");
	HDF5Reader::StampPose cStampPose = find_image_gt(train_image, ground_truth);
	HDF5Reader point_3d;
	string hdf5_path = "/run/user/1000/gvfs/smb-share:server=irislab-nas.local,share=irislab_nas/SLAM_DATASET/NAVERLABS/INDOOR/dataset/1f/train/depth/" + train_image + ".hdf5";
	point_3d.Read3DData(hdf5_path);
	vector<int> feature_idx;
	vector<cv::Point3d> matched_points_3d = map_to_vec(train_feature, point_3d, train_cam_matrix, feature_idx);
	vector<cv::Point2f> matched_feature = get_matched_feature(test_feature, feature_idx);

	cout<<"Number of Features Given :" << train_feature.size() << endl;
	cout<<"Number of Matched Features : " << matched_feature.size() << endl;

	//// -- Solve Cam Pose
	Json::Value Pose = solve_campose(matched_feature, matched_points_3d, test_cam_matrix, test_dist_coef, train_image, test_image, ground_truth);

	//// -- Save to Json
	save_Json(Pose);

}

void RootSIFT(cv::Mat &desc, cv::Mat& RootSIFTdesc)
{
	for(int i = 0; i < desc.rows; i ++){
		// Convert to float type
		cv::Mat f;
		desc.row(i).convertTo(f,CV_32FC1);

		normalize(f,f,1,0,cv::NORM_L1); // l1 normalize
		sqrt(f,f); // sqrt-root  root-sift
		RootSIFTdesc.push_back(f);
	}
}

void get_feature(string train_image, string test_image, vector<cv::Point2f>& train_feature, vector<cv::Point2f>& test_feature)
{
	string path_to_train = search_image_path(train_image);
	string train_path = path_to_train + "images/" + train_image + ".jpg";
	string test_path = "/run/user/1000/gvfs/smb-share:server=irislab-nas.local,share=irislab_nas/SLAM_DATASET/NAVERLABS/INDOOR/dataset/1f/test/2019-08-21_12-10-13/images/" + test_image + ".jpg";
	cv::Mat src1;
	cv::Mat src2;
	char Original[200];
	strcpy(Original,train_path.c_str());
	char Target[200];
	strcpy(Target,test_path.c_str());

	//// -- FLANN Based Matcher
	vector<cv::DMatch> good_matches;
	cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
	src1 = cv::imread(Original, cv::IMREAD_GRAYSCALE);
	src2 = cv::imread(Target, cv::IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty())
		cout<<"Could not open or find the Image"<<endl;

	//// -- SIFT Detector-Descriptor
	cv::Mat desc1, desc2, Rdesc1, Rdesc2;
	vector<cv::KeyPoint> keypoints1, keypoints2;
	detector->detectAndCompute(src1, cv::Mat(), keypoints1, desc1);
	detector->detectAndCompute(src2, cv::Mat(), keypoints2, desc2);
	RootSIFT(desc1, Rdesc1);
	RootSIFT(desc2, Rdesc2);
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
	vector<vector<cv::DMatch>> matches;

	matcher->knnMatch(Rdesc1, Rdesc2, matches, 2);
	const float ratio_thresh = 0.5f;

	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
			if (isnan(good_matches[i].distance)){
				good_matches[i].distance = 0;
			}
		}
	}

//	cv::Mat dst;
//	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
//	            cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(),
//	            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	imshow("dst", dst);
//	cv::waitKey(0);

	for (int i=0;i<good_matches.size();i++)
	{
		// this is how the DMatch structure stores the matching information
		int idx1=good_matches[i].trainIdx;
		int idx2=good_matches[i].queryIdx;

		//now use those match indices to get the keypoints, add to your two lists of points
		train_feature.push_back(keypoints1[idx2].pt);
		test_feature.push_back(keypoints2[idx1].pt);
	}

}

vector<string> get_matched_list(string path_to_list)
{
	vector<string> list;
	ifstream fileinput(path_to_list);
	string line;

	while(getline(fileinput, line))
	{
		string data;
		char linechar[100];
		strcpy(linechar, line.c_str());
		char *tok = strtok(linechar, "."); // \n
		data = tok;
		list.push_back(data);
	}
	return list;
}

void get_feature_and_solve()
{
	//// -- Get Name of the matched images from txt
	string path_to_train = "/run/user/1000/gvfs/smb-share:server=irislab-nas.local,share=irislab_nas/SLAM_DATASET/1F_Database.txt";
	vector<string> train_list = get_matched_list(path_to_train);
	string path_to_test = "/run/user/1000/gvfs/smb-share:server=irislab-nas.local,share=irislab_nas/SLAM_DATASET/1F_Query.txt";
	vector<string> test_list = get_matched_list(path_to_test);

	//// -- Get features and Solve!
	if (train_list.size() != test_list.size())
		cout<<"Size of the list is different!"<<endl;
	else
	{
		for (int i = 0; i < train_list.size();i++)
		{
			string train_image = train_list[i];
			string test_image = test_list[i];
			vector<cv::Point2f> train_feature;
			vector<cv::Point2f> test_feature;
			get_feature(train_image, test_image, train_feature, test_feature);
			solve(train_image, test_image, train_feature, test_feature);
		}
	}
}

int main(int argc, char **argv){
	ros::init(argc, argv, "rootsift_pnp_1f_node");
	ros::NodeHandle n;
	ros::NodeHandle pn("~");

	get_feature_and_solve();

	ros::spin();

	return 0;
}
