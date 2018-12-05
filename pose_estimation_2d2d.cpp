#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,std::vector<KeyPoint> keypoints_2,std::vector<DMatch> matches,Mat& R, Mat& t);

void feature_extration(
  const Mat& img_1, 
  const Mat& img_2,
  std::vector<KeyPoint>& keypoints_1,
  std::vector<KeyPoint>& keypoints_2,
  std::vector<DMatch>& good_matches);

Point2d pixel2cam ( const Point2d& p, const Mat& K );

int main ( int argc, char** argv )
{
  Mat R,t;
  if(argc!=3)
    {
      cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
      return 1;
    }
  Mat img_1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
  Mat img_2=imread(argv[2],CV_LOAD_IMAGE_COLOR);
  vector<KeyPoint> keypoints_1,keypoints_2;
  vector<DMatch> matches;
  feature_extration(img_1,img_2,keypoints_1,keypoints_2,matches);
  cout<<"the total number of matches:"<<matches.size()<<endl;
  pose_estimation_2d2d(keypoints_1,keypoints_2,matches,R,t);
  Mat t_x=( Mat_<double> ( 3,3 ) <<
                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
                -t.at<double> ( 1,0 ),     t.at<double> ( 0,0 ),      0 );
  cout<<"t^x="<<endl<<t_x*R<<endl;
  Mat K=(Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  
  for(DMatch m:matches)
  {
    Point2d pt1=pixel2cam(keypoints_1[m.queryIdx].pt,K);
    Mat y1=(Mat_<double>(3,1)<<pt1.x,pt1.y,1);
    Point2d pt2=pixel2cam(keypoints_1[m.trainIdx].pt,K);
    Mat y2=(Mat_<double>(3,1)<<pt2.x,pt2.y,1);
    Mat d=y2.t()*t_x*R*y1;
    cout<<"epipolar constraint="<<d<<endl;
  }
  
  return 0;
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat& R, Mat& t)
{
  Mat K=(Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point2f> points1;
  vector<Point2f> points2;
  for (int i=0;i<matches.size();++i)
  {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }
  
  Mat fundamental_matrix;
  fundamental_matrix=findFundamentalMat(points1,points2,CV_FM_8POINT);
  cout<<"fundamental matrix is:"<<endl<<fundamental_matrix<<endl;
  
  Point2d principal_point(325.1,249.7);
  int focal_length=521;
  Mat essential_matrix;
  essential_matrix=findEssentialMat(points1,points2,focal_length,principal_point,RANSAC);
  cout<<"essential matrix is:"<<endl<<essential_matrix<<endl;
  
  
  Mat homography_matrix;
  homography_matrix=findHomography(points1,points2,RANSAC);
  cout<<"homography matrix is:"<<endl<<homography_matrix<<endl;
  
  //recoverPose(essential_matrix,points1,points2,K,R,t);
  recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
  cout<<"R matrix is:"<<endl<<R<<endl;
  cout<<"t matrix is:"<<endl<<t<<endl;
  
  
}

void feature_extration(
  const Mat& img_1, 
  const Mat& img_2,
  std::vector<KeyPoint>& keypoints_1,
  std::vector<KeyPoint>& keypoints_2,
  std::vector<DMatch>& good_matches)
{
    Mat descriptor_1,descriptor_2;
    Ptr<FeatureDetector> detector=ORB::create();
    Ptr<DescriptorExtractor> descriptor=ORB::create();
    Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
    
    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);
    
    descriptor->compute(img_1,keypoints_1,descriptor_1);
    descriptor->compute(img_2,keypoints_2,descriptor_2);
    
    Mat outimg_1,outimg_2;
    drawKeypoints(img_1,keypoints_1,outimg_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    drawKeypoints(img_2,keypoints_2,outimg_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    //imshow("Feature image 1",outimg_1);
    //imshow("Feature image 2",outimg_2);
    
    vector<DMatch> matches;
    matcher->match(descriptor_1, descriptor_2, matches);

    double min_dist=10000, max_dist=0;
    
    for (int i=0;i<descriptor_1.rows;i++)
    {
      double dist=matches[i].distance;
      if(dist<min_dist) min_dist=dist;
      if(dist>max_dist) max_dist=dist;
    }
    cout<<"min_dist:"<<min_dist<<endl;
    cout<<"max_dist:"<<max_dist<<endl;
    
    //vector<DMatch> good_matches;
    for (int i=0;i<descriptor_1.rows;++i)
    {
      if(matches[i].distance <= max ( 2*min_dist, 30.0 ))
      {
	good_matches.push_back(matches[i]);
      }
    }
    //Mat img_match;
    //Mat img_goodmatch;
    //drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);
   // drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_goodmatch);
    
}