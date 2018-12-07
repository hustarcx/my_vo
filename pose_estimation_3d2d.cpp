#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void feature_extration(
  const Mat& img_1, 
  const Mat& img_2,
  std::vector<KeyPoint>& keypoints_1,
  std::vector<KeyPoint>& keypoints_2,
  std::vector<DMatch>& good_matches);

Point2d pixel2cam ( const Point2d& p, const Mat& K );

int main ( int argc, char** argv )
{
  if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
  Mat img_1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
  Mat img_2=imread(argv[2],CV_LOAD_IMAGE_COLOR);
  vector<KeyPoint> keypoints_1,keypoints_2;
  vector<DMatch> matches;
  feature_extration(img_1,img_2,keypoints_1,keypoints_2,matches);
  cout<<"the total number of matches:"<<matches.size()<<endl;
  
  Mat d1=imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
  Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for(DMatch m:matches)
  {
    ushort d=d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if(d==0)
      continue;
    float dd=d/1000.0;
    Point2d p1=pixel2cam(keypoints_1[m.queryIdx].pt,K);
    pts_3d.push_back(Point3f(p1.x*dd,p1.y*dd,dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }
  cout<<"3d-2d pairs number:"<<pts_3d.size()<<endl;
  Mat r,t;
  solvePnP(pts_3d,pts_2d,K,Mat(),r,t,false,cv::SOLVEPNP_EPNP);
  Mat R;
  cv::Rodrigues(r,R);
  cout<<"R=:"<<endl<<R<<endl;
  cout<<"t=:"<<endl<<t<<endl;
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