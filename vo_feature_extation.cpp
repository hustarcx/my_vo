#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if(argc!=3)
    {
      cout<<"usage: feature_extraction img1 img2"<<endl;
      return 1;
    }
    Mat img_1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
    Mat img_2=imread(argv[2],CV_LOAD_IMAGE_COLOR);
    //imshow("Orgin image 1",img_1);
    //imshow("Orgin image 2",img_1);
    
    vector<KeyPoint> keypoints_1,keypoints_2;
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
    
    vector<DMatch> good_matches;
    for (int i=0;i<descriptor_1.rows;++i)
    {
      if(matches[i].distance <= max ( 2*min_dist, 30.0 ))
      {
	good_matches.push_back(matches[i]);
      }
    }
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_goodmatch);
    imshow("Original Matches",img_match);
    imshow("Good Matches",img_goodmatch);
    waitKey(0);
    return 0;
}
