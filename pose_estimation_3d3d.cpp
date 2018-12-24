#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void feature_extration(
  const Mat& img_1, 
  const Mat& img_2,
  std::vector<KeyPoint>& keypoints_1,
  std::vector<KeyPoint>& keypoints_2,
  std::vector<DMatch>& good_matches);
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void pose_estimation_3d3d(
  const vector<Point3f>& pts1,
  const vector<Point3f>& pts2,
  Mat& R, Mat& t
);

void bundleAdjustment(
  const vector<Point3f>& pts1,
  const vector<Point3f>& pts2,
  Mat& R, Mat& t
);

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;
};

int main ( int argc, char** argv )
{
  if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
  Mat img_1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
  Mat img_2=imread(argv[2],CV_LOAD_IMAGE_COLOR);
  vector<KeyPoint> keypoints_1,keypoints_2;
  vector<DMatch> matches;
  feature_extration(img_1,img_2,keypoints_1,keypoints_2,matches);
  cout<<"the total number of matches:"<<matches.size()<<endl;
  
  Mat d1=imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
  Mat d2=imread(argv[4],CV_LOAD_IMAGE_UNCHANGED);
  Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
  vector<Point3f> pts_3d_1;
  vector<Point3f> pts_3d_2;
  for(DMatch m:matches)
  {
    ushort d_1=d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d_2=d2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if(d_1==0||d_2==0)
      continue;
    Point2d p1=pixel2cam(keypoints_1[m.queryIdx].pt,K);
    Point2d p2=pixel2cam(keypoints_2[m.trainIdx].pt,K);
    float dd1 = float ( d_1 ) /5000.0;
    float dd2 = float ( d_2 ) /5000.0;
    pts_3d_1.push_back(Point3f(p1.x*dd1,p1.y*dd1,dd1));
    pts_3d_2.push_back(Point3f(p2.x*dd1,p2.y*dd2,dd2));
  }
  cout<<"3d 3d pairs number:"<<pts_3d_1.size()<<endl;
  Mat R,t;
  pose_estimation_3d3d(pts_3d_1,pts_3d_2,R,t);
  cout<<"ICP via SVD results: "<<endl;
  cout<<"R = "<<R<<endl;
  cout<<"t = "<<t<<endl;
  cout<<"R_inv = "<<R.t() <<endl;
  cout<<"t_inv = "<<-R.t() *t<<endl;
  
  
  cout<<"calling bundle adjustment"<<endl;

    bundleAdjustment( pts_3d_1, pts_3d_2, R, t );

    // verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts_3d_1[i]<<endl;
        cout<<"p2 = "<<pts_3d_2[i]<<endl;
        cout<<"(R*p2+t) = "<<
            R * (Mat_<double>(3,1)<<pts_3d_2[i].x, pts_3d_2[i].y, pts_3d_2[i].z) + t
            <<endl;
        cout<<endl;
    }
  return 0;
}
void bundleAdjustment(
  const vector<Point3f>& pts1,
  const vector<Point3f>& pts2,
  Mat& R, Mat& t
)
{
  typedef g2o::BlockSolver<g2o::BlockSolverTraits< 6, 3 >> Block;
  Block::LinearSolverType* linearSlover=new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
  Block* solver_ptr=new Block(linearSlover);
  g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  
  //verter
  g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(Eigen::Matrix3d::Identity(),Eigen::Vector3d(0,0,0)));
  optimizer.addVertex(pose);
  int index = 1;
  vector<EdgeProjectXYZRGBDPoseOnly*> edges;
  for (size_t i=0;i<pts1.size();i++)
  {
    EdgeProjectXYZRGBDPoseOnly* edge=new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    edge->setId( index );
    edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );
    edge->setMeasurement( Eigen::Vector3d(
            pts1[i].x, pts1[i].y, pts1[i].z) );
    edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
    optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
  }
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
  
}
void pose_estimation_3d3d(
  const vector<Point3f>& pts1,
  const vector<Point3f>& pts2,
  Mat& R, Mat& t
)
{
  Point3f p1,p2;
  int N=pts1.size();
  for (int i=0;i<N;i++)
  {
    p1+=pts1[i];
    p2+=pts2[i];
  }
  p1=Point3f(Vec3f(p1)/N);
  p2=Point3f(Vec3f(p2)/N);
  vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
  for ( int i=0; i<N; i++ )
  {
    q1[i]=pts1[i]-p1;
    q2[i]=pts2[i]-p2;
  }
  Eigen::Matrix3d W=Eigen::Matrix3d::Zero();
  for ( int i=0; i<N; i++ )
  {
    W+=Eigen::Vector3d(q1[i].x,q1[i].y,q1[i].z)*Eigen::Vector3d(q2[i].x,q2[i].y,q2[i].z).transpose();
  }
  cout<<"W="<<W<<endl;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU|Eigen::ComputeFullV);
  Eigen::Matrix3d U=svd.matrixU();
  Eigen::Matrix3d V=svd.matrixV();
  cout<<"U="<<U<<endl;
  cout<<"V="<<V<<endl;
  
  Eigen::Matrix3d R_=U*(V.transpose());
  Eigen::Vector3d t_=Eigen::Vector3d(p1.x,p1.x,p1.z)-R_*Eigen::Vector3d(p2.x,p2.y,p2.z);
  
  // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
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