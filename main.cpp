/**
    Opencv example code: camera resection computing by solvepnp o matrix operations. Augmented reality cube
    Enrique Marin
    88enrique@gmail.com
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>

// Switch to true to compute
#define SOLVEPNP    true
#define MATRIX      false

using namespace std;
using namespace cv;

int main(){

    // Variables
    VideoCapture capture;
    Mat frame;

    namedWindow("video", CV_WINDOW_AUTOSIZE);

    // Open video file
    capture.open("../Videos/chessboard-1.avi");
    //capture.open(0);

    // Check that the video was opened
    if (!capture.isOpened()){
        cout << "Cannot open video device or file!" << endl;
        return -1;
    }

    // Read the video
    while(true){

        // Read new frame
        capture.read(frame);
        if (frame.empty())
            break;

        // Chessboard size
        Size patternsize(9,6); //interior number of corners
        vector<Point2f> corners; //this will be filled by the detected corners

        // Find corners on a chessboard pattern
        bool patternfound = findChessboardCorners(frame, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        //drawChessboardCorners(frame, patternsize, Mat(corners), patternfound);

        if (patternfound){

            // Image size
            double width = frame.cols;
            double height = frame.rows;

            // Camera matrix
            double k[] = {750,0, width/2, 0, 750, height/2, 0, 0 ,1};
            Mat K = Mat(3, 3, CV_64FC1, k);

            // Correspondences between image points and world points
            vector<Point2f> world2d;
            vector<Point3f> world3d;
            vector<Point2f> img;
            world2d.push_back(Point2f(450,300));
            world3d.push_back(Point3f(450,300,0));
            img.push_back(Point2f(corners.at(0)));
            world2d.push_back(Point2f(50,300));
            world3d.push_back(Point3f(50,300,0));
            img.push_back(Point2f(corners.at(8)));
            world2d.push_back(Point2f(450,50));
            world3d.push_back(Point3f(450,50,0));
            img.push_back(Point2f(corners.at(45)));
            world2d.push_back(Point2f(50,50));
            world3d.push_back(Point3f(50,50,0));
            img.push_back(Point2f(corners.at(53)));

            // Compute H explicitly through matricial operations
            if (MATRIX){
                // Find homography matrix
                // H:world->image; H-1:image->world
                Mat H = findHomography(world2d, img, CV_RANSAC);

                // Compute Rotation Matrix (R) and Camera Pose (C)
                Mat rt = K.inv()*H;

                // Rotation matrix
                Mat R = Mat::eye(3, 3, CV_64FC1);
                Mat r1 = rt.col(0);
                Mat r2 = rt.col(1);
                Mat r3 = r1.cross(r2);
                r2 = -r1.cross(r3);
                R.col(0) = r1/norm(r1);
                R.col(1) = r2/norm(r2);
                R.col(2) = r3/norm(r3);

                // Traslation vector t
                Mat t = rt.col(2)/norm(rt.col(0));

                // Camera position
                Mat C = -R.t()*t;

                // Camera Matrix (M)
                double m[] = {1,0,0,-C.at<double>(0),0,1,0,-C.at<double>(1),0,0,1,-C.at<double>(2)};
                Mat M = Mat(3, 4, CV_64FC1, m);
                M = K*R*M;

                // Paint 3d points in image
                Mat point3d1 = Mat(4, 1, CV_64FC1, (double[]){50,50,0,1});
                Mat point3d2 = Mat(4, 1, CV_64FC1, (double[]){450,50,0,1});
                Mat point3d3 = Mat(4, 1, CV_64FC1, (double[]){50,300,0,1});
                Mat point3d4 = Mat(4, 1, CV_64FC1, (double[]){450,300,0,1});

                Mat point2d1 = M*point3d1;
                Mat point2d2 = M*point3d2;
                Mat point2d3 = M*point3d3;
                Mat point2d4 = M*point3d4;

                // Pintamos los points
                circle(frame, Point(point2d1.at<double>(0)/point2d1.at<double>(2), point2d1.at<double>(1)/point2d1.at<double>(2)), 10, cvScalar(255,0,0));
                circle(frame, Point(point2d2.at<double>(0)/point2d2.at<double>(2), point2d2.at<double>(1)/point2d2.at<double>(2)), 10, cvScalar(255,0,0));
                circle(frame, Point(point2d3.at<double>(0)/point2d3.at<double>(2), point2d3.at<double>(1)/point2d3.at<double>(2)), 10, cvScalar(255,0,0));
                circle(frame, Point(point2d4.at<double>(0)/point2d4.at<double>(2), point2d4.at<double>(1)/point2d4.at<double>(2)), 10, cvScalar(255,0,0));

                Mat point3d5 = Mat(4, 1, CV_64FC1, (double[]){200,150,0,1});
                Mat point3d6 = Mat(4, 1, CV_64FC1, (double[]){300,150,0,1});
                Mat point3d7 = Mat(4, 1, CV_64FC1, (double[]){200,50,0,1});
                Mat point3d8 = Mat(4, 1, CV_64FC1, (double[]){300,50,0,1});
                Mat point3d9 = Mat(4, 1, CV_64FC1, (double[]){200,150,-100,1});
                Mat point3d10 = Mat(4, 1, CV_64FC1, (double[]){300,150,-100,1});
                Mat point3d11 = Mat(4, 1, CV_64FC1, (double[]){200,50,-100,1});
                Mat point3d12 = Mat(4, 1, CV_64FC1, (double[]){300,50,-100,1});

                Mat point2d5 = M*point3d5;
                Mat point2d6 = M*point3d6;
                Mat point2d7 = M*point3d7;
                Mat point2d8 = M*point3d8;
                Mat point2d9 = M*point3d9;
                Mat point2d10 = M*point3d10;
                Mat point2d11 = M*point3d11;
                Mat point2d12 = M*point3d12;

                line(frame, Point(point2d5.at<double>(0)/point2d5.at<double>(2), point2d5.at<double>(1)/point2d5.at<double>(2)),
                     Point(point2d6.at<double>(0)/point2d6.at<double>(2), point2d6.at<double>(1)/point2d6.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d5.at<double>(0)/point2d5.at<double>(2), point2d5.at<double>(1)/point2d5.at<double>(2)),
                     Point(point2d7.at<double>(0)/point2d7.at<double>(2), point2d7.at<double>(1)/point2d7.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d7.at<double>(0)/point2d7.at<double>(2), point2d7.at<double>(1)/point2d7.at<double>(2)),
                     Point(point2d8.at<double>(0)/point2d8.at<double>(2), point2d8.at<double>(1)/point2d8.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d6.at<double>(0)/point2d6.at<double>(2), point2d6.at<double>(1)/point2d6.at<double>(2)),
                     Point(point2d8.at<double>(0)/point2d8.at<double>(2), point2d8.at<double>(1)/point2d8.at<double>(2)), cvScalar(255,0,0));

                line(frame, Point(point2d9.at<double>(0)/point2d9.at<double>(2), point2d9.at<double>(1)/point2d9.at<double>(2)),
                     Point(point2d10.at<double>(0)/point2d10.at<double>(2), point2d10.at<double>(1)/point2d10.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d9.at<double>(0)/point2d9.at<double>(2), point2d9.at<double>(1)/point2d9.at<double>(2)),
                     Point(point2d11.at<double>(0)/point2d11.at<double>(2), point2d11.at<double>(1)/point2d11.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d11.at<double>(0)/point2d11.at<double>(2), point2d11.at<double>(1)/point2d11.at<double>(2)),
                     Point(point2d12.at<double>(0)/point2d12.at<double>(2), point2d12.at<double>(1)/point2d12.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d12.at<double>(0)/point2d12.at<double>(2), point2d12.at<double>(1)/point2d12.at<double>(2)),
                     Point(point2d10.at<double>(0)/point2d10.at<double>(2), point2d10.at<double>(1)/point2d10.at<double>(2)), cvScalar(255,0,0));

                line(frame, Point(point2d5.at<double>(0)/point2d5.at<double>(2), point2d5.at<double>(1)/point2d5.at<double>(2)),
                     Point(point2d9.at<double>(0)/point2d9.at<double>(2), point2d9.at<double>(1)/point2d9.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d6.at<double>(0)/point2d6.at<double>(2), point2d6.at<double>(1)/point2d6.at<double>(2)),
                     Point(point2d10.at<double>(0)/point2d10.at<double>(2), point2d10.at<double>(1)/point2d10.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d7.at<double>(0)/point2d7.at<double>(2), point2d7.at<double>(1)/point2d7.at<double>(2)),
                     Point(point2d11.at<double>(0)/point2d11.at<double>(2), point2d11.at<double>(1)/point2d11.at<double>(2)), cvScalar(255,0,0));
                line(frame, Point(point2d8.at<double>(0)/point2d8.at<double>(2), point2d8.at<double>(1)/point2d8.at<double>(2)),
                     Point(point2d12.at<double>(0)/point2d12.at<double>(2), point2d12.at<double>(1)/point2d12.at<double>(2)), cvScalar(255,0,0));
            }

            /* SolvePNP method */
            if (SOLVEPNP){
                // Distortion coefficients
                Mat distCoeffs(4,1,DataType<double>::type);
                distCoeffs.at<double>(0) = 0;
                distCoeffs.at<double>(1) = 0;
                distCoeffs.at<double>(2) = 0;
                distCoeffs.at<double>(3) = 0;

                // R matrix and t vector
                Mat rvec(3,1,DataType<double>::type);
                Mat tvec(3,1,DataType<double>::type);
                Mat R;

                // Compute R matrix
                solvePnP(world3d, img, K, distCoeffs, rvec, tvec);
                Rodrigues(rvec, R);

                // Computing world coordinates from image points
                vector<Point3f> points;
                points.push_back(Point3f(200,150,0));
                points.push_back(Point3f(300,150,0));
                points.push_back(Point3f(200,50,0));
                points.push_back(Point3f(300,50,0));
                points.push_back(Point3f(200,150,-100));
                points.push_back(Point3f(300,150,-100));
                points.push_back(Point3f(200,50,-100));
                points.push_back(Point3f(300,50,-100));
                vector<Point2f> projectedPoints;
                projectPoints(points, rvec, tvec, K, distCoeffs, projectedPoints);

                // Draw a 3d cube in chessboard
                line(frame, projectedPoints.at(0), projectedPoints.at(1), cvScalar(0,0,255));
                line(frame, projectedPoints.at(0), projectedPoints.at(2), cvScalar(0,0,255));
                line(frame, projectedPoints.at(2), projectedPoints.at(3), cvScalar(0,0,255));
                line(frame, projectedPoints.at(1), projectedPoints.at(3), cvScalar(0,0,255));

                line(frame, projectedPoints.at(4), projectedPoints.at(5), cvScalar(0,0,255));
                line(frame, projectedPoints.at(4), projectedPoints.at(6), cvScalar(0,0,255));
                line(frame, projectedPoints.at(6), projectedPoints.at(7), cvScalar(0,0,255));
                line(frame, projectedPoints.at(5), projectedPoints.at(7), cvScalar(0,0,255));

                line(frame, projectedPoints.at(0), projectedPoints.at(4), cvScalar(0,0,255));
                line(frame, projectedPoints.at(1), projectedPoints.at(5), cvScalar(0,0,255));
                line(frame, projectedPoints.at(2), projectedPoints.at(6), cvScalar(0,0,255));
                line(frame, projectedPoints.at(3), projectedPoints.at(7), cvScalar(0,0,255));
            }
        }

        // Show frame
        imshow("video", frame);

        if ((cvWaitKey(10) & 255) == 27) break;
    }

    // Release memory from Mat
    frame.release();

    return 0;
}
