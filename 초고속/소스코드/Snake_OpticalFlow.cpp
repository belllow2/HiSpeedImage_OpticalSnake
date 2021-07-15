//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/legacy/legacy.hpp>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//#include <iostream>
//#include <cstdio>
//#include <vector>
//#include <math.h>
//
//using namespace std;
//using namespace cv;
//cv::Mat image = imread("C:\\Users\\Jisu\\Desktop\\초고속\\프레임4\\frame_70.jpg", IMREAD_GRAYSCALE);
//cv::Mat image_ = imread("C:\\Users\\Jisu\\Desktop\\초고속\\프레임4\\frame_70.jpg", 1);
//Mat iframe = imread("C:\\Users\\Jisu\\Desktop\\초고속\\이진\\iframe_80.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//Mat bin = imread("C:\\Users\\Jisu\\Desktop\\초고속\\C0084_Moment.jpg", 0);
//cv::Point2f PtOld;
//static void help();
//int returnLargestContourIndex(vector<vector<Point> > contours);
//Mat image2,image_2, image3;
//const int MAX_CORNERS = 8000;
//vector<vector<Point>> contours;
//vector<vector<Point>> contours_i;
//vector<Point> l_contour;
//Mat mask(image.size(), image.type());
//int mouse_i = 0, file_cnt = 70, op_cnt = 0, file_cnt2 = 71;
//CvPoint2D32f* cornersA = new CvPoint2D32f[MAX_CORNERS];
//CvPoint2D32f* cornersB = new CvPoint2D32f[MAX_CORNERS];
//CvPoint* contour_points;
//vector<Point> largest_contour_i; 
//IplImage* imgC;
//IplImage* imgF;
//IplImage* imgE;
//IplImage* imgB;
//Mat imgD;
//int lng = 0, fl=0, length=0;
//void on_mouse(int event, int x, int y, int flags, void * param)
//{
//	switch (event)
//	{
//		case EVENT_LBUTTONDOWN:
//		{ 
//		
//			line(mask, Point(x, y), Point(x, y), Scalar(255, 255, 255), 5);
//			line(image, Point(x, y), Point(x, y), Scalar(0, 0, 255), 5);
//		
//			break;
//		}
//		case EVENT_MOUSEMOVE:
//		{	
//			if (flags & EVENT_FLAG_LBUTTON) {
//				for(int i=0;i<40;i++)
//					line(image, Point(x, y), Point(x, y), Scalar(0, 0, 255), 5);
//				for (int i = 0; i<40; i++)
//					line(mask, Point(x, y), Point(x, y), Scalar(255, 255, 255), 5);
//			
//				
//				imshow("src", image);
//				
//			}
//			break;
//		}
//		case EVENT_LBUTTONUP: {
//			line(mask, Point(x, y), Point(x, y), Scalar(255, 255, 255), 5);
//		}
//	}
//}
//float alpha = 1.5f, beta = 1.5f, gama =0.5f;
//int main(int argc, char** argv)
//{
//	vector<Vec4i> hierarchy;
//	GaussianBlur(image, image, Size(13, 13), 3, 3);
//	resize(image, image, Size(1024, 768));
//	resize(image_, image_, Size(1024, 768));
//	resize(mask, mask, Size(1024, 768));
//	image.copyTo(image2);
//	image_.copyTo(image_2);
//	imshow("src", image);
//	setMouseCallback("src", on_mouse);
//	waitKey();
//	Mat binary_image(image.size(), image.type());
//	Mat binary_image_i(image.size(), image.type());
//	Mat test_i(image.size(), image.type());
//	threshold(mask, binary_image, 0, 255, THRESH_BINARY | THRESH_OTSU);
//	threshold(iframe, binary_image_i, 0, 255, THRESH_BINARY | THRESH_OTSU);
//
//	Mat binary_image_clone = binary_image.clone();
//	Mat binary_image_clone_i = binary_image_i.clone();
//
//	findContours(binary_image_clone, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//	//findContours(binary_image_clone_i, contours_i, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//
//
//	int largest_contour_idx = returnLargestContourIndex(contours);
//	vector<Point> largest_contour = contours[largest_contour_idx];
//
//	//int largest_contour_idx_i = returnLargestContourIndex(contours_i);
//	//vector<Point> largest_contour_i = contours_i[largest_contour_idx_i];
//	//for (int i = 0; i < largest_contour.size(); i++) {
//	//	line(test_i, largest_contour[i], largest_contour[i], Scalar(255, 255, 255), 5);
//	//}
//
//	IplImage* img = &IplImage(image2);
//	contour_points = new CvPoint[largest_contour.size()];
//	for (unsigned int i = 0; i < largest_contour.size(); ++i)
//		contour_points[i] = largest_contour[i];
//
//
//
//for (;;) {
//	char c = (char)waitKey(0);
//	switch (c)
//	{
//	case 'g':
//	{
//		if (op_cnt == 0) {
//			image_2.copyTo(image3);
//			cvSnakeImage(img, contour_points, static_cast<int>(largest_contour.size()), (float*)&alpha,
//				(float*)&beta, (float*)&gama, 1, cvSize(15, 15), TermCriteria(CV_TERMCRIT_ITER, 1, 0.5), 1);
//
//			for (int i = 0; i < largest_contour.size() - 1; i++) {
//				cv::line(image3, contour_points[i], contour_points[i + 1], Scalar(0, 0, 255), 5);
//			}
//			cv::line(image3, contour_points[largest_contour.size() - 1], contour_points[0], Scalar(0, 0, 255), 5);
//
//			imshow("image2", image3);
//		}
//		else {
//			
//			imgD.copyTo(image3);
//			cvSnakeImage(imgB, contour_points, length, (float*)&alpha,
//				(float*)&beta, (float*)&gama, 1, cvSize(15, 15), TermCriteria(CV_TERMCRIT_ITER, 1, 0.5), 1);
//
//			for (int i = 0; i < length - 1; i++) {
//				cv::line(image3, contour_points[i], contour_points[i + 1], Scalar(0, 0, 255), 5);
//			}
//			cv::line(image3, contour_points[length - 1], contour_points[0], Scalar(0, 0, 255), 5);
//
//			imshow("image2", image3);
//			
//		}
//		break;
//	}
//	case 'h':
//	{
//		Mat test_f(image.size(), image.type());
//		string s = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\프레임4\\frame_%d.jpg", file_cnt);
//		IplImage* imgA = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 1);
//		IplImage* img = cvLoadImage(s.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//		cvResize(img, imgA, CV_INTER_CUBIC);
//		file_cnt = file_cnt + 1;
//		string s2 = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\프레임4\\frame_%d.jpg", file_cnt);
//		imgB = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 1);
//		img = cvLoadImage(s2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//		//imgB = cvLoadImage(s2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//		cvResize(img, imgB, CV_INTER_CUBIC);
//		Mat f = imread(s, 0);
//		Mat f2 = imread(s2, 0);
//		Mat element(5, 5, CV_8U, Scalar(1));
//		resize(f, f, Size(1024, 768)); resize(f2, f2, Size(1024, 768));
//		//imshow("rqw", f);
//	
//		//imgB = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 1);
//		//IplImage* img = cvLoadImage(s.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//		//cvResize(img, imgB, CV_INTER_CUBIC);
//
//		Mat test_i(f.size(), f.type());
//		resize(bin, bin, Size(1024, 768));
//		absdiff(f2, f, f2);
//		absdiff(f, bin, f);
//		imshow("Diffimg", f2);
//		GaussianBlur(f, f, Size(9, 9),9,9);
//		//threshold(f2, f2, 3, 255, THRESH_BINARY);
//		threshold(f, f, 50, 255, THRESH_BINARY);
//		//adaptiveThreshold(f, f, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 27, 1);
//		//morphologyEx(f2, f2, MORPH_OPEN, element);
//		//morphologyEx(f2, f2, MORPH_CLOSE, element);
//		morphologyEx(f, f, MORPH_OPEN, element);
//		morphologyEx(f, f, MORPH_CLOSE, element);
//		//erode(f2, f2, Mat()); erode(f2, f2, Mat()); erode(f2, f2, Mat()); erode(f2, f2, Mat()); erode(f2, f2, Mat());
//		//dilate(f, f, Mat()); dilate(f, f, Mat()); //dilate(f2, f2, Mat()); dilate(f2, f2, Mat()); dilate(f2, f2, Mat());
//		erode(f, f, Mat()); erode(f, f, Mat());
//		//GaussianBlur(f, f, Size(13, 13), 3, 3);
//		//adaptiveThreshold(f, f, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 8);
//		//imshow("asd", f);
//		//Canny(f, f, 40, 80);
//		string s3 = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\이진4\\iframe_%d.jpg", file_cnt2);
//		file_cnt2 = file_cnt2 + 1;
//
//
//		Mat f3 = imread(s3, 0);
//
//		resize(f3, f3, Size(1024, 768));
//		imshow("BGSimg", f3);
//		//GaussianBlur(f2, f2, Size(15, 15), 15, 15);
//	/*	threshold(f3, f3, 2, 255, THRESH_BINARY);
//		morphologyEx(f3, f3, MORPH_OPEN, element);
//		morphologyEx(f3, f3, MORPH_CLOSE, element);
//		dilate(f3, f3, Mat()); dilate(f3, f3, Mat()); dilate(f3, f3, Mat()); dilate(f3, f3, Mat()); dilate(f3, f3, Mat());
//		resize(f3, f3, Size(1024, 768));*/
//
//
//
//		findContours(f, contours_i, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//
//		int largest_contour_idx = returnLargestContourIndex(contours_i);
//		largest_contour_i = contours_i[largest_contour_idx];
//		length = largest_contour_i.size();
//		for (int i = 0; i < largest_contour_i.size(); i++) {
//			line(test_i, largest_contour_i[i], largest_contour_i[i], Scalar(255, 255, 255), 3);
//		}
//
//		imshow("test", test_i);
//
//	
//
//		Mat a, b;
//		a = cvLoadImage(s.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//
//		cvSmooth(imgA, imgA, CV_GAUSSIAN, 13, 13, 3, 3);
//		cvSmooth(imgB, imgB, CV_GAUSSIAN, 13, 13, 3, 3);
//		
//		imgD = cvLoadImage(s2.c_str(), CV_LOAD_IMAGE_UNCHANGED);
//		resize(imgD, imgD, Size(1024, 768));
//
//		CvSize img_sz = cvGetSize(imgA);
//		int win_size = 90;
//
//		imgC = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 3);
//		imgF = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 3);
//		imgE = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 3);
//		img = cvLoadImage(s2.c_str(), CV_LOAD_IMAGE_UNCHANGED);
//		cvResize(img, imgC, CV_INTER_CUBIC);
//		cvResize(img, imgF, CV_INTER_CUBIC);
//		cvResize(img, imgE, CV_INTER_CUBIC);
//		//imgC = cvLoadImage(s2.c_str(), CV_LOAD_IMAGE_UNCHANGED);
//		
//		//추적할 특징을 검출한다
//		IplImage* eig_image = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
//		IplImage* tmp_image = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
//
//		int corner_count = MAX_CORNERS;
//	
//
//		
//
//		//이미지에서 코너를 추출함
//
//		//cvGoodFeaturesToTrack(imgA, eig_image, tmp_image,cornersA, &corner_count, 0.01, 5.0, 0, 3, 0, 0.04);
//
//		//서브픽셀을 검출하여 정확한 서브픽셀 위치를 산출해냄
//
//		//cvFindCornerSubPix(imgA, cornersA, corner_count, cvSize(win_size, win_size), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
//
//
//		const int cnt = largest_contour.size();
//		//루카스-카나데 알고리즘
//
//		char feature_found[MAX_CORNERS];
//
//		float feature_errors[MAX_CORNERS];
//
//		CvSize pyr_sz = cvSize(imgA->width + 8, imgB->height / 3);
//
//		IplImage* pyrA = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
//		IplImage* pyrB = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
//
//
//		if (op_cnt == 1 || op_cnt == 0) {
//			for (int i = 0; i < largest_contour.size(); i++) {
//				cornersA[i] = cvPoint2D32f(contour_points[i].x, contour_points[i].y);
//			}
//		}
//	
//		//추출한 코너(cornerA)를 추적함 -> 이동한 점들의 위치는 cornerB에 저장된다.
//
//		cvCalcOpticalFlowPyrLK(imgA, imgB, pyrA, pyrB, cornersA, cornersB, corner_count,cvSize(win_size, win_size), 100, feature_found, feature_errors,
//			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3), 0);
//
//		
//
//		for (int i = 0; i<corner_count; i++) {
//			if (feature_found[i] == 0 || feature_errors[i] > 100) {
//				//feature_found[i]값이 0이 리턴이 되면 대응점을 발견하지 못함
//				//feature_errors[i] 현재 프레임과 이전프레임 사이의 거리가 550이 넘으면 예외로 처리
//				printf("Error is %f\n", feature_errors[i]);
//				cornersB[i].x = -1; cornersB[i].y = -1;
//				continue;
//
//			}
//
//			printf("Got it\n");
//			CvPoint p0 = cvPoint(cvRound(cornersA[i].x), cvRound(cornersA[i].y));
//			CvPoint p1 = cvPoint(cvRound(cornersB[i].x), cvRound(cornersB[i].y));
//			if(i % 20 == 0)
//			cvLine(imgC, p0, p1, CV_RGB(0, 255, 0), 2);
//
//			cvLine(imgC, p0, p0, CV_RGB(255, 255, 255), 5);
//			cvLine(imgC, p1, p1, CV_RGB(255, 0, 0), 5);
//			//cvLine(imgF, p0, p1, CV_RGB(0, 255, 0), 5);
//			//line(test_f, p1, p1, Scalar(255, 255, 255), 5);
//
//		}
//		for (int i = 0; i<corner_count; i++) {
//			if (feature_found[i] == 0 || feature_errors[i] > 100) {
//				//feature_found[i]값이 0이 리턴이 되면 대응점을 발견하지 못함
//				//feature_errors[i] 현재 프레임과 이전프레임 사이의 거리가 550이 넘으면 예외로 처리
//				printf("Error is %f\n", feature_errors[i]);
//				cornersB[i].x = -1; cornersB[i].y = -1;
//				continue;
//
//			}
//			CvPoint p0 = cvPoint(cvRound(cornersA[i].x), cvRound(cornersA[i].y));
//			CvPoint p1 = cvPoint(cvRound(cornersB[i].x), cvRound(cornersB[i].y));
//
//			if(i % 20 == 0)
//			cvLine(imgF, p0, p1, CV_RGB(0, 255, 0), 5);
//			//line(test_f, p1, p1, Scalar(255, 255, 255), 5);
//
//		}
//		for (int i = 0; i < corner_count; i++)
//			cornersA[i] = cornersB[i];
//
//		cvShowImage("Lkpyr_OpticalFlow", imgC);
//		cvShowImage("Lkpyr_OpticalFlow_2", imgF);
//		op_cnt++;
//		
//	
//		for (int i = 0; i < corner_count; i++) {
//
//			if (cornersB[i].x > 1900 || cornersB[i].y > 1060 )
//			{
//				cornersB[i].x = -1; cornersB[i].y = -1;
//			}
//			else if (cornersB[i].x < 20 || cornersB[i].y < 20)
//			{
//				cornersB[i].x = -1; cornersB[i].y = -1;
//			}
//			else {
//				lng++;
//			}
//				
//		}	
//		contour_points = new CvPoint[lng];
//		length = lng;
//		lng = 0;
//		for (int i = 0; i < corner_count; i++) {
//
//			if (cornersB[i].x != -1 && cornersB[i].y != -1)
//			{
//				contour_points[fl] = cvPointFrom32f(cornersB[i]);
//				fl++;
//			}
//		}
//		fl = 0;
//		/*findContours(test_f, contours_i, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//		
//		int largest_contour_idx_i = returnLargestContourIndex(contours_i);
//		largest_contour_i = contours_i[largest_contour_idx_i];
//
//		for(int i=0;i <largest_contour_i.size();i++)
//			line(test_i, largest_contour_i[i], largest_contour_i[i], Scalar(255, 255, 255), 5);
//		contour_points = new CvPoint[largest_contour_i.size()];
//		for (unsigned int i = 0; i < largest_contour_i.size(); ++i)
//			contour_points[i] = largest_contour_i[i];*/
//
//		break;
//	}
//	case 'n':
//	{
//		file_cnt = file_cnt -1;
//		string s = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\프레임4\\frame_%d.jpg", file_cnt);
//		file_cnt = file_cnt + 1;
//		string s2 = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\프레임4\\frame_%d.jpg", file_cnt);
//		file_cnt = file_cnt +1;
//		Mat element(5, 5, CV_8U, Scalar(1));
//		imgD = cvLoadImage(s.c_str(), CV_LOAD_IMAGE_UNCHANGED);
//		imgB = cvLoadImage(s.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//		//file_cnt2 = file_cnt2 + 1;
//		Mat f = imread(s, 0);
//		Mat f2 = imread(s2, 0);
//		resize(f, f, Size(1024, 768)); resize(f2, f2, Size(1024, 768));
//		//imshow("rqw", f);
//		resize(imgD, imgD, Size(1024, 768));
//		imgB = cvCreateImage(cvSize(1024, 768), IPL_DEPTH_8U, 1);
//		IplImage* img = cvLoadImage(s.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//		cvResize(img, imgB, CV_INTER_CUBIC);
//
//		Mat test_i(f.size(), f.type());
//		resize(bin, bin, Size(1024, 768));
//		absdiff(f2,f, f2);
//		absdiff(f, bin, f);
//		imshow("Diffimg", f2);
//		//absdiff(f, bin, f);
//		GaussianBlur(f, f, Size(9, 9),9,9);
//		//threshold(f2, f2, 3, 255, THRESH_BINARY);
//		threshold(f, f, 50, 255, THRESH_BINARY);
//		//imshow("asdasd", f);
//		//adaptiveThreshold(f, f, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 27, 1);
//		//morphologyEx(f2, f2, MORPH_OPEN, element);
//		//morphologyEx(f2, f2, MORPH_CLOSE, element);
//		morphologyEx(f, f, MORPH_OPEN, element);
//		morphologyEx(f, f, MORPH_CLOSE, element);
//	//erode(f2, f2, Mat()); erode(f2, f2, Mat()); erode(f2, f2, Mat()); erode(f2, f2, Mat());
//		//dilate(f, f, Mat()); dilate(f, f, Mat());  //dilate(f, f, Mat()); dilate(f2, f2, Mat()); dilate(f2, f2, Mat());
//		erode(f, f, Mat()); erode(f, f, Mat());
//
//		//GaussianBlur(f, f, Size(13, 13), 3, 3);
//		//adaptiveThreshold(f, f, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 8);
//		//imshow("asd", f);
//		//Canny(f, f, 40, 80);
//
//		string s3 = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\이진4\\iframe_%d.jpg", file_cnt2);
//		file_cnt2 = file_cnt2 + 1;
//
//
//		Mat f3 = imread(s3, 0);
//
//		resize(f3, f3, Size(1024, 768));
//		imshow("BGSimg", f3);
//		//GaussianBlur(f2, f2, Size(15, 15), 15, 15);
//	/*	threshold(f3, f3, 2, 255, THRESH_BINARY);
//		morphologyEx(f3, f3, MORPH_OPEN, element);
//		morphologyEx(f3, f3, MORPH_CLOSE, element);
//		dilate(f3, f3, Mat()); dilate(f3, f3, Mat()); dilate(f3, f3, Mat()); dilate(f3, f3, Mat()); dilate(f3, f3, Mat());
//		resize(f3, f3, Size(1024, 768));*/
//		
//		//findContours(f2, contours_i, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//		findContours(f, contours_i, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//			
//		int largest_contour_idx = returnLargestContourIndex(contours_i);
//		largest_contour_i = contours_i[largest_contour_idx];
//		length = largest_contour_i.size();
//		for (int i = 0; i < largest_contour_i.size(); i++) {
//			line(test_i, largest_contour_i[i], largest_contour_i[i], Scalar(255, 255, 255), 3);
//		}
//
//		imshow("test", test_i);
//		delete contour_points;
//		contour_points = new CvPoint[length];
//		for (unsigned int i = 0; i < largest_contour_i.size(); ++i)
//			contour_points[i] = largest_contour_i[i];
//
//		op_cnt = 1;
//
//		break;
//	}
//	}
//}
//return 0;
//}
//
//
//int returnLargestContourIndex(vector<vector<Point>> contours)
//{
//	unsigned int max_contour_size = 0;
//	int max_contour_idx = -1;
//	for (unsigned int i = 0; i < contours.size(); ++i)
//	{
//		if (contours[i].size() > max_contour_size)
//		{
//				max_contour_size = static_cast<int>(contours[i].size());
//				max_contour_idx = i;
//		}
//	}
//	return max_contour_idx;
//}