#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

#include <opencv2/bgsegm.hpp>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture("C:\\Users\\Jisu\\Desktop\\초고속\\test_Trim_Trim.mp4");

	//create Background Subtractor objects
	Ptr<BackgroundSubtractorMOG2> pBackSub;
	
	pBackSub = createBackgroundSubtractorMOG2();
	int count = 0;
	Mat frame, fgMask;
	while (true) {
		capture >> frame;
		if (frame.empty())
			break;
		//update the background model
		pBackSub->apply(frame, fgMask);
		//get the frame number and write it on the current frame
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(0, 0, 2), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		string name = cv::format("C:\\Users\\Jisu\\Desktop\\초고속\\야외축구(그림자)\\iframe_%d.jpg", count);
				imwrite(name.c_str(), fgMask);
			count++;
		//get the input from the keyboard
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}
	return 0;
}