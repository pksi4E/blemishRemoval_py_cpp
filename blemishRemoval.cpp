#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

constexpr uchar ESC = 27;
constexpr uchar BACKSPACE = 8;

cv::String windowName{ "Blemish Removal" };
cv::Mat gImg;
cv::Mat gImgClone;
cv::Mat gImgBlur;
cv::Mat gImgGray;
cv::Mat gImgWithBorder;
cv::Mat gImgGrayWithBorder;

void onMouse(int event, int x, int y, int flags, void *userdata);

// Program works in a specific way, that is:
// 1) first, I'm performing initial blurring to the original image
//      for the purpose of finding blemishes
// 2) next, by clicking on the blemish, I'm enlarging the original
//      image with border, so I can choose the blemish near the corner,
//      and perform later searching for a patch also near the corner
// 3) next, I'm applying Sobel derivative on the box with the blemish
// 4) next, I'm detecting the blemish inside this box, because I don't know
//      how big is the blemish inside the box or if the user clicked exactly
//      on the center
// 5) next, when the blemish is detected enclosed by the proper circle,
//      I'm saving this circle as the mask, so I can use it to find the
//      mean value of the pixels around the blemish, and to paste in the next steps
//      the chosen patch not as a circle but with this circular mask
// 6) next, I'm searching for a patch around the blemish box in k^2 = 25 points,
//      but how I decide what patch to choose? I'm selecting the patch which has
//      the mean value of pixels is the nearest to the value of pixels around the blemish
//      in the selected blemish box
// 7) at the end, I'm performing saemless cloning on the image with border, and
//       copying this image but without border to the original image gImg.
int main()
{
    // loading an image
    cv::String fileName { "blemish.png" };
    gImg = cv::imread(fileName, cv::IMREAD_COLOR);
    if (gImg.empty()) {
        std::cout << "Error! Couldn't load an image!\n";
        return -1;
    }

    // initial modifications of the image
    cv::GaussianBlur(gImg, gImgBlur, cv::Size{5, 5}, 0, 0);
    cv::medianBlur(gImgBlur, gImgBlur, 3);
    cv::cvtColor(gImgBlur, gImgGray, cv::COLOR_BGR2GRAY);

    // displaying main window for mouse callback
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, onMouse);

    // --- Choosing blemish to remove.
    // ESC to exit the application.
    // BACKSPACE to undo one step.
    for (int keyPressed = 0; keyPressed != ESC; )
    {
        cv::imshow(windowName, gImg);

        keyPressed = cv::waitKey(1) & 0xFF;

        // the undo option
        if (keyPressed == BACKSPACE) {
            gImgClone.copyTo(gImg);
        }
    }

    //----------Ending------------//
    cv::destroyAllWindows();
    return 0;
}

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        // --- UNDO option
        // -- cloning the image for undo option
        gImgClone = gImg.clone();

        // the coordinates of the selected point
        cv::Point pt { x, y };

        // size of the square with the blemish
        constexpr uchar size { 35 }; // odd number
        constexpr uchar halfSize { (size - 1) / 2 };

        // //////////////////////////////////// //
        // --------- Adding border ------------ //
        // enlarging the image with border to be able to choose points near the border,
        //  and to enable patch searching around every point
        int bordSize{ size + halfSize };
        cv::copyMakeBorder(gImgGray, gImgGrayWithBorder, bordSize, bordSize, bordSize, bordSize,
                           cv::BORDER_REFLECT_101);
        cv::copyMakeBorder(gImg, gImgWithBorder, bordSize, bordSize, bordSize, bordSize,
                           cv::BORDER_REFLECT_101);

        // moving the (x,y), because now working on the image with border
        cv::Point pt2; // point on the image with border
        pt2 = cv::Point(x + bordSize, y + bordSize);

        cv::Mat blemishRect; // a small matrix to hold the blemish
        blemishRect = gImgGrayWithBorder(cv::Range(pt2.y - halfSize, pt2.y + halfSize + 1),
                          cv::Range(pt2.x - halfSize, pt2.x + halfSize + 1));

        // ///////////////////////////////////////////////// //
        // Computing the Sobel derivative of the blemish box //
        cv::Mat sobelx, sobely;
        cv::Mat blemishRectGradient;
        cv::Mat blemishRectThresh;
        int ksize = cv::FILTER_SCHARR;
        cv::Sobel(blemishRect, sobelx, CV_32F, 1, 0, ksize);
        cv::Sobel(blemishRect, sobely, CV_32F, 0, 1, ksize);
        cv::convertScaleAbs(sobelx, sobelx);
        cv::convertScaleAbs(sobely, sobely);
        cv::addWeighted(sobelx, 0.5, sobely, 0.5, 0, blemishRectGradient);
        cv::threshold(blemishRectGradient, blemishRectThresh, 50, 255, cv::THRESH_BINARY);

        // /////////////////////////////////////// //
        // ------- Detecting the blemish --------- //
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(blemishRectThresh, contours, hierarchy,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        int max_area = 0, max_id = 0;
        int area{};
        for (int i = 0; i < contours.size(); ++i) {
            area = cv::contourArea(contours[i]);
            if (max_area < area) {
                max_id = i;
                max_area = area;
            }
        }
        cv::drawContours(blemishRectThresh, contours, max_id, cv::Scalar{100, 0, 0}, 1);
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[max_id], center, radius);

        // mask enclosing the detected blemish to find the mean in the box
        cv::Mat mask1 {blemishRectThresh.size(), blemishRectThresh.depth(), cv::Scalar(0)};
        cv::circle(mask1, center, radius, cv::Scalar{255, 255, 255}, -1);

        // mask enclosing the detected blemish enlarged to perform seamless cloning
        cv::Mat mask2 {blemishRectThresh.size(), blemishRectThresh.depth(), cv::Scalar(0)};
        cv::circle(mask2, center, halfSize, cv::Scalar{255, 255, 255}, -1);

        // What is the mean of the area around the blemish?
        cv::Mat mask1Not;
        cv::bitwise_not(mask1, mask1Not);
        cv::Scalar meanBlemish = cv::mean(blemishRect, mask1Not);


        // ///////////////////////////////////////////////// //
        //-------------- Searching for a patch --------------//
        // We're searching for a patch in k^2 points around the blemish.
        double nearestMean{ 255 };
        double difference{};
        int k{ 5 };
        int move = size / ((k - 1) / 2.0);
        int start_x = pt2.x - size; // initial values of x in patch searching
        int start_y = pt2.y - size; // the same for y
        int patch_x{}, patch_y{}; // x and y value of the patch
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                int moved_x = start_x + j * move;
                int moved_y = start_y + i * move;
                if (i == 1 && j == 1) {
                    continue;
                }
                // searching for the square with the nearest mean
                //      to the mean around the blemish
                cv::Rect square{ cv::Point(moved_x - halfSize, moved_y - halfSize),
                            cv::Size(size, size) };
                cv::Scalar theMean = cv::mean(gImgGrayWithBorder(square), mask1);
                difference = std::abs(meanBlemish[0] - theMean[0]);
                if (difference < nearestMean) {
                    nearestMean = difference;
                    patch_x = moved_x - halfSize;
                    patch_y = moved_y - halfSize;
                }
            }
        }
        cv::Rect patchRect{ patch_x, patch_y, size, size };
        cv::Mat patch = gImgWithBorder(patchRect);

        // ////////////////////////////// //
        // ------ Seamless cloning ------ //
        cv::seamlessClone(patch, gImgWithBorder, mask2, pt2, gImgWithBorder, cv::NORMAL_CLONE);

        // Copying the modified image with border to the gImg without border //
        gImgWithBorder(cv::Range(bordSize, gImg.size().height + bordSize),
                       cv::Range(bordSize, gImg.size().width + bordSize)).copyTo(gImg);
    }
}
