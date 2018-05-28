#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>

/***********************************************************************
  This is the only file you need to change for your assignment.  The
  other files control the UI (in case you want to make changes.)
************************************************************************/

// The first six functions provide example code to help get you started

// Make sure we don't over or under saturate
QRgb restrictColor(int r, int g, int b)
{
    return qRgb(max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)));
}

// Normalize the values of the kernel to sum-to-one
void NormalizeKernel(double *kernel, int kernelWidth, int kernelHeight)
{
    double denom = 0.000001; int i;
    for(i=0; i<kernelWidth*kernelHeight; i++)
        denom += kernel[i];
    for(i=0; i<kernelWidth*kernelHeight; i++)
        kernel[i] /= denom;
}

// Convert an image to grayscale
void MainWindow::BlackWhiteImage(QImage *image)
{
    for(int r=0;r<image->height();r++)
    {
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
    }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int noiseMag = mag;
    noiseMag *= 2;

    for(int r=0;r<image->height();r++)
    {
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            int red = qRed(pixel);
            int green = qGreen(pixel);
            int blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;
                red += noise;
                green += noise;
                blue += noise;
            }
            image->setPixel(c, r, restrictColor(red,green,blue));
        }
    }
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it is not.
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    if(radius == 0)
        return;

    int r, c, rd, cd, i;
    int size = 2*radius + 1; // This is the size of the kernel

    int w = image->width();
    int h = image->height();

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    QImage buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute kernel to convolve with the image.
    double *kernel = new double [size*size];

    for(i=0;i<size*size;i++)
    {
        kernel[i] = 1.0;
    }
    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // For each pixel in the image...
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];
            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     QRgb pixel = buffer.pixel(c + cd + radius, r + rd + radius);
                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
            // Store the pixel in the image to be returned after rounding the value to the nearest integer
            image->setPixel(c, r, restrictColor((int)(floor(rgb[0]+0.5)),(int)(floor(rgb[1]+0.5)),(int)(floor(rgb[2]+0.5))));
        }
    }
    // Clean up.
    delete[] kernel;
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    int w = image.width();
    int h = image.height();
    QImage buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(int r=0;r<h/2;r++)
        for(int c=0;c<w/2;c++)
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
}

/********** TASK 1 **********/

// Convolve the image with the kernel
void Convolution(QImage *image, double *kernel, int kernelWidth, int kernelHeight, bool add)
{
    //Add your code here.
}

/********** TASK 2 **********/

// Apply the 2-D Gaussian kernel on an image to blur it
void MainWindow::GaussianBlurImage(QImage *image, double sigma)
{
    // Add your code here.  Look at MeanBlurImage to get yourself started.
}

/********** TASK 3 **********/

// Perform the Gaussian Blur first in the horizontal direction and then in the vertical direction
void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
    // Add your code here. Done right, you should be able to copy most of the code from GaussianBlurImage and call the Convolution function twice.
}

/********** TASK 4 (a) **********/

// Compute the First derivative of an image along the horizontal direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_x(QImage *image, double sigma)
{
    // Add your code here.
}

/********** TASK 4 (b) **********/

// Compute the First derivative of an image along the vertical direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_y(QImage *image, double sigma)
{
    //Add your code here.
}

/********** TASK 4 (c) **********/

// Compute the Second derivative of an image using the Laplacian operator and then apply Gaussian blur
void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    // Add your code here.
}

/********** TASK 5 **********/

// Sharpen an image by subtracting the image's second derivative from the original image
void MainWindow::SharpenImage(QImage *image, double sigma, double alpha)
{
    // Add your code here.  It's probably easiest to call SecondDerivImage as a helper function.
}

/********** TASK 6 **********/

// Display the magnitude and orientation of the edges in an image using the Sobel operator in both X and Y directions
void MainWindow::SobelImage(QImage *image)
{
    // Add your code here.

    /***********************************************************************
    When displaying the orientation image I recommend the following:

    double mag; // magnitude of the gradient
    double orien; // orientation of the gradient

    double red = (sin(orien) + 1.0)/2.0;
    double green = (cos(orien) + 1.0)/2.0;
    double blue = 1.0 - red - green;

    red *= mag*4.0;
    green *= mag*4.0;
    blue *= mag*4.0;

    // Make sure the pixel values range from 0 to 255
    image->setPixel(c, r, restrictColor( (int) (red), (int) (green), (int) (blue)));

    ************************************************************************/
}

/********** TASK 7 **********/

// Compute the RGB values at a given point in an image using bilinear interpolation.
void MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    // Add your code here.  Return the RGB values for the pixel at location (x,y) in double rgb[3].
}

/*******************************************************************************
 Here is the code provided for rotating an image.  I assume orien is in degrees.
********************************************************************************/

// Rotating an image by "orien" degrees
void MainWindow::RotateImage(QImage *image, double orien)
{
    int h = image->height();
    int w = image->width();
    double radians = -2.0*3.141*orien/360.0;

    QImage buffer = image->copy();
    QRgb pixel = qRgb(0, 0, 0);
    image->fill(pixel);

    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double rgb[3], x0, y0, x1, y1;

            // Rotate around the center of the image
            x0 = (double) (c - w/2);
            y0 = (double) (r - h/2);

            // Rotate using rotation matrix
            x1 = x0*cos(radians) - y0*sin(radians);
            y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (w/2);
            y1 += (double) (h/2);

            BilinearInterpolation(&buffer, x1, y1, rgb);

            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }
}

/********** TASK 8 **********/

// Find the peaks of the edge responses perpendicular to the edges
void MainWindow::FindPeaksImage(QImage *image, double thres)
{
    // Add your code here.
}

/********** TASK 9 (a) **********/

// Perform K-means clustering on a color image using random seeds
void MainWindow::RandomSeedImage(QImage *image, int num_clusters)
{
     // Add your code here
}

/********** TASK 9 (b) **********/

// Perform K-means clustering on a color image using seeds from the image itself
void MainWindow::PixelSeedImage(QImage *image, int num_clusters)
{
    // Add your code here
}

/********** EXTRA CREDIT TASKS **********/

// Perform K-means clustering on a color image using the color histogram
void MainWindow::HistogramSeedImage(QImage *image, int num_clusters)
{
    // Add your code here
}

// Apply the median filter on a noisy image to remove the noise
void MainWindow::MedianImage(QImage *image, int radius)
{
    // Add your code here
}

// Apply Bilater filter on an image
void MainWindow::BilateralImage(QImage *image, double sigmaS, double sigmaI)
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

// Perform the Hough transform
void MainWindow::HoughImage(QImage *image)
{
    // Add your code here
}

// Perform smart K-means clustering
void MainWindow::SmartKMeans(QImage *image)
{
    // Add your code here
}
