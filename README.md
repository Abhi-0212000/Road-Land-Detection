# Road-Lane-Detection
<h2>Road-Lane-Detection using OpenCV, Numpy, Hough Transformations</h2>


__Process followed in order to implement:__
  1. Pre-process the image (Gaussian Blur)
  2. Creating a Colored road mask that allows white and yellow (typical road lane colors) 
  3. Converting Colored road mask to grayscale
  4. Detecting edges of grayscale image. (Canny Edge detection)
  5. Identifying ROI (Region of Interest) and creating masked_canny_frame using bitwise_and
  6. Detecting the lines using Hough Transform
  7. Optimizing all the lines detected into 2 final lines (left and right)
     - using numpy.polyfit to get the 1st-order straight lines for every line detected by Hough Transform
     - splitting detected lines into left and right lines based on slope.
     - Averaging all the slopes, intercepts into single slope, intercept (for both right and left lines)
     - Calculating pixel coordinates from above slope, intercept in order to represent line for certain length.
  9. Drawing the lines, and optimized region to drive(using cv2.fillPolly) on a mask.
  10. And blending the mask with the original image (using cv2.addWeighted).

Also, implemented functionality to use the optimized lines from the previously detected frames. It is useful in situations like:
  1. If Hough Transform is not able to detect the lines in the current frame.
  2. If there is a discontinuity in road lane paint for long distances.

you can download the video used for implementing code from <a href="https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project/code)https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project/code" target='_blank'>here</a>
