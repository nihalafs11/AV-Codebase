#@author Nihal Afsal
# An image processing pipeline that uses OpenCV functions to manipulate a image and identify the lane lines.
import cv2
import numpy as np

def process_image(image_path):
    # Load image the image from user path
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Image conversion to the HLS color space 
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Set the saturation channel threshold 
    lower_saturation_threshold = 100
    upper_saturation_threshold = 400

    # Obtain the asphault's binary image. 
    binary_image = np.zeros_like(hls_image[:,:,2])
    binary_image[(hls_image[:,:,2] > lower_saturation_threshold) & (hls_image[:,:,2] <= upper_saturation_threshold)] = 255

    # Kernel for morphological transformation definition 
    morph_kernel = np.ones((5,5),np.uint8)

    # To remove noise, use morphological transformation. 
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, morph_kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morph_kernel)

    # Utilize Canny edge detection 
    edges = cv2.Canny(binary_image, 50, 150)

    # Set the parameters for the Hough transform. 
    hough_rho = 1
    hough_theta = np.pi/180
    hough_threshold = 20
    hough_min_line_len = 10
    hough_max_line_gap = 25

    # Run Hough on edge detected
    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, np.array([]), hough_min_line_len, hough_max_line_gap) 

    # Create an blank image to draw the lines on lanes
    line_image = np.zeros_like(image)

    # On the blank image, draw the lines when lane is found
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255, 0, 127),10)

    # Resize the line picture to match the scale of the source image. 
    line_image = cv2.resize(line_image, (image.shape[1], image.shape[0]))

    # Using an alpha of 0.8, merge the original image and the line image. 
    result = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)

    # Show the finished photo 
    cv2.imshow("Lane Lines Image Example:", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Call image from user’s directory
process_image("example.jpg")
