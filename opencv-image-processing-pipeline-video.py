#@author Nihal Afsal
# An image processing pipeline that uses OpenCV functions to manipulate a video and identify the lane lines.
import cv2
import numpy as np

def process_video(video_path):
    # Open the video file.
    video_capture = cv2.VideoCapture(video_path)

    # Loop over the frames of a video
    while True:
        # Check the current frame in the video
        success, current_frame = video_capture.read()

        # Stop the loop if the video ends.
        if not success:
            break

        # Frame conversion to HSV color space
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Define the HSV yellow color spectrum.
        yellow_lower = np.array([20, 20, 70])
        yellow_upper = np.array([50, 255, 255])

        # Define the HSV White color spectrum.
        white_lower = np.array([0, 0, 190])
        white_upper = np.array([180, 60, 255])

        # Set the HSV image threshold to only produce yellow and white colors.
        yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)

        # connect the two masks
        color_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Apply the mask to the frame
        filtered_frame = cv2.bitwise_and(current_frame, current_frame, mask=color_mask)

        # Use the mask to cover the frame.
        blur_frame = cv2.GaussianBlur(filtered_frame, (5, 5), 0)

        # Canny edge detection can be used to find edges.
        edges = cv2.Canny(blur_frame, 50, 150)

        # Make a mask that only picks the area of interest (ROI)
        roi_mask = np.zeros_like(edges)
        ignore_color = 255
        roi_vertices = np.array([[(0, current_frame.shape[0]), (450, 290), (490, 290), (current_frame.shape[1], current_frame.shape[0])]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, ignore_color)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # Apply the Hough transform to the ROI to find the lines.
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 15, np.array([]), minLineLength=40, maxLineGap=20)

        # On the original frame, draw lines.
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the edited frame.
        cv2.imshow("Lane Lines Video Example:", current_frame)

    # Show the finished video
    video_capture.release()
    cv2.destroyAllWindows()

# Call video from user’s directory
process_video("example.mp4")
