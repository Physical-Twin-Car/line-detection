import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineDetectionNode(Node):
    def __init__(self):
        super().__init__('line_detection_node')
        
        self.grayscale_subscription = self.create_subscription(Image, '/camera/image_grayscale', self.image_callback, 10)
        
        self.line_detection_publisher = self.create_publisher(Image, '/line_detection/output', 10)
        
        # conversie OpenCV <-> ROS images
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # conversie naar OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # image dimensies
        # _ wordt genegeerd, dit is aantal kleurkanalen. 
        height, width, _ = cv_image.shape
        
        # crop de area voor een kleiner image
        top = int(height * 0.4)   # Start cropping at 40% from the top
        bottom = int(height * 0.8)  # End cropping at 80% from the top
        left = int(width * 0.3)   # Start cropping at 30% from the left
        right = int(width * 0.7)  # End cropping at 70% from the left

        # focus op het midden
        cropped_image = cv_image[top:bottom, left:right]
        
        # Step 1: Convert to HLS color space and extract the L-channel
        hls = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]  # L channel (lightness)

        # Step 2: Apply thresholding to detect bright (white) lanes
        _, binary_output = cv2.threshold(l_channel, 225, 255, cv2.THRESH_BINARY)
        
        # Step 3: Detect edges using Canny edge detection
        edges = cv2.Canny(binary_output, 50, 150)

        # Step 4: Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

        # Step 5: Create a blank image to draw the lines
        line_image = np.zeros_like(cropped_image)

        if lines is not None:
            left_line_x = []
            right_line_x = []
            left_line_y = []
            right_line_y = []

            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                    if slope < 0:  # Left lane
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    elif slope > 0:  # Right lane
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])

            if left_line_x and right_line_x:
                # Get the average coordinates for left and right lines
                left_avg_x = int(np.mean(left_line_x))
                left_avg_y = int(np.mean(left_line_y))
                right_avg_x = int(np.mean(right_line_x))
                right_avg_y = int(np.mean(right_line_y))

                # Draw the polygon between the lines
                pts = np.array([[left_avg_x, 0], [right_avg_x, 0], 
                                [right_avg_x, line_image.shape[0]], 
                                [left_avg_x, line_image.shape[0]]], np.int32)
                pts = pts.reshape((-1, 1, 2))

                # Fill the polygon with green color
                cv2.fillPoly(line_image, [pts], (0, 255, 0))

                # Draw the left and right lines
                cv2.line(line_image, (left_avg_x, 0), (left_avg_x, line_image.shape[0]), (255, 0, 0), 2)
                cv2.line(line_image, (right_avg_x, 0), (right_avg_x, line_image.shape[0]), (0, 0, 255), 2)

        # Step 6: Combine the cropped image with the line image
        output_image = cv2.addWeighted(cropped_image, 1, line_image, 0.6, 0)

        # Step 7: Convert the output image back to ROS Image and publish
        line_image_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
        self.line_detection_publisher.publish(line_image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LineDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
