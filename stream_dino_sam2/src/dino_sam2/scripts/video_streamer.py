#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Header


class VideoStreamer:
    def __init__(self):
        rospy.init_node("video_streamer", anonymous=True)
        self.image_pub = rospy.Publisher("/camera/image_bgr", Image, queue_size=10)
        self.image_pub_copy = rospy.Publisher(
            "/camera/image_bgr_copy",
            Image,
            queue_size=10,
        )
        self.mask_sub = rospy.Subscriber("/camera/mask_bgr", Image, self.callback)

        self.bridge = CvBridge()
        self._mask = None

        self.camera_index = self.find_camera()
        if self.camera_index is None:
            raise RuntimeError("No camera found")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera")
            raise RuntimeError("Cannot open camera")

        self.rate = rospy.Rate(30)

    def find_camera(self):
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                cap.release()
                return index
            cap.release()
            index += 1
            if index > 10:  # Limit the search to the first 10 indices
                rospy.logerr("No camera found")
                return None

    def callback(self, mask_msg):
        self._mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="bgr8")

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                break

            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_frame"

            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header = header

            self.image_pub.publish(image_msg)
            self.image_pub_copy.publish(image_msg)

            if self._mask is not None:
                frame = cv2.addWeighted(frame, 1, self._mask, 0.5, 0)

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            self.rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        streamer = VideoStreamer()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as e:
        rospy.logerr(str(e))
    finally:
        print("Video streamer stopped")
