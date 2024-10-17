#!/usr/bin/env python

import rospy
from rospy import ROSInterruptException
from std_msgs.msg import String
from dino_sam2.msg import object_info
import pathlib
from std_msgs.msg import Header


class ObjectCheck:
    def __init__(self):
        rospy.init_node("object_checker", anonymous=True)

        self.file_path = (
            pathlib.Path(__file__).parent.parent / "object" / "name_list.txt"
        )
        self.previous_objects = set()

        self.pub = rospy.Publisher("/object_info", object_info, queue_size=10)
        self.rate = rospy.Rate(1)

    def read_objects_from_file(self):
        try:
            with open(self.file_path, "r") as f:
                objects = {line.strip() for line in f if line.strip()}
                return objects
        except Exception as e:
            rospy.logwarn("Error reading file: %s", str(e))
            return set()

    def check_for_changes(self):
        current_objects = self.read_objects_from_file()

        # find new objects and removed objects
        added_objects = current_objects - self.previous_objects
        removed_objects = self.previous_objects - current_objects

        msg = object_info()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "object_checker"
        msg.header = header

        # publish new objects
        for obj in added_objects:
            msg.name = obj
            msg.mode = 0
            self.pub.publish(msg)

        # publish removed objects
        for obj in removed_objects:
            msg.name = obj
            msg.mode = 1
            self.pub.publish(msg)

        self.previous_objects = current_objects

    def run(self):
        while not rospy.is_shutdown():
            self.check_for_changes()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        checker = ObjectCheck()
        checker.run()
    except ROSInterruptException:
        pass
# apple
