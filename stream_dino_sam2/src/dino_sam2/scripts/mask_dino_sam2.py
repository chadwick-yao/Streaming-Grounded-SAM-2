import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dino_sam2.msg import object_info
from message_filters import ApproximateTimeSynchronizer, Subscriber

import torch
import PIL
import cv2
import os
import numpy as np
from collections import defaultdict

from sam2.sam2_camera_predictor import SAM2CameraPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ProduceMask:
    def __init__(
        self,
        device,
        dino_checkpoint: str = "IDEA-Research/grounding-dino-tiny",
        sam2_checkpoint: str = "facebook/sam2-hiera-small",
    ):
        rospy.init_node("produce_mask", anonymous=True)

        # subscriber for grounding dino
        img_sub = Subscriber("/camera/image_bgr", Image)
        obj_info_sub = Subscriber("/object_info", object_info)
        self.sub_ts = ApproximateTimeSynchronizer(
            [img_sub, obj_info_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sub_ts.registerCallback(self.callback_update_bbox)

        # subscriber for sam2
        # self.camera_sub = rospy.Subscriber(
        #     "/camera/image_bgr_copy",
        #     Image,
        #     self.callback_produce_mask,
        # )

        # publisher for mask
        self.mask_pub = rospy.Publisher("/camera/mask_bgr", Image, queue_size=10)

        self.bridge = CvBridge()
        self.object_list = set()
        self.object_bbox = dict()
        self.if_init = False

        # load model
        self.device = device

        self.grounding_processor = AutoProcessor.from_pretrained(dino_checkpoint)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_checkpoint
        ).to(self.device)

        self.mask_predictor: SAM2CameraPredictor = SAM2CameraPredictor.from_pretrained(
            sam2_checkpoint
        )

        rospy.loginfo("Model loaded")

    def callback_update_bbox(self, img_msg, obj_info_msg):
        if obj_info_msg.mode == 0:
            self.object_list.add(obj_info_msg.name)
            rospy.loginfo(f"Added object: {obj_info_msg.name}")
        elif obj_info_msg.mode == 1:
            self.object_list.discard(obj_info_msg.name)
            rospy.loginfo(f"Removed object: {obj_info_msg.name}")
        else:
            rospy.logwarn(f"Unknown mode: {obj_info_msg.mode}")

        # update bbox
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.update_bbox_dict(frame)
        except Exception as e:
            rospy.logwarn(f"Erorr updating bbox: {str(e)}")

    def callback_produce_mask(self, img_msg):
        if len(self.object_list) > 0:
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            width, height = frame.shape[:2][::-1]

            if not self.if_init:
                self.mask_predictor.load_first_frame(frame)
                ann_frame_idx = 0

                for obj_idx, (label, boxes) in enumerate(self.object_bbox.items()):
                    _, out_obj_idx, out_mask_logits = (
                        self.mask_predictor.add_new_points(
                            frame_idx=ann_frame_idx,
                            obj_id=obj_idx,
                            box=boxes,
                        )
                    )
                if self.object_bbox:
                    rospy.loginfo(f"Added bbox into mask predictor.")
                    self.if_init = True
            else:
                import ipdb

                ipdb.set_trace()
                out_obj_ids, out_mask_logits = self.mask_predictor.track(frame)
                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                # print(all_mask.shape)
                for i in range(0, len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(
                        1, 2, 0
                    ).cpu().numpy().astype(np.uint8) * 255

                    all_mask = cv2.bitwise_or(all_mask, out_mask)
                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                image_msg = self.bridge.cv2_to_imgmsg(all_mask, "bgr8")

                self.mask_pub.publish(image_msg)

    def update_bbox_dict(self, frame):
        # combine all object names
        text_prompt = ". ".join(self.object_list) + "."
        image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = self.grounding_processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.6,
            text_threshold=0.6,
            target_sizes=[frame.shape[:2]],  # TODO: check this
        )
        """
        Results is a list of dict with the following structure:
        [
            {
                'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
                'labels': ['car', 'tire', 'tire', 'tire'], 
                'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                                [1392.4701,  554.4064, 1628.6133,  777.5872],
                                [ 436.1182,  621.8940,  676.5255,  851.6897],
                                [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
            }
        ]
        """
        # save the results in the object_bbox dict
        labels = results[0]["labels"]
        boxes = results[0]["boxes"]

        # clear object_bbox
        self.object_bbox = dict()
        grouped = defaultdict(list)

        for label, box in zip(labels, boxes):
            grouped[label].append(box)

        for label, box_list in grouped.items():
            stacked_boxes = torch.stack(box_list, dim=0)
            self.object_bbox[label] = stacked_boxes

        if self.object_bbox:
            rospy.loginfo(f"Updated bbox dict: {self.object_bbox}")
            self.if_init = False

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        masker = ProduceMask(device=device)
        masker.run()
    except rospy.ROSInterruptException:
        pass
