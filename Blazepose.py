from openvino.inference_engine import IECore
from pathlib import Path
import cv2
import numpy as np
import mediapipe_utils as mpu
import open3d as o3d
from o3d_utils import create_grid, create_segment

SCRIPT_DIR = Path(__file__).resolve().parent
PD_MODEL_XML = SCRIPT_DIR / "models/pose_detection_FP32.xml"
LM_MODEL_XML = SCRIPT_DIR / "models/pose_landmark_heavy_FP32.xml"
PD_MODEL_BIN = SCRIPT_DIR / "models/pose_detection_FP32.bin"
LM_MODEL_BIN = SCRIPT_DIR / "models/pose_landmark_heavy_FP32.bin"

LINES_FULL_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], 
                    [23,24],
                    [22,16,18,20,16,14,12], 
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]

rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINE_MESH_FULL_BODY = [ [9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15],
                        [24,26],[26,28],[32,30],
                        [23,25],[25,27],[29,31]]

COLORS_FULL_BODY = ["middle","right","left",
                    "right","right","right","right","right",
                    "middle","middle","middle","middle",
                    "left","left","left","left","left",
                    "right","right","right","left","left","left"]
COLORS_FULL_BODY = [rgb[x] for x in COLORS_FULL_BODY]

class Blazepose:
    def __init__(self, 
                 pd_xml=PD_MODEL_XML,
                 pd_bin=PD_MODEL_BIN,
                 lm_xml=LM_MODEL_XML,
                 lm_bin=LM_MODEL_BIN,
                 device="GPU",
                 pd_score_thresh = 0.5,
                 lm_score_thresh = 0.5,
                 filter_window_size = 5,
                 filter_velocity_scale = 10):

        self.pd_bin = pd_bin
        self.pd_xml = pd_xml
        self.lm_bin = lm_bin
        self.lm_xml = lm_xml
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh

        self.nb_lms = 35

        self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_lms-2, 3))

        self.load_models(pd_xml, lm_xml, pd_bin, lm_bin, device)

        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt

        anchor_options = mpu.SSDAnchorOptions(
                                num_layers=5, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=224,
                                input_size_width=224,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 32, 32, 32],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        
        self.anchors = mpu.generate_anchors(anchor_options)

        self.cap = cv2.VideoCapture(0)
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 3D
        self.vis3d = o3d.visualization.Visualizer()
        self.vis3d.create_window()
        opt = self.vis3d.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        z = min(video_height, video_width)/3
        self.grid_floor = create_grid([0,video_height,-z],[video_width,video_height,-z],[video_width,video_height,z],[0,video_height,z],5,2, color=(1,1,1))
        self.grid_wall = create_grid([0,0,z],[video_width,0,z],[video_width,video_height,z],[0,video_height,z],5,2, color=(1,1,1))
        self.vis3d.add_geometry(self.grid_floor)
        self.vis3d.add_geometry(self.grid_wall)
        view_control = self.vis3d.get_view_control()
        view_control.set_up(np.array([0,-1,0]))
        view_control.set_front(np.array([0,0,-1]))
    
    def load_models(self, pd_xml, lm_xml, pd_bin, lm_bin, device):
        self.ie = IECore()

        # Pose detection model
        self.pd_net = self.ie.read_network(model=pd_xml, weights=pd_bin)
        # Input blob: input_1 - shape: [1, 3, 224, 224]
        # Output blob: Identity - shape: [1, 2254, 12]
        # Output blob: Identity_1 - shape: [1, 2254, 1]
        
        _,_,self.pd_h,self.pd_w = self.pd_net.input_info["input_1"].input_data.shape
        self.pd_scores = "Identity_1"
        self.pd_bboxes = "Identity"

        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=device)

        # Landmarks model
        self.lm_net = self.ie.read_network(model=lm_xml, weights=lm_bin)
        # Input blob: input_1 - shape: [1, 3, 256, 256]
        # Output blob: ld_3d - shape: [1, 195]
        # Output blob: output_poseflag - shape: [1, 1]

        _,_,self.lm_h,self.lm_w = self.lm_net.input_info["input_1"].input_data.shape
        self.lm_score = "output_poseflag"
        self.lm_landmarks = "ld_3d"

        print("Loading landmark model to the plugin")
        self.lm_exec_net = self.ie.load_network(network=self.lm_net, num_requests=1, device_name=device)
    
    def pd_postprocess(self, inference):
        scores = np.squeeze(inference[self.pd_scores])  # 2254
        bboxes = inference[self.pd_bboxes][0] # 2254x12
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        
        mpu.detections_to_rect(self.regions, kp_pair=[0,1])
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def lm_postprocess(self, region, inference):
        region.lm_score = np.squeeze(inference[self.lm_score])
        if region.lm_score > self.lm_score_thresh:  

            lm_raw = inference[self.lm_landmarks].reshape(-1,5)

            # Normalize x,y,z. Here self.lm_w = self.lm_h and scaling in z = scaling in x = 1/self.lm_w
            lm_raw[:,:3] /= self.lm_w
            
            # region.landmarks contains the landmarks normalized 3D coordinates in the relative oriented body bounding box
            region.landmarks = lm_raw[:,:3]
            # Calculate the landmark coordinate in square padded image (region.landmarks_padded)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(region.landmarks[:self.nb_lms,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            # A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
            # original image. Then I arbitrarily divide by 4 for a more realistic appearance.
            lm_z = region.landmarks[:self.nb_lms,2:3] * region.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))

            lm_xyz = self.filter.apply(lm_xyz)
            region.landmarks_padded = lm_xyz.astype(np.int)
            # If we added padding to make the image square, we need to remove this padding from landmark coordinates
            # region.landmarks_abs contains absolute landmark coordinates in the original image (padding removed))
            region.landmarks_abs = region.landmarks_padded.copy()
            if self.pad_h > 0:
                region.landmarks_abs[:,1] -= self.pad_h
            if self.pad_w > 0:
                region.landmarks_abs[:,0] -= self.pad_w

    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_thresh:

            list_connections = LINES_FULL_BODY
            lines = [np.array([region.landmarks_padded[point,:2] for point in line]) for line in list_connections]
            cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
            
            for i,x_y in enumerate(region.landmarks_padded[:self.nb_lms-2,:2]):
                if i > 10:
                    color = (0,255,0) if i%2==0 else (0,0,255)
                elif i == 0:
                    color = (0,255,255)
                elif i in [4,5,6,8,10]:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

            points = region.landmarks_abs
            lines = LINE_MESH_FULL_BODY
            colors = COLORS_FULL_BODY
            
            # Angle calculator
            right_shoulder = points[12]
            right_elbow = points[14]
            right_wrist = points[16]

            se = right_elbow - right_shoulder
            ew = right_wrist - right_elbow

            theta_rad = np.pi -  np.arccos(se.dot(ew)/(np.linalg.norm(se) * np.linalg.norm(ew)))

            theta = int(np.degrees(theta_rad))

            blank = np.zeros((1000, 500), dtype=np.uint8)
            counter = cv2.putText(blank, str(theta), (150, 100), cv2.FONT_HERSHEY_COMPLEX, 3, 255, 2)
            counter = cv2.rectangle(counter, (0, theta*5+100), (500, 1000), 255, -1)
            cv2.imshow('counter', counter)

            for i,a_b in enumerate(lines):
                a, b = a_b
                line = create_segment(points[a], points[b], radius=5, color=colors[i])
                if line: self.vis3d.add_geometry(line, reset_bounding_box=False)

    def run(self):

        while True:
            ok, vid_frame = self.cap.read()
            if not ok:
                break
            
            h, w = vid_frame.shape[:2]

            # Padding on the small side to get a square shape
            self.frame_size = max(h, w)
            self.pad_h = int((self.frame_size - h)/2)
            self.pad_w = int((self.frame_size - w)/2)
            video_frame = cv2.copyMakeBorder(vid_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

            annotated_frame = video_frame.copy()

            # Infer pose detection
            # Resize image to NN square input shape
            frame_nn = cv2.resize(video_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
            # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2,0,1))[None,]

            inference = self.pd_exec_net.infer(inputs={"input_1": frame_nn})
            self.pd_postprocess(inference)

            self.vis3d.clear_geometries()
            self.vis3d.add_geometry(self.grid_floor, reset_bounding_box=False)
            self.vis3d.add_geometry(self.grid_wall, reset_bounding_box=False)

            # Landmarks
            for r in self.regions:
                frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_w, self.lm_h)
                # Transpose hxwx3 -> 1x3xhxw
                frame_nn = np.transpose(frame_nn, (2,0,1))[None,] / 255.0
                # Get landmarks
                inference = self.lm_exec_net.infer(inputs={"input_1": frame_nn})
                self.lm_postprocess(r, inference)
                self.lm_render(annotated_frame, r)
            
            self.vis3d.poll_events()
            self.vis3d.update_renderer()

            self.filter.reset()

            cv2.imshow("Blazepose", annotated_frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break



BP = Blazepose()

BP.run()