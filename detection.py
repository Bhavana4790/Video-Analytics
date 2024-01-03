from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

import supervision as sv
import numpy as np 
import pandas as pd 
import os
import cv2
from ultralytics import YOLO

model_person = YOLO("yolov8x.pt")
model_person.fuse()

model_helmet = YOLO('models/best.pt')


def excel_results(frame_num, result, names):
  track_id = result.tracker_id
  classes = [int(i.item()) for i in result.class_id]
  bboxes = [[round(val.item()) for val in bx] for bx in result.xyxy]
  cls_nm = [names[i] for i in classes]
  Output = {
      "Frame_num" : frame_num,
      "Track_id": track_id,
      "Classes" : classes,
      "Coordinates" : bboxes,
      "Prediction" : cls_nm,
  }
  return Output

def callback1(frame: np.ndarray, 
              index:int, 
              selected_classes, 
              byte_tracker, 
              byte_tracker1,
              trace_annotator,
              box_annotator,
              line_zone,
              line_zone_annotator,
              sink,
              ):
  results = model_person(frame, verbose=False)[0]
  # results = results[np.isin(results.boxes.cls.int().cpu(), selected_classes)]
  names = results.names

  results1 = model_helmet(frame)[0]
  names1 = results1.names

  detections = sv.Detections.from_ultralytics(results)
  detections = detections[np.isin(detections.class_id, selected_classes)]
  # print("before tracking", detections)
  detections = byte_tracker.update_with_detections(detections)
  # print("after tracking: ",detections)

  detections1 = sv.Detections.from_ultralytics(results1)
  # print("before tracking", detections1)
  detections1 = byte_tracker1.update_with_detections(detections1)
  # print("after tracking: ",detections1)

  Output1 = excel_results(index, detections, names)
  Output2 = excel_results(index, detections1, names1)

  # tracking code - video
  labels = [
      f"#{tracker_id} {model_person.model.names[class_id]} {confidence:0.2f}"
      for _, _, confidence, class_id, tracker_id
      in detections
  ]

  labels1 = [
      f"#{tracker_id} {model_helmet.model.names[class_id]} {confidence:0.2f}"
      for _, _, confidence, class_id, tracker_id
      in detections1
  ]

  annotated_frame = trace_annotator.annotate(
      scene=frame.copy(),
      detections=detections
  )
  annotated_frame1 = trace_annotator.annotate(
      scene=frame.copy(),
      detections=detections1
  )

  annotated_frame=box_annotator.annotate(
      scene=annotated_frame,
      detections=detections,
      labels=labels)

  annotated_frame1 = box_annotator.annotate(
      scene=annotated_frame1,
      detections=detections1,
      labels=labels1)

  # update line counter
  line_zone.trigger(detections1)

  # return frame with box and line annotated result
  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
  line_zone_annotator.annotate(annotated_frame1, line_counter=line_zone)

  sink.write_frame(line_zone_annotator.annotate(annotated_frame, line_counter=line_zone))
  sink.write_frame(line_zone_annotator.annotate(annotated_frame1, line_counter=line_zone))

  return Output1, Output2

def main(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH):
    # create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # create frame generator
    # generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model_person.model.names

    selected_classes = [0,1] # 0-person , 1- bicycle

    LINE_START = sv.Point(10, 980)
    LINE_END = sv.Point(500, 120)

    byte_tracker = sv.ByteTrack(track_thresh=0.25, frame_rate=30)
    byte_tracker1 = sv.ByteTrack(track_thresh=0.25, frame_rate=30)

    # create LineZone instance, it is previously called LineCounter class
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    # create instance of BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)

    # create instance of TraceAnnotator
    trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=50)

    # create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=2)

    Md1_res = []
    Md2_res = []
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for index, frame in enumerate(
            sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
        ):
            Res1, Res2 = callback1(frame, index, selected_classes, byte_tracker, byte_tracker1, trace_annotator, box_annotator, line_zone, line_zone_annotator, sink)
            Md1_res.append(Res1)
            Md2_res.append(Res2)
            # break

    Opt1 = pd.DataFrame(Md1_res, columns = Md1_res[0].keys())
    Opt2 = pd.DataFrame(Md2_res, columns = Md2_res[0].keys())

    merged_df = pd.merge(Opt1, Opt2, on='Frame_num', how='inner')
    del merged_df['Classes_x']
    del merged_df['Classes_y']

    # merged_df.to_csv("Merged_Output.csv")
    return merged_df

    
# print(main("kidicalmass (1).mp4", "Output.avi"))
