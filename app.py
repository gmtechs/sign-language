from ultralytics import YOLO
import gradio as gr
import requests
import cv2
import os
 
model = YOLO('best.pt')

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, box in enumerate(results.boxes):
        det = box.xyxy[0]
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        label = f"{results.names[box.cls.item()]} {box.conf.item():.2}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        cv2.rectangle(
            image,
            (int(det[0]), int(det[1]) - 20),
            (int(det[0] + w), int(det[1])), 
            (0, 0, 255),
            -1)
        cv2.putText(image, label, (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs_image  = gr.components.Image(type="filepath", label="Input Image")
outputs_image = gr.components.Image(type="numpy", label="Output Image")
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Hand Sign Language",
    examples=[["sample.jpg"]],
    cache_examples=False,
)

def show_preds_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            outputs = model.predict(source=frame)
            results = outputs[0].cpu().numpy()
            for i, box in enumerate(results.boxes):
                det = box.xyxy[0]
                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                label = f"{results.names[box.cls.item()]} {box.conf.item():.2}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1]) - 20),
                    (int(det[0] + w), int(det[1])), 
                    (0, 0, 255),
                    -1)
                cv2.putText(frame_copy, label, (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
 
inputs_video  = gr.components.Video(label="Input Video")
outputs_video = gr.components.Image(type="numpy", label="Output Image")
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Hand Sign Language",
)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference']
).queue().launch()