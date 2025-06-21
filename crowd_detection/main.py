import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from tqdm import tqdm
import cv2



def main(input_filename: str, output_filename: str):
    """ A function that detects people and renders them on video

    - input_filename: the filename located in the project root with the extension .mp4
    - output_filename: the filename of the output video with the extension ('.mp4'>)
    """
    # initilize the model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # read video
    capture = cv2.VideoCapture(input_filename)

    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initilize the output video according to the given video parametres
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_filename, fourcc, fps, (frame_width, frame_height)
    )

    # processing with pretrained model
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = capture.read()

        if not ret:
            break

        # convert frame to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(frame_rgb).to(device)

        # run inference
        with torch.no_grad():
            predictions = model([image_tensor])[0]

        # filter predictions for 'person' class
        person_boxes = predictions["boxes"][(predictions["labels"] == 1)]
        person_scores = predictions["scores"][(predictions["labels"] == 1)]
        # print('Number of detected persons:', len(person_boxes))

        # draw bounding boxes
        for box, score in zip(
            person_boxes.cpu().numpy(), person_scores.cpu().numpy()
        ):
            if score > 0.8:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    f"person: {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # write the frame
        out.write(frame)

    # release resources
    cv2.destroyAllWindows()
    out.release()
    capture.release()


if __name__ == "__main__":
    main("crowd.mp4", "crowd_detected.mp4")
