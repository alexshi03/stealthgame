from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from functions import load_trained_model
from torchvision import transforms

app = Flask(__name__)

model_path = "./faster_rcnn_mobile_net_epoch_29_loss_0.0041_val_map_1.000000.pth"

model = load_trained_model(model_path)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    img_bytes = request.data
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # print(image)

    # Transform and predict
    trans = transforms.Compose([transforms.ToTensor()])
    img_tensor = trans(image)
    print(img_tensor.shape)
    output = model([img_tensor])

    # Your logic to interpret output
    # player_detected = output_confirms_player_presence(output)

    print(output)

    bx = output[0]['boxes'].tolist()
    sc = output[0]['scores'].tolist()
    player_box = []
    score = -1
    box_score = []
    if len(bx) > 0:
        print(output)
        for box, score in zip(bx, sc):
            box_score.append((box, score))

        box_score.sort(key=lambda a : a[1], reverse=True)

        player_box = box_score[0][0]
        score = box_score[0][1]

    if len(output):
        boxes = player_box
        confidence = score

    return jsonify({
            "boxes": boxes,
            "confidence": confidence
        })

if __name__ == "__main__":
    app.run(port=5000)