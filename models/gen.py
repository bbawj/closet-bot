import open_clip
from PIL import Image
import torch.onnx
from ultralytics import YOLO

device = 'cpu'

# Load the model
yolo_model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', force_custom_text=True)
state_dict = torch.load("finetuned_clip.pt", map_location=device)
state_dict = open_clip.model.convert_to_custom_text_state_dict(state_dict['CLIP'])
clip_model.load_state_dict(state_dict)
clip_model = clip_model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

#Load the image
img = Image.open('maxi_dress.jpg')
img = preprocess(img).to(device)

prompt = "a photo of a"
text_inputs = ["blue cowl neck maxi-dress", "red t-shirt", "white shirt"]
text_inputs = [prompt + " " + t for t in text_inputs]

tokenized_prompt = tokenizer(text_inputs).to(device)

yolo_model.export(format="onnx")

torch.onnx.export(
        clip_model.visual,
        img.unsqueeze(0),
        "clip-image-vit-32.onnx",
        opset_version=19,
        export_params=True,
        input_names = ['input'], output_names = ['output'],
        dynamic_axes= { 'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'} }
        )

torch.onnx.export(
        clip_model.text,
        # https://github.com/microsoft/onnxruntime/issues/9760
        tokenized_prompt.to(torch.int32),
        "clip-text-vit-32.onnx",
        opset_version=19,
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes= { 'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'} }
        )   

