import io
import os
import time
from base64 import b64decode

import numpy as np
import torch.hub
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def object_detection_api(api_request):

    json_object = {'success': False}

    if api_request.method == "POST":
        if api_request.POST.get("image64", None) is not None:
            base64_data = api_request.POST.get("image64", None).split(',', 1)[1]
            data = b64decode(base64_data)
            data = np.array(Image.open(io.BytesIO(data)))
            detection_time = detect(data)

        elif api_request.FILES.get("image", None) is not None:
            image_api_request = api_request.FILES["image"]
            image_bytes = image_api_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            detection_time = detect(image, web=False)

        json_object['success'] = True
        json_object['time'] = str(round(detection_time)) + " seconds"
    return JsonResponse(json_object)


def detect_request(api_request):
    return render(api_request, 'index.html')


def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")


def detect(original_image, web=True):
    model_name = find_model()
    model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
    print(model.eval())

    start = time.time()
    result = model(original_image, size=640)
    result.save('media/')
    end = time.time()

    return round(end - start)
