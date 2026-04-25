import os
import shutil
import tempfile
import glob
import base64
from PIL import Image
from django.shortcuts import render
from ultralytics import YOLO

# Load model once
model = YOLO("best.pt")

def index(request):
    result_img = None

    if request.method == 'POST' and request.FILES.get('image'):

        temp_dir = tempfile.mkdtemp()

        try:
            uploaded_file = request.FILES['image']
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            # Save uploaded file
            with open(temp_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            # 🔥 Convert to JPG (fix web image issue)
            img = Image.open(temp_path).convert("RGB")
            temp_path = os.path.splitext(temp_path)[0] + ".jpg"
            img.save(temp_path)

            # Run YOLO
            results = model.predict(
                temp_path,
                save=True,
                project=temp_dir,
                name="output"
            )

            # Get actual saved file
            output_folder = os.path.join(temp_dir, "output")
            files = glob.glob(os.path.join(output_folder, "*"))

            if not files:
                raise Exception("No output image generated")

            output_path = files[0]

            # Convert to base64
            with open(output_path, "rb") as img_file:
                result_img = base64.b64encode(img_file.read()).decode('utf-8')

        finally:
            shutil.rmtree(temp_dir)

    return render(request, 'index.html', {'result_img': result_img})