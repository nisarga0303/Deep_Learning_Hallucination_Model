import os

import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(BASE_DIR, "..", "data", "imgs", "xmlab56", "source.jpg")
URL = "http://127.0.0.1:8000/predict"


def run_test(model_type, question):
    with open(TEST_IMAGE, "rb") as handle:
        files = {"file": ("source.jpg", handle, "image/jpeg")}
        data = {"question": question, "modelType": model_type}
        response = requests.post(URL, files=files, data=data, timeout=30)
    print(model_type, response.status_code, response.json())


if __name__ == "__main__":
    run_test("traditional", "Which organ is shown in this image?")
    run_test("hybrid", "Which organ is shown in this image?")
