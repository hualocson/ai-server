import requests

# get image from url and return base64 image
def get_remote_image(image_url: str):
  try:
    response = requests.get(image_url)
    image_data = response.content
    return image_data
  except Exception as e:
    print(str(e))