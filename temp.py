from google import genai
from google.genai import types
client = genai.Client(
    api_key='AIzaSyDU2yJ7byCkHr3RiJ5KfDAYm1An0RPSSHM',
    http_options=types.HttpOptions(api_version='v1beta')
)
for m in client.models.list():
    print(m.name)
    