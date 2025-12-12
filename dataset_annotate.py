import api_key
from label_studio_sdk import LabelStudio

client = LabelStudio(
    base_url='http://localhost:8080',  
    api_key="key",
)