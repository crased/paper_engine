from label_studio_sdk import LabelStudio
import json

# set url for label studio
label_studio_url = "http://localhost:8080"
# set api key for label studio
label_studio_api_key = ""

# connect to label studio API
cli = LabelStudio(base_url=label_studio_url, api_key=label_studio_api_key)
# A basic request to verify connection is working
me = client.users.whoami()

print("username:", me.username)
print("email:", me.email)