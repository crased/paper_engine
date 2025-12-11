from label_studio_sdk import LabelStudio
import json

# set url for label studio
label_studio_url = "http://localhost:8080"
# set api key for label studio
label_studio_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MjY4OTgyMCwiaWF0IjoxNzY1NDg5ODIwLCJqdGkiOiJhOWQ3ZjgzNjYwMDU0NDQ5YjZhYjVlMDI4YjY1ZGRkNSIsInVzZXJfaWQiOiIxIn0.npUJ3DboN9IQjkx3igOO41dhdSjwtdoOuIZag4enfg4"

# connect to label studio API
client = LabelStudio(base_url=label_studio_url, api_key=label_studio_api_key)
# A basic request to verify connection is working
me = client.users.whoami()

print("username:", me.username)
print("email:", me.email)