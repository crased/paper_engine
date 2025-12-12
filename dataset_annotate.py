import api_key
# Define the URL where Label Studio is accessible
LABEL_STUDIO_URL = 'get_url'

# API key can be either your personal access token or legacy access token
LABEL_STUDIO_API_KEY = 'key'

# Import the SDK and the client module
from label_studio_sdk import LabelStudio
client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)