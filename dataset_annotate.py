from label_studio_sdk import Client
import json
import os
from dotenv import load_dotenv

# Load environment variables (optional but recommended)
load_dotenv()

# Configuration
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MjY5NTg1MiwiaWF0IjoxNzY1NDk1ODUyLCJqdGkiOiI5NjU4NWEyMGM4YjY0NDgzOGQ1MmNhZDY2Mzg4OGE0OSIsInVzZXJfaWQiOiIxIn0.1dS1nhbi60iknzeGdzYnG1PyVyhCags3BQ7EFzrzp-0")  # Get from Label Studio

def connect_to_label_studio():
    """Connect to Label Studio and verify connection"""
    
    # Check if API key is set
    if not LABEL_STUDIO_API_KEY or LABEL_STUDIO_API_KEY == "api_key_here":
        print("❌ Error: Please set your Label Studio API key")
        print("\nHow to get your API key:")
        print("1. Open Label Studio (http://localhost:8080)")
        print("2. Click on your username (top right)")
        print("3. Go to 'Account & Settings'")
        print("4. Copy the API key from 'Access Token'")
        print("5. Set it in .env file or replace in code")
        return None
    
    try:
        # Connect to Label Studio
        client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
        
        # Verify connection
        me = client.get_me()  # or client.users.get_current_user()
        
        print("✅ Successfully connected to Label Studio!")
        print(f"User: {me.email}")
        print(f"ID: {me.id}")
        
        # List projects (optional)
        projects = client.get_projects()
        print(f"\nProjects found: {len(projects)}")
        for project in projects:
            print(f"  - {project.title} (ID: {project.id})")
        
        return client
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Label Studio is running (label-studio start)")
        print("2. Check if URL is correct:", LABEL_STUDIO_URL)
        print("3. Verify your API key is valid")
        return None

def main():
    # Connect
    client = connect_to_label_studio()
    
    if client:
        # Example: Create a new project
        try:
            project = client.create_project(
                title="Game Bot Detection",
                label_config='''
                    <View>
                      <Image name="image" value="$image"/>
                      <RectangleLabels name="label" toName="image">
                        <Label value="player" background="green"/>
                        <Label value="enemy" background="red"/>
                        <Label value="item" background="blue"/>
                      </RectangleLabels>
                    </View>
                '''
            )
            print(f"\n✅ Created project: {project.title}")
        except Exception as e:
            print(f"Project creation: {e}")

if __name__ == "__main__":
    main()