#!/usr/bin/env python
"""Test GCP authentication with service account."""

import os
import json
from pathlib import Path

# Set up environment variables
service_account_path = "C:/Users/david/.auth/business/service-account-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
os.environ["GOOGLE_CLOUD_PROJECT"] = "auricleinc-gemini"
os.environ["GCP_PROFILE"] = "business"

# Remove any GEMINI_API_KEY if present
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]
    print("✓ Removed GEMINI_API_KEY environment variable")

print("=" * 60)
print("GCP Authentication Test")
print("=" * 60)

# Display environment configuration
print(f"Service Account: {service_account_path}")
print(f"Project ID: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
print(f"Profile: {os.environ.get('GCP_PROFILE')}")
print()

# Verify service account file exists
if Path(service_account_path).exists():
    print("✓ Service account key file found")

    # Load and display service account info
    with open(service_account_path, 'r') as f:
        sa_data = json.load(f)
        print(f"✓ Service Account Email: {sa_data.get('client_email')}")
        print(f"✓ Project ID: {sa_data.get('project_id')}")
else:
    print("✗ Service account key file not found!")
    exit(1)

print()
print("Testing Google Cloud authentication...")
print("-" * 40)

try:
    # Test authentication with google.auth
    import google.auth
    from google.auth import default

    credentials, project = default()
    print(f"✓ Authentication successful!")
    print(f"✓ Active project: {project}")
    print(f"✓ Credentials type: {type(credentials).__name__}")

except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("  Install with: pip install google-auth")

except Exception as e:
    print(f"✗ Authentication failed: {e}")
    exit(1)

print()
print("Testing Vertex AI / Gemini access...")
print("-" * 40)

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    # Initialize Vertex AI
    vertexai.init(project=project, location="us-central1", credentials=credentials)

    # Try to create a model instance
    model = GenerativeModel("gemini-2.0-flash-exp")
    print(f"✓ Vertex AI initialized successfully")
    print(f"✓ Model loaded: gemini-2.0-flash-exp")

    # Optional: Test a simple generation
    try:
        response = model.generate_content("Say 'Hello, authenticated world!' in 5 words or less")
        print(f"✓ Model response: {response.text.strip()}")
    except Exception as gen_err:
        print(f"⚠ Generation test failed (might be quota/permission issue): {gen_err}")

except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("  Install with: pip install google-cloud-aiplatform")

except Exception as e:
    print(f"✗ Vertex AI initialization failed: {e}")

print()
print("=" * 60)
print("Authentication Configuration Summary")
print("=" * 60)
print(f"✓ Service Account: Configured")
print(f"✓ Project: auricleinc-gemini")
print(f"✓ Profile: business")
print(f"✓ GEMINI_API_KEY: Removed (using service account instead)")
print()
print("Next steps:")
print("1. Ensure all Python scripts use service account auth")
print("2. Remove GEMINI_API_KEY from any .env files")
print("3. Update any hardcoded API key references in code")