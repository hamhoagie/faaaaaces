#!/usr/bin/env python3
"""
Test script to verify face reconstruction functionality
"""

import requests
import json
import os
import sys

def test_face_reconstruction():
    """Test the complete face reconstruction flow"""
    
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Face Reconstruction Flow")
    print("=" * 50)
    
    # Step 1: Get all masked faces
    print("\n1. Getting all masked faces...")
    response = requests.get(f"{base_url}/api/masked/all_masked_faces")
    
    if response.status_code != 200:
        print(f"❌ Failed to get masked faces: {response.status_code}")
        return False
    
    data = response.json()
    total_faces = data.get('total_faces', 0)
    print(f"   Found {total_faces} masked faces")
    
    if total_faces == 0:
        print("❌ No masked faces found - run mask detection first")
        return False
    
    # Step 2: Select a test face
    test_face = data['all_faces'][0]
    face_id = test_face['face_id']
    print(f"   Selected test face: {face_id}")
    
    # Step 3: Test single face reconstruction
    print(f"\n2. Testing reconstruction of face {face_id}...")
    
    reconstruction_data = {
        'face_ids': [face_id],
        'method': 'classical'
    }
    
    response = requests.post(
        f"{base_url}/api/masked/batch_reconstruct",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(reconstruction_data)
    )
    
    if response.status_code != 200:
        print(f"❌ Reconstruction failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    
    result = response.json()
    
    if not result.get('success', False):
        print(f"❌ Reconstruction unsuccessful: {result}")
        return False
    
    print(f"   ✅ Reconstruction successful!")
    print(f"   - Method: {result.get('method')}")
    print(f"   - Successful: {result.get('successful_reconstructions')}")
    print(f"   - Failed: {result.get('failed_reconstructions')}")
    
    # Step 4: Verify reconstructed image exists
    if result.get('results'):
        first_result = result['results'][0]
        reconstructed_path = first_result.get('reconstructed_path', '')
        
        if reconstructed_path:
            # Convert web path to file path
            file_path = reconstructed_path.lstrip('/')
            full_path = os.path.join(os.getcwd(), file_path)
            
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                print(f"   ✅ Reconstructed image exists: {file_path} ({file_size} bytes)")
                
                # Step 5: Test image accessibility via web server
                print(f"\n3. Testing reconstructed image web access...")
                img_response = requests.head(f"{base_url}{reconstructed_path}")
                
                if img_response.status_code == 200:
                    print(f"   ✅ Reconstructed image accessible via web server")
                    return True
                else:
                    print(f"   ❌ Reconstructed image not accessible: {img_response.status_code}")
                    return False
            else:
                print(f"   ❌ Reconstructed image file not found: {full_path}")
                return False
        else:
            print(f"   ❌ No reconstructed path in result")
            return False
    else:
        print(f"   ❌ No results in response")
        return False

if __name__ == "__main__":
    try:
        success = test_face_reconstruction()
        if success:
            print("\n🎉 All tests passed! Face reconstruction is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Tests failed! Face reconstruction has issues.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        sys.exit(1)