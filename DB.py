import os
import datetime
import logging
import io
from PIL import Image
import psycopg
from dotenv import load_dotenv
import imagehash
import time
from motor import motor_asyncio

# load the .env file using the load_dotenv() method:
load_dotenv(dotenv_path=".env")

# DB Connection
DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")
DSN = f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS}"

print(DSN)

session = None

MONGO_URI = os.environ.get("MONGO_URI")
client = motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = motor_asyncio.AsyncIOMotorDatabase(client, "jscammie-com")
collection = db.userGenerations



async def insert_image_hashes(image_hashes, metadata, image_hash_mongo, image_results):
    
    print(f"DEBUG insert_image_hashes: Received {len(image_hashes)} hashes, metadata image_count: {metadata.get('image_count', 'unknown')}")
    
    # Fix for img2video: use actual hash count instead of metadata image_count
    actual_image_count = len(image_hashes)
    metadata_image_count = int(metadata.get('image_count', actual_image_count))
    
    if metadata.get('request_type') == 'img2video':
        print(f"DEBUG: img2video request - using actual image count {actual_image_count} instead of metadata count {metadata_image_count}")
        image_count = actual_image_count
    else:
        image_count = metadata_image_count
        
    if actual_image_count != image_count:
        print(f"WARNING: Image count mismatch - actual: {actual_image_count}, using: {image_count}")
    
    insert_query = """ INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date, loras) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) """
    
    try:
        print(f"DEBUG: Creating values for {image_count} images")
        
        values = []
        for i in range(image_count):
            if i >= len(image_hashes):
                print(f"ERROR: Trying to access image_hashes[{i}] but array only has {len(image_hashes)} elements")
                raise IndexError(f"image_hashes index {i} out of range (length: {len(image_hashes)})")
                
            value_tuple = (
                image_hashes[i],
                str(metadata['prompt']),
                str(metadata['negative_prompt']),
                str(metadata['seedNumber']),
                str(metadata['guidance']),
                str(metadata['model']),
                datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'),
                metadata['loraStrings']
            )
            values.append(value_tuple)
            print(f"DEBUG: Created value tuple for image {i}")
            
    except Exception as values_error:
        print(f"ERROR creating values array: {values_error}")
        raise values_error

    try:
        async with await psycopg.AsyncConnection.connect(DSN) as aconn:
            async with aconn.cursor() as acur:
                # Insert into PostgreSQL
                await acur.executemany(insert_query, values)
                await aconn.commit()
    except Exception as e:
        logging.error(str(e))
        
    try:

        # convert aiData to a yaml string like this javascript code:
        # Insert into MongoDB
        print(f"DEBUG: Storing model in MongoDB: '{metadata['model']}'")
        
        # Fix for img2video: use actual hash count instead of metadata image_count
        actual_image_count = len(image_hash_mongo)
        metadata_image_count = int(metadata.get('image_count', actual_image_count))
        
        if metadata.get('request_type') == 'img2video':
            print(f"DEBUG: img2video MongoDB - using actual image count {actual_image_count} instead of metadata count {metadata_image_count}")
            image_count = actual_image_count
        else:
            image_count = metadata_image_count
            
        print(f"DEBUG: Creating MongoDB documents for {image_count} images with {len(image_hash_mongo)} hashes")
        
        documents = []
        
        for i in range(image_count):
            if i >= len(image_hash_mongo):
                print(f"ERROR: Trying to access image_hash_mongo[{i}] but array only has {len(image_hash_mongo)} elements")
                raise IndexError(f"image_hash_mongo index {i} out of range (length: {len(image_hash_mongo)})")
                
            document = {
                "hash": str(image_hash_mongo[i]),
                "request_type": str(metadata['request_type']),
                "prompt": str(metadata['prompt']),
                "negative_prompt": str(metadata['negative_prompt']),
                "aspectRatio": str(metadata['aspect_ratio']),
                "seed": str(metadata['seedNumber']),
                "loras": str(metadata['lora']),
                "lora_strengths": str(metadata['lora_strengths']),
                "cfg": str(metadata['guidance']),
                "model": str(metadata['model']),
                "timestamp": time.time(),
            }
            documents.append(document)
            print(f"DEBUG: Created MongoDB document for image {i}")
            
        result = await collection.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
    except Exception as e:
        logging.error(str(e))
        


import numpy as np

async def process_images_and_store_hashes(image_results, metadata):
    print("Processing images and storing hashes")
    print(f"DEBUG: Processing {len(image_results)} images, metadata says {metadata.get('image_count', 'unknown')} images")
    image_hashes_mongo = []
    image_hashes = []
    
    for idx, image in enumerate(image_results):
        try:
            print(f"DEBUG: Processing image {idx + 1}/{len(image_results)}")
            # For img2video (animated WebP) or any animated image, extract 5th frame for hashing
            hash_image = image
            try:
                if hasattr(image, 'is_animated') and image.is_animated:
                    print(f"DEBUG: Image {idx} is animated, extracting 5th frame")
                    try:
                        image.seek(4)  # Go to 5th frame (0-indexed, so frame 4)
                        hash_image = image.copy()  # Copy the 5th frame as a static image
                        print(f"DEBUG: Successfully extracted 5th frame for image {idx}")
                    except EOFError:
                        print(f"DEBUG: 5th frame not available for image {idx}, falling back to first frame")
                        image.seek(0)  # Fall back to first frame
                        hash_image = image.copy()
                elif metadata.get('request_type') == 'img2video':
                    print(f"DEBUG: img2video request for image {idx}, ensuring 5th frame extraction")
                    # Even if is_animated is not detected, try to extract 5th frame for img2video
                    try:
                        image.seek(4)  # Try to go to 5th frame
                        hash_image = image.copy()
                        print(f"DEBUG: Successfully extracted 5th frame for img2video image {idx}")
                    except (EOFError, Exception) as seek_error:
                        print(f"DEBUG: 5th frame seek failed for image {idx}: {seek_error}, trying first frame")
                        try:
                            image.seek(0)
                            hash_image = image.copy()
                        except Exception as fallback_error:
                            print(f"DEBUG: First frame fallback also failed for image {idx}: {fallback_error}, using original image")
                            hash_image = image
            except Exception as frame_error:
                print(f"Warning: Could not process animated image {idx} for hashing: {frame_error}, using original image")
                hash_image = image
                
            # Extracting hashes from the processed image
            print(f"DEBUG: Computing hash for image {idx}")
            image_hash = imagehash.phash(hash_image, hash_size=8)
            image_hash_converted = await twos_complement(str(image_hash), 64)
            print(f"DEBUG: Hash computed for image {idx}: {image_hash_converted}")

            image_hashes.append(image_hash_converted)
            image_hashes_mongo.append(image_hash_converted)
            
        except Exception as hash_error:
            print(f"ERROR: Failed to process hash for image {idx}: {hash_error}")
            # Don't append anything to the arrays if hashing fails
            logging.error(f"Hash processing failed for image {idx}: {hash_error}")

    print(f"DEBUG: Final arrays - image_hashes: {len(image_hashes)}, image_hashes_mongo: {len(image_hashes_mongo)}")
    print(f"DEBUG: Expected image_count from metadata: {metadata.get('image_count', 'unknown')}")
    
    if len(image_hashes) == 0:
        print("ERROR: No hashes were successfully generated!")
        return
        
    try:
        await insert_image_hashes(image_hashes, metadata, image_hashes_mongo, image_results)
    except Exception as e:
        print(f"ERROR in insert_image_hashes: {e}")
        logging.error(str(e))

    try:
        await insert_image_hashes(image_hashes, metadata, image_hashes_mongo, image_results)
    except Exception as e:
        logging.error(str(e))

async def twos_complement(hexstr, bits):
    value = int(hexstr, 16)  # convert hexadecimal to integer

    # convert from unsigned number to signed number with "bits" bits
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value