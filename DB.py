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
    insert_query = """ INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date) VALUES (%s, %s, %s, %s, %s, %s, %s) """
    values = [
        (
            image_hashes[i],
            str(metadata['prompt']),
            str(metadata['negative_prompt']),
            str(metadata['seedNumber']),
            str(metadata['guidance']),
            str(metadata['model']),
            datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')
        )
        for i in range(int(metadata['image_count']))
    ]

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
        documents = [
            {
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
            for i in range(int(metadata['image_count']))
        ]
        result = await collection.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
    except Exception as e:
        logging.error(str(e))
        


import numpy as np

async def process_images_and_store_hashes(image_results, metadata):
    print("Processing images and storing hashes")
    image_hashes_mongo = []
    image_hashes = []
    for image in image_results:
        # Extracting hashes
        image_hash = imagehash.phash(image, hash_size=8)
        image_hash = await twos_complement(str(image_hash), 64)

        image_hashes.append(image_hash)
        image_hashes_mongo.append(image_hash)

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