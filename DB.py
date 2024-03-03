import os
import datetime
import logging
import io

from PIL import Image
import psycopg
from dotenv import load_dotenv
import imagehash

# DB Connection
DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")
DSN = f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS}"

session = None


async def insert_image_hashes(image_hashes, metadata):
    insert_query = """
        INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    values = [
        (
            image_hashes[i],
            metadata['prompt'],
            metadata['negative_prompt'],
            metadata['seed'],
            metadata['guidance'],
            metadata['model'],
            datetime.now(),
        )
        for i in range(4)
    ]

    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            # Use executemany to insert multiple records
            await acur.executemany(insert_query, values)
            await aconn.commit()  # Commit the transaction


async def twos_complement(hexstr, bits):
    value = int(hexstr, 16)  # convert hexadecimal to integer

    # convert from unsigned number to signed number with "bits" bits
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


async def process_images_and_store_hashes(image_results, metadata):
    image_hashes = []
    for image in image_results:
        image_hash = imagehash.average_hash(image, 8)
        image_hash = await twos_complement(str(image_hash), 64)
        image_hashes.append(image_hash)

    try:
        await insert_image_hashes(image_hashes, metadata)
    except Exception as e:
        logging.error(str(e))
