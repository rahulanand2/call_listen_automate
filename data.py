__author__ = 'Rahul Anand'

from pymongo import MongoClient
from pymongo.errors import PyMongoError

def store_response(file_name, formatted_transcript, mongo_uri, db_name = 'Audio_conversation', collection_name= 'call_listening'):
    """
    Stores the formatted transcript in MongoDB with the file name as the key.

    Args:
        file_name (str): The name of the file.
        formatted_transcript (str): The formatted transcript of the audio file.
        mongo_uri (str): URI for connecting to MongoDB.
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection in the database.

    Returns:
        bool: True if storage is successful, False otherwise.
    """
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Store the formatted transcript with the file name as key
        data = {"file_name": file_name, "transcript": formatted_transcript}
        collection.insert_one(data)
        print(f"Transcript for '{file_name}' stored in MongoDB.")
        return True
    except PyMongoError as e:
        print(f"Failed to store transcript for '{file_name}' in MongoDB. Error: {e}")
        return False


def get_text_from_mongodb(file_name, mongo_uri, db_name = 'Audio_conversation', collection_name= 'call_listening'):
    """
    Retrieves the formatted transcript text from MongoDB based on the file name.

    Args:
        file_name (str): The name of the file whose transcript is to be retrieved.
        mongo_uri (str): URI for connecting to MongoDB.
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection in the database.

    Returns:
        str: The formatted transcript text if found, None otherwise.
    """
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Retrieve the document matching the file_name
        document = collection.find_one({"file_name": file_name})

        if document:
            return document["transcript"]
        else:
            print(f"No document found for file_name: {file_name}")
            return None
    except PyMongoError as e:
        print(f"Failed to retrieve data from MongoDB for '{file_name}'. Error: {e}")
        return None

# mongo_uri = "mongodb://localhost:27017/"
# db_name = "Audio Conversation"
# collection_name = "call_listening"
# file_path = "path_to_your_audio_file.wav"

