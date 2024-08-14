from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PointInsertOperations
from configure import COLLECTION_NAME, MODEL_EMBED_SIZE
import time

class Qdrant() :
    def __init__(self) :
        start = time.time()
        print("Loading DB ...")
        self.db_client = QdrantClient(path ='./db/')
        end = time.time()

        print(f"! Completed in {end - start:.4f} sec \n")

        print("Loading Collection ...")
        try :
            self.db_client.create_collection(
                collection_name = COLLECTION_NAME,
                vectors_config = VectorParams(size = MODEL_EMBED_SIZE, distance = Distance.COSINE)
            )
        except ValueError as e :
            print(f'Collection {COLLECTION_NAME} already exists ... skip create_collection')
        print("! Complete !\n")

    def get_dbclient(self) :
        return self.db_client