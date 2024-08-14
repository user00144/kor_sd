from huggingface_hub import login
TOKEN = ""
login(TOKEN)
print('\n')

from db_loader import Qdrant
db_client = Qdrant().get_dbclient()

from text_embed import Text_Embeder
TE = Text_Embeder()

# from upsert_db import Upserter
# operation_info = Upserter(TE, db_client).upsert()

from querysystem import Query_system
query_system = Query_system(TE, db_client)

query_system.exp_()