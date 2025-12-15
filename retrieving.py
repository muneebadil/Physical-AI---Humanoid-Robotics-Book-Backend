import cohere
from qdrant_client import QdrantClient

# Initialize Cohere client
cohere_client = cohere.Client("oT2uItxfZZXcibYkWpFA6MhjH2wfiN86zkehMjAK")

# Connect to Qdrant
qdrant= QdrantClient(
    url="https://092f4da6-e5bc-46dd-8f91-c1953e8513e5.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.U2-91G5R8r_8BvEo5PrOpbya1gxlyfP1sqr2lSL0BkM",
)

def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding

def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]

# Test
print(retrieve("What data do you have?"))
