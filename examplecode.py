from ollama import Client
import chromadb
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

client = chromadb.DummyClient()


# client = chromadb.PersistentClient()
remote_client = Client(host=f"http://127.0.0.1:11434")
collection = client.get_or_create_collection(name="articles_demo")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separators=[","]
) 

with open("counter.txt","r") as f:
    count=int(f.read().strip())

print("Reading articles.jsonl and generating embeddings...")

with open("articles.jsonl", "r") as f:

    for i, line in enumerate(f):

        if i < count:
            print("Skipping already processed article")
            continue

        article = json.loads(line)
        content = article["content"]

        sentences = text_splitter.split_text(content)

        for sentence in sentences:

            response = remote_client.embed(
                model="qwen2.5:3b",
                input=f"search_document: {sentence}"
            )

            embedding = response["embeddings"][0]

            collection.add(
                ids=[f"article_{i}_{sentence[:5]}"],
                embeddings=[embedding],
                documents=[sentence],
                metadatas=[{"title": article["title"]}],
            )

        count += 1

print("Database built successfully!")

#save counter
with open("counter.txt","w")as f:
   f.write(str(count))

#Query section
while True:
    print("----------------------------------")
    query= input("how may i help you?\n")
    if query=="break":
        break
    # query = "what are different problems provinces of nepal are facing?"
    #query = "are there any predicted hindrance for upcoming election ?"
    query_embed = remote_client.embed(model="qwen2.5:3b", input=f"query: {query}")["embeddings"][0]
    results = collection.query(query_embeddings=[query_embed], n_results=1)
    #print(f"\nQuestion: {query}")
    #print(f'\n Title : {results["metadatas"][0][0]["title"]} \n {results["documents"][0][0]} ')
    context='\n'.join(results["documents"][0])

    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

    Context: {context}

    Question: {query}

    Answer:"""

    response = remote_client.generate(
            model="qwen2.5:3b",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )

    answer = response['response']

    print(answer)
