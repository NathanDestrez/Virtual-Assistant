# Use Chroma DB

Initialize the chroma persistent collection

```
chroma_client = client = chromadb.PersistentClient(path="your_path") 
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

Using Chroma Client with Langchain

```
langchain_chroma = Chroma(
    client=chroma_client,
    collection_name="Skyminer-T",
    embedding_function=embedding_function,

print("There are", langchain_chroma._collection.count(), "in the collection")
```

## Readings
- [Documenation](https://docs.trychroma.com/) 
- [Github](https://github.com/chroma-core/chroma/tree/main)
