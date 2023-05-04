# kashmir
A Vector Database Implementation in GoLang

Supports the following APIs:

#### 1. Create a new Vector DB

```
  // This creates an instance of a vector DB. You can create multiple vector DBs as required. 
  db:= NewVectorDB()
```

#### 2. Create a Collection 
```
  // Creates a collection. A collection is a set of one or more documents. 
  err := db.CreateCollection(collectionName)
```

#### 3. Add a Document to the Collection
```
  // Add a Document to the Collection. You can add one or more Documents to a Collection. 
  err = db.AddDocument(collectionName, documentID, document)
```

#### 4. Add a Document to the Collection w/Metadata
```
  // Add a Document to the Collection with Metadata. You can add one or more Documents to a Collection. 
  err = db.AddDocument(collectionName, documentID, document, metadata[])
```

#### 5. Query a Collection for a Document 
```
  // You can query a Collection to find the "closest matching" document to the input "phrase"
	nearestID, err := db.Query(collectionName, phrase)
```

#### 6. Query a Collection for a Document, while filtering by Metadata
```
  // You can query a Collection to find the "closest matching" document to the input "phrase". Only look for documents that match the provided "Metadata"
	nearestID, err := db.Query(collectionName, phrase, metadata)
```
