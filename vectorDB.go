/*
MIT License

Copyright (c) 2023 Sharath Rajasekar, 
Kashmir: Vector DB in GoLang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

package main

import (
	"errors"
	"fmt"
	"math"
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/cockroachdb/pebble"
)

const openAIAPIURL = "https://api.openai.com/v1/embeddings"

/*
 *  Document represents a document in the collection
 */ 
type Document struct {
	ID   string
	Text string
	Embedding []float64
	Metadata map[string]interface{}
}

/*
 * Vector represents a vector of floats
 */ 
type Vector []float64

/*
 * Collection represents a collection of documents
 */ 
type Collection struct {
	name     string
	documents []Document
	vectors   []Vector
}

/*
 * VectorDB represents a database of collections
 */ 
type VectorDB struct {
	db *pebble.DB
}

/*
 * EmbeddingsRequest represents the request payload for the OpenAI Embeddings API
 */ 
type EmbeddingsRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

/*
 * EmbeddingsResponse represents the response payload for the OpenAI Embeddings API
 */ 
type EmbeddingsResponse struct {
	Embedding []float64 `json:"embedding"`
}

/*
 * This function calls OpenAI Embeddings API to generate an embedding for the input text
 * ideally, you want to create the embedding once and store it in a database
 */
func generateEmbedding(inputText string) ([]float64, error) {
	// Define the URL and model name.
	openAIAPIURL := "https://api.openai.com/v1/embeddings"
	modelName := "text-embedding-ada-002"

	// Create the request payload.
	payload := EmbeddingsRequest{
		Input: inputText,
		Model: modelName,
	}

	jsonBody, err := json.Marshal(payload)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
		return nil, err
	}

	// Create an HTTP POST request.
	req, err := http.NewRequest("POST", openAIAPIURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		fmt.Println("Error creating HTTP request:", err)
		return nil, err
	}

	// Set the required headers.
	apiKey := os.Getenv("OPENAI_API_KEY") // Get the API key from an environment variable.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	// Send the request and get the response.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Error sending HTTP request:", err)
		return nil, err
	}
	defer resp.Body.Close()

	// Read and parse the response body.
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Error reading response body:", err)
		return nil, err
	}

	type EmbeddingData struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	}

	type EmbeddingsListResponse struct {
		Object string         `json:"object"`
		Data   []EmbeddingData `json:"data"`
	}

	// Unmarshal the JSON response.
	var embeddingsListResponse EmbeddingsListResponse
	err = json.Unmarshal(body, &embeddingsListResponse)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		return nil, err
	}

	// Extract the first embedding from the response (if available).
	if len(embeddingsListResponse.Data) > 0 {
		embedding := embeddingsListResponse.Data[0].Embedding
		return embedding, nil
	} else {
		fmt.Println("No embeddings found in the response")
		return nil, errors.New("no embeddings found in the response")
	}
}

/*
 * This function creates a new VectorDB
 */ 
func NewVectorDB(dbPath string) (*VectorDB, error) {
	// Open a Pebble DB instance.
	db, err := pebble.Open(dbPath, &pebble.Options{})
	if err != nil {
		fmt.Println("Error opening Pebble DB:", err)
		return nil, err
	}

	return &VectorDB{
		db: db,	
	}, nil
}

/*
 * This function creates a new Collection
 */ 
func NewCollection(name string) *Collection {
	return &Collection{
		name:     name,
		documents: []Document{},
		vectors:   []Vector{},
	}
}

/*
 * This function creates a new Collection
 */ 
func (db *VectorDB) CreateCollection(name string) error {
	// Define the prefix for the keys in the collection.
	prefix := []byte(name + ":")

	// Define the key range for the iterator.
	iterOptions := &pebble.IterOptions{
		LowerBound: prefix,
		UpperBound: []byte(strings.TrimRight(name, ":") + ";"), // Next character after ":"
	}

	// Check if the collection already exists.
	iter := db.db.NewIter(iterOptions)
	defer iter.Close()
	if iter.First() {
		return errors.New("collection already exists")
	}

	// No need to create a collection explicitly in Pebble.
	// Collections are created implicitly when documents are added with the corresponding prefix.
	return nil
}

/*
 * This function adds a document to a collection, include metadata
 */ 
func (db *VectorDB) AddDocument(collectionName, docID, text string, metadata map[string]interface{}) error {
	// Construct the document key using the collection name as a prefix.
	docKey := []byte(collectionName + ":" + docID)

	// Check if the document already exists.
	_, closer, err := db.db.Get(docKey)
	if err == nil {
		closer.Close()
		return errors.New("document already exists")
	} else if err != pebble.ErrNotFound {
		return fmt.Errorf("error checking document existence: %w", err)
	}

	// Generate the embedding for the document text.
	embedding, err := generateEmbedding(text)
	if err != nil {
		return fmt.Errorf("error generating embedding: %w", err)
	}

	// Create the document struct.
	doc := Document{
		ID:       docID,
		Text:     text,
		Embedding: embedding,
		Metadata: metadata,
	}
	fmt.Println("doc:", doc)

	// Serialize the document to JSON.
	docBytes, err := json.Marshal(doc)
	if err != nil {
		return fmt.Errorf("error serializing document: %w", err)
	}

	// Write the document to the Pebble DB.
	err = db.db.Set(docKey, docBytes, pebble.Sync)
	if err != nil {
		return fmt.Errorf("error writing document to Pebble DB: %w", err)
	}

	return nil
}


/*
 * This function adds a list of documents to a collection.
 * Fast concurrent loading of documents using go-routines
 */ 
func (db *VectorDB) AddDocuments(collectionName string, documents []Document) error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(documents))

	for _, doc := range documents {
		wg.Add(1)
		go func(doc Document) {
			defer wg.Done()
			err := db.AddDocument(collectionName, doc.ID, doc.Text, doc.Metadata)
			if err != nil {
				errChan <- err
			}
		}(doc)
	}

	wg.Wait()
	close(errChan)

	// Check for errors from the goroutines.
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}

/*
 * Query with metadata filter
*/
func (db *VectorDB) Query(collectionName string, queryText string, metadataFilter map[string]interface{}) (Document, error) {
	// Generate the embedding for the query text.
	var matchingDoc Document

	queryVec, err := generateEmbedding(queryText)
	if err != nil {
		return matchingDoc, err
	}

	// Define the prefix for the keys in the collection.
	prefix := []byte(collectionName + ":")

	// Initialize variables to keep track of the nearest document.
	//var nearestID string
	maxSimilarity := -1.0

	// Define the key range for the iterator based on the collection name.
	lowerBound := prefix
	upperBound := append(prefix, '\xff')

	// Create an iterator with the specified key range.
	iter := db.db.NewIter(&pebble.IterOptions{
		LowerBound: lowerBound,
		UpperBound: upperBound,
	})

	// Convert metadata filter keys to lowercase.
	for key, value := range metadataFilter {
		delete(metadataFilter, key)
		metadataFilter[strings.ToLower(key)] = value
	}

	for iter.First(); iter.Valid(); iter.Next() {
		// Deserialize the document.
		var doc Document
		err := json.Unmarshal(iter.Value(), &doc)
		if err != nil {
			return matchingDoc, err
		}

		//fmt.Println("doc.Metadata:", doc.Metadata)
		//fmt.Println("metadataFilter:", metadataFilter)
		
		// Check if the document matches the metadata filter.
		matchesFilter := true
		for key, value := range metadataFilter {
			//fmt.Printf("Filter Key: %v, Filter Value: %v, Filter Key Type: %T, Filter Value Type: %T\n", key, value, key, value)
			//fmt.Printf("Doc Metadata Value: %v, Doc Metadata Value Type: %T\n", doc.Metadata[key], doc.Metadata[key])

			//fmt.Println(key, value, doc.Metadata)
			// Convert document metadata keys to lowercase.
			lowercaseKey := strings.ToLower(key)

			//docMetadataValue, ok := doc.Metadata[strings.ToLower(key)]
			//if !ok || docMetadataValue != value {
			if doc.Metadata[lowercaseKey] != value {
				matchesFilter = false
				break
			}
		}
		//fmt.Println("matchesFilter", matchesFilter)

		// If the document matches the filter, calculate its similarity to the query.
		if matchesFilter {
			// Ensure that both vectors have the same non-zero length.
			
			if len(queryVec) > 0 && len(queryVec) == len(doc.Embedding) {
				similarity := cosineSimilarity(queryVec, doc.Embedding)
				if similarity > maxSimilarity {
					maxSimilarity = similarity
					//nearestID = doc.ID
					matchingDoc = doc
				}
			}
		}
	}

	if err := iter.Error(); err != nil {
		return matchingDoc, err
	}

	// Return the ID of the nearest document.
	return matchingDoc, nil
}



/*
 * Helper function to check if a document's metadata matches the metadata filter
*/
func matchesMetadataFilter(metadata map[string]interface{}, metadataFilter map[string]interface{}) bool {
	for key, filterValue := range metadataFilter {
		if metadataValue, ok := metadata[key]; !ok || metadataValue != filterValue {
			return false
		}
	}
	return true
}

/*
 * This function calculates the cosine similarity between two vectors
 */
func cosineSimilarity(a, b Vector) float64 {
	dotProduct := 0.0
	squaredMagnitudeA := 0.0
	squaredMagnitudeB := 0.0

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		squaredMagnitudeA += a[i] * a[i]
		squaredMagnitudeB += b[i] * b[i]
	}

	return dotProduct / (math.Sqrt(squaredMagnitudeA) * math.Sqrt(squaredMagnitudeB))
}

/*
 *	Remove main() function before packaging, Usage Example
 */
 
// Usage example:
func main() {
	// Initialize the VectorDB.
	db, err := pebble.Open("vector-db", &pebble.Options{})
	if err != nil {
		fmt.Println("Error opening Pebble DB:", err)
		return
	}
	vectorDB := &VectorDB{db: db}
	defer db.Close()

	// Define documents to be added.
	documents := []Document{
		{ID: "doc3", Text: "The Manifold on the Moonrings", Metadata: map[string]interface{}{"source": "Notion"}},
		{ID: "doc4", Text: "Bettymore Bought Some MoreButter", Metadata: map[string]interface{}{"source": "Notion"}},
	}

	// Add documents to the VectorDB.
	err = vectorDB.AddDocuments("MyTestCollection", documents)
	if err != nil {
		fmt.Println("Error adding documents:", err)
	}

	// Query the VectorDB.
	//var nearestID string
	queryString := "Moon"
	var matchingDoc Document
	matchingDoc, err = vectorDB.Query("MyTestCollection", queryString, map[string]interface{}{"source": "Notion"})
	if err != nil {
		fmt.Println("Error querying VectorDB:", err)
	}

	fmt.Printf("Nearest document to query %s is ID: %s : %s \n", queryString, matchingDoc.ID, matchingDoc.Text)
}
