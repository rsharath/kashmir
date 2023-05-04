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
)

const openAIAPIURL = "https://api.openai.com/v1/embeddings"

/*
 *  Document represents a document in the collection
 */ 
type Document struct {
	ID   string
	Text string
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
	collections map[string]*Collection
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
		return nil, err
	}

	// Extract the first embedding from the response (if available).
	if len(embeddingsListResponse.Data) > 0 {
		embedding := embeddingsListResponse.Data[0].Embedding
		//fmt.Printf("Embedding: %v\n", embedding)
		return embedding, nil
	} else {
		return nil, errors.New("no embeddings found in the response")
	}
}

/*
 * This function creates a new VectorDB
 */ 
func NewVectorDB() *VectorDB {
	return &VectorDB{
		collections: make(map[string]*Collection),
	}
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
	if _, exists := db.collections[name]; exists {
		return errors.New("collection already exists")
	}
	db.collections[name] = NewCollection(name)
	return nil
}

/*
 * This function adds a document to a collection
 */ 
func (db *VectorDB) AddDocument(collectionName, docID, text string) error {
	collection, exists := db.collections[collectionName]
	if !exists {
		return errors.New("collection not found")
	}
	
	collection.documents = append(collection.documents, Document{ID: docID, Text: text})

	// Call the generateEmbedding function to get the embedding and error.
	embedding, err := generateEmbedding(text)
	if err != nil {
		// Handle the error (e.g., return the error or print an error message).
		return err
	}

	// Append the generated embedding to the collection's vectors.
	collection.vectors = append(collection.vectors, embedding)

	return nil
}

/*
 * This function adds a document to a collection, include metadata
 */ 
func (db *VectorDB) AddDocumentWithMetadata(collectionName, docID, text string, metadata map[string]interface{}) error {
	collection, exists := db.collections[collectionName]
	if !exists {
		fmt.Println("Collection not found")
		return errors.New("collection not found")
	}
 
	doc := Document{
		ID:       docID,
		Text:     text,
		Metadata: metadata,
	}
	collection.documents = append(collection.documents, doc)
	
	// Call the generateEmbedding function to get the embedding and error.
	embedding, err := generateEmbedding(text)
	if err != nil {
		// Handle the error (e.g., return the error or print an error message).
		fmt.Println("Error generating embedding:", err)
		return err
	}

	// Append the generated embedding to the collection's vectors.
	collection.vectors = append(collection.vectors, embedding)

	return nil
}

/*
 * This function adds a list of documents to a collection.
 * Fast concurrent loading of documents using go-routines
 */ 
func (db *VectorDB) AddDocuments(collectionName string, documents []Document) error {
	collection, exists := db.collections[collectionName]
	if !exists {
		return errors.New("collection not found")
	}

	// Create a channel to receive the results of the concurrent API calls.
	results := make(chan []float64, len(documents))
	errors := make(chan error, len(documents))

	// Define a function to generate embeddings concurrently.
	generateEmbeddingConcurrent := func(text string, results chan []float64, errors chan error) {
		embedding, err := generateEmbedding(text)
		if err != nil {
			errors <- err
		} else {
			results <- embedding
		}
	}

	// Launch goroutines to generate embeddings for each document.
	for _, doc := range documents {
		go generateEmbeddingConcurrent(doc.Text, results, errors)
	}

	// Collect the results and errors from the goroutines.
	for i := 0; i < len(documents); i++ {
		select {
		case embedding := <-results:
			collection.vectors = append(collection.vectors, embedding)
		case err := <-errors:
			return err
		}
	}

	// Append the documents to the collection.
	collection.documents = append(collection.documents, documents...)

	return nil
}

/*
 *	This function queries a collection and returns the ID of the document with the closest embedding
 */
func (db *VectorDB) Query(collectionName string, queryText string) (string, error) {
	collection, exists := db.collections[collectionName]
	if !exists {
		return "", errors.New("collection not found")
	}

	queryVec, err := generateEmbedding(queryText)
	nearestID, err := findNearestNeighbor(queryVec, collection.vectors, collection.documents)

	return nearestID, err
}

/*
 * This function finds the nearest neighbor of a query vector in a collection of vectors
 */
func findNearestNeighbor(queryVec Vector, vectors []Vector, documents []Document) (string, error) {
	if len(vectors) == 0 {
		return "", errors.New("collection is empty")
	}

	maxSimilarity := -1.0
	nearestID := ""

	for i, vec := range vectors {
		similarity := cosineSimilarity(queryVec, vec)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
			nearestID = documents[i].ID
		}
	}

	return nearestID, nil
}

/*
 * Query with metadata filter
*/
func (db *VectorDB) QueryWithMetadata(collectionName string, queryText string, metadataFilter map[string]interface{}) (string, error) {
	collection, exists := db.collections[collectionName]
	if !exists {
		return "", errors.New("collection not found")
	}

	queryVec, err := generateEmbedding(queryText)
	nearestID, err := findNearestNeighborWithMetadata(queryVec, collection.vectors, collection.documents, metadataFilter)
	return nearestID, err
}

/*
 * find nearest neighbor after filtering by the metadata filter
*/
func findNearestNeighborWithMetadata(queryVec Vector, vectors []Vector, documents []Document, metadataFilter map[string]interface{}) (string, error) {
	if len(vectors) == 0 {
		return "", errors.New("collection is empty")
	}

	maxSimilarity := -1.0
	nearestID := ""

	for i, vec := range vectors {
		// Check if the document's metadata matches the metadata filter
		if !matchesMetadataFilter(documents[i].Metadata, metadataFilter) {
			continue
		}

		similarity := cosineSimilarity(queryVec, vec)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
			nearestID = documents[i].ID
		}
	}

	return nearestID, nil
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
 *	Remove main() function before packaging
 */
func main() {
	db := NewVectorDB()
	collectionName := "articles"

	// Create a collection
	fmt.Printf("Creating a collection...[%s]", collectionName)
	err := db.CreateCollection(collectionName)
	if err != nil {
		fmt.Println("Error creating collection:", err)
		return
	}

	// Add documents to the collection
	err = db.AddDocument(collectionName, "doc1", "nothing burger")
	if err != nil {
		fmt.Println("Error adding document - doc1:", err)
		return
	}

	err = db.AddDocument(collectionName, "doc2", "Another example document.")
	if err != nil {
		fmt.Println("Error adding document: doc2", err)
		return
	}

	// Query the collection
	queryText := "This is a burger"
	nearestID, err := db.Query(collectionName, queryText)
	if err != nil {
		fmt.Println("Error querying the collection:", err)
		return
	}

	fmt.Printf("\nThe closest document to the query \"%s\" is document with ID \"%s\".\n", queryText, nearestID)
}

