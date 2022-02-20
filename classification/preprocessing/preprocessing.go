// This script produces word embeddings for a corpus using GloVe

package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/korovkin/limiter"
	"github.com/schollz/progressbar/v3"
)

type Embedding []float32

// The maxium number of words in a review to use
const max_tokens = 200

// Regex to match html tags
var html_regex = regexp.MustCompile("<[^>]*>")

// Regex to match non-alphanumeric characters except for dashes and spaces
var non_alpha_regex = regexp.MustCompile("[^a-zA-Z0-9\\- ]")

// Load the glove embeddings as a map
func load_glove(filename string, embedding_size int) map[string]Embedding {
	embeddings := make(map[string]Embedding)
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		tokens := strings.Split(line, " ")
		embeddings[tokens[0]] = make(Embedding, embedding_size)
		for i := 1; i <= embedding_size; i++ {
			num, _ := strconv.ParseFloat(tokens[i], 32)
			embeddings[tokens[0]][i-1] = float32(num)
		}
	}
	return embeddings
}

// Embed a review using GloVe
func embed_review(review string, embeddings map[string]Embedding, embedding_size int) []Embedding {
	// Remove html tags
	review = html_regex.ReplaceAllString(review, "")
	// Remove non-alphanumeric characters except for dashes and spaces
	review = non_alpha_regex.ReplaceAllString(review, " ")
	// Replace all dashes and spaces with a single space
	review = strings.Replace(review, "-", " ", -1)
	review = strings.Replace(review, " ", " ", -1)
	// Convert to lowercase
	review = strings.ToLower(review)
	// Split the review into words
	words := strings.Split(review, " ")
	// Truncate the review to max_tokens
	if len(words) > max_tokens {
		words = words[:max_tokens]
	}
	// Initialize the embedding vector
	review_embedding := make([]Embedding, max_tokens)
	// For each word in the review
	for i, word := range words {
		// If the word is in the embeddings map
		if word_embedding, ok := embeddings[word]; ok {
			// Add the embedding to the review embedding
			review_embedding[i] = word_embedding
		} else {
			// Otherwise, add a zero vector to represent the OOV word
			review_embedding[i] = make(Embedding, embedding_size)
		}
	}
	// Pad the embedding vector with zeros
	for i := len(words); i < max_tokens; i++ {
		review_embedding[i] = make(Embedding, embedding_size)
	}

	return review_embedding
}

func writer(output_file string, output_channel <-chan []Embedding) {
	// Open the output file
	file, err := os.Create(output_file)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Write the embeddings to the file
	for embedded_review := range output_channel {
		for _, word_embedding := range embedded_review {
			binary.Write(file, binary.LittleEndian, word_embedding)
		}
	}
}

func embed_dataset(data_dir string, output_file string, embeddings map[string]Embedding, embedding_size int) {
	// Create the output channel
	output_channel := make(chan []Embedding)

	// Create wait group
	var wg sync.WaitGroup

	// Create the writer goroutine
	wg.Add(1)
	go func() {
		writer(output_file, output_channel)
		wg.Done()
	}()

	// Create a worker pool
	limit := limiter.NewConcurrencyLimiter(250)

	// Create a progress bar
	bar := progressbar.Default(-1)

	// Walk the data directory
	filepath.WalkDir(data_dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			panic(err)
		}

		if strings.HasSuffix(path, ".txt") {
			// Open the file
			review, err := os.ReadFile(path)
			if err != nil {
				panic(err)
			}

			limit.Execute(func() {
				// Embed the review
				embedded_review := embed_review(string(review), embeddings, embedding_size)
				// Send the embedding to the output channel
				output_channel <- embedded_review
				bar.Add(1)
			})
		}

		return nil
	})

	// Wait for the workers to finish
	limit.Wait()
	// Close the output channel
	close(output_channel)
	wg.Wait()
}

func main() {
	// Load the glove embeddings
	fmt.Println("Loading 50-dimensional GloVe embeddings...")
	embeddings := load_glove("data/glove_embeddings/glove.6B.50d.txt", 50)
	fmt.Println("Loaded GloVe embeddings.")

	fmt.Println("Embedding dataset with 50 dims...")
	fmt.Println("Positive training data...")
	embed_dataset("data/train/pos", "data/embedded/train_pos_50d.bin", embeddings, 50)
	fmt.Println()
	fmt.Println("Negative training data...")
	embed_dataset("data/train/neg", "data/embedded/train_neg_50d.bin", embeddings, 50)
	fmt.Println()
	fmt.Println("Positive test data...")
	embed_dataset("data/test/pos", "data/embedded/test_pos_50d.bin", embeddings, 50)
	fmt.Println()
	fmt.Println("Negative test data...")
	embed_dataset("data/test/neg", "data/embedded/test_neg_50d.bin", embeddings, 50)
	fmt.Println()

	fmt.Println("Loading 100-dimensional GloVe embeddings...")
	embeddings = load_glove("data/glove_embeddings/glove.6B.100d.txt", 100)
	fmt.Println("Loaded GloVe embeddings.")

	fmt.Println("Embedding dataset with 100 dims...")
	fmt.Println("Positive training data...")
	embed_dataset("data/train/pos", "data/embedded/train_pos_100d.bin", embeddings, 100)
	fmt.Println()
	fmt.Println("Negative training data...")
	embed_dataset("data/train/neg", "data/embedded/train_neg_100d.bin", embeddings, 100)
	fmt.Println()
	fmt.Println("Positive test data...")
	embed_dataset("data/test/pos", "data/embedded/test_pos_100d.bin", embeddings, 100)
	fmt.Println()
	fmt.Println("Negative test data...")
	embed_dataset("data/test/neg", "data/embedded/test_neg_100d.bin", embeddings, 100)
	fmt.Println()

	fmt.Println("Loading 200-dimensional GloVe embeddings...")
	embeddings = load_glove("data/glove_embeddings/glove.6B.200d.txt", 200)
	fmt.Println("Loaded GloVe embeddings.")

	fmt.Println("Embedding dataset with 200 dims...")
	fmt.Println("Positive training data...")
	embed_dataset("data/train/pos", "data/embedded/train_pos_200d.bin", embeddings, 200)
	fmt.Println()
	fmt.Println("Negative training data...")
	embed_dataset("data/train/neg", "data/embedded/train_neg_200d.bin", embeddings, 200)
	fmt.Println()
	fmt.Println("Positive test data...")
	embed_dataset("data/test/pos", "data/embedded/test_pos_200d.bin", embeddings, 200)
	fmt.Println()
	fmt.Println("Negative test data...")
	embed_dataset("data/test/neg", "data/embedded/test_neg_200d.bin", embeddings, 200)
	fmt.Println()

	fmt.Println("Loading 300-dimensional GloVe embeddings...")
	embeddings = load_glove("data/glove_embeddings/glove.6B.300d.txt", 300)
	fmt.Println("Loaded GloVe embeddings.")

	fmt.Println("Embedding dataset with 300 dims...")
	fmt.Println("Positive training data...")
	embed_dataset("data/train/pos", "data/embedded/train_pos_300d.bin", embeddings, 300)
	fmt.Println()
	fmt.Println("Negative training data...")
	embed_dataset("data/train/neg", "data/embedded/train_neg_300d.bin", embeddings, 300)
	fmt.Println()
	fmt.Println("Positive test data...")
	embed_dataset("data/test/pos", "data/embedded/test_pos_300d.bin", embeddings, 300)
	fmt.Println()
	fmt.Println("Negative test data...")
	embed_dataset("data/test/neg", "data/embedded/test_neg_300d.bin", embeddings, 300)
	fmt.Println()

	fmt.Println("Done!")
}
