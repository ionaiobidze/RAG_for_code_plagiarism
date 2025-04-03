# Code Plagiarism Detector using RAG and Google Drive

## Overview

This project implements a code plagiarism detection system using a Retrieval-Augmented Generation (RAG) approach. It indexes specified code repositories stored on Google Drive, chunks the code files based on token limits, generates vector embeddings using a sentence-transformer model, and stores these embeddings in a ChromaDB vector database.

When provided with a code snippet, the system:
1.  Generates an embedding for the input snippet.
2.  Queries the ChromaDB database to find the most similar code chunks from the indexed repositories.
3.  Constructs a prompt containing the user's snippet and the retrieved code chunks.
4.  Sends this prompt to an OpenAI LLM (GPT-4o) to assess the likelihood of plagiarism.
5.  Returns a JSON response indicating whether plagiarism is likely, along with reasoning and references to the potential source files.

The system is designed to run within a Google Colab environment, leveraging Google Drive for source code access and Colab Secrets for API key management.

## Features

*   **Google Drive Integration:** Indexes code files directly from specified directories in your Google Drive.
*   **Configurable Indexing:** Allows specifying repository paths and file extensions (e.g., `.js`) to index.
*   **Token-Based Chunking:** Splits code files into manageable, overlapping chunks based on token count for effective embedding.
*   **Vector Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for generating dense vector representations of code chunks.
*   **Vector Database:** Utilizes ChromaDB for efficient similarity search of code embeddings.
*   **RAG Pipeline:** Retrieves relevant code context from the database before analysis.
*   **LLM Analysis:** Leverages OpenAI's `gpt-4o` model for intelligent plagiarism assessment based on retrieved context.
*   **JSON Output:** Provides structured output including plagiarism likelihood, reasoning, and source references.
*   **Batch Processing:** Implements batching for efficient embedding generation and database insertion during indexing.
*   **GPU Acceleration:** Automatically utilizes GPU if available in the Colab environment.

## How it Works

1.  **Configuration:** Define Google Drive paths containing code, target file extensions, embedding/LLM models, and database settings.
2.  **File Discovery:** Scan the specified Google Drive directories for code files matching the configured extensions.
3.  **Chunking:** Read each discovered file and split its content into overlapping chunks, ensuring each chunk respects the maximum token limit.
4.  **Embedding:** Generate vector embeddings for each code chunk using the sentence-transformer model.
5.  **Indexing:** Store the chunks, their embeddings, and metadata (repository name, file path, chunk index) in the ChromaDB collection. *Note: The current implementation clears the database collection before each indexing run.*
6.  **Plagiarism Check (RAG):**
    *   Embed the user-provided code snippet.
    *   Query ChromaDB using the snippet's embedding to find the `k` most similar indexed chunks (context).
    *   Format a prompt containing the user snippet and the retrieved context.
    *   Send the prompt to the configured OpenAI LLM (GPT-4o).
    *   Parse the LLM's JSON response containing the plagiarism decision and reasoning.

## Technology Stack

*   **Language:** Python 3
*   **Core Libraries:**
    *   `torch`: For deep learning framework support.
    *   `transformers`: For accessing pre-trained models (embedding model).
    *   `sentence-transformers`: Specific library for sentence/text embeddings.
    *   `openai`: Official client library for OpenAI API interaction.
    *   `chromadb`: Vector database for storing and querying embeddings.
    *   `pandas`: (Used in dependency list, potentially for future data handling).
    *   `PyYAML`: (Used in dependency list, potentially for future config loading).
*   **Environment:** Google Colab
*   **Services:**
    *   Google Drive: For storing source code repositories.
    *   OpenAI API: For accessing the LLM (GPT-4o).

## Setup and Configuration

This project is designed to run in Google Colab.

**Prerequisites:**

1.  **Google Account:** Required for Google Colab and Google Drive.
2.  **Google Drive:** Your source code repositories must be accessible within your Google Drive.
3.  **OpenAI API Key:** You need an API key from OpenAI.

**Steps:**

1.  **Open in Colab:** Open the `.ipynb` notebook file in Google Colab.
2.  **Mount Google Drive:** Run the necessary Colab cell to mount your Google Drive. Ensure you grant permissions.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  **Set OpenAI API Key:**
    *   In Colab, navigate to the "Secrets" tab (key icon on the left sidebar).
    *   Add a new secret named `OPENAI_API_KEY`.
    *   Paste your OpenAI API key as the value.
    *   Ensure the "Notebook access" toggle is enabled for this secret.
4.  **Configure Paths and Settings (Cell 1):**
    *   **`DRIVE_REPO_PATHS`**: **Crucially**, update this list with the *exact paths* within your `/content/drive/MyDrive/` where your target code repositories reside. The script will scan these directories.
        ```python
        DRIVE_REPO_PATHS = [
            "/content/drive/MyDrive/path/to/your/repo1",
            "/content/drive/MyDrive/another/project/repo2",
            # Add more paths as needed
        ]
        ```
    *   **`PROGRAMMING_LANGUAGES`**: Modify the list of file extensions if you need to index languages other than `.js`.
        ```python
        PROGRAMMING_LANGUAGES = [".js", ".py", ".jsx"] # Example
        ```
    *   Review other constants like `EMBEDDING_MODEL_NAME`, `LLM_MODEL_NAME`, `COLLECTION_NAME`, `DB_PATH`, chunking parameters (`MAX_CHUNK_LENGTH_TOKENS`, `CHUNK_OVERLAP_TOKENS`), `RAG_NUM_RESULTS`, and `INDEXING_BATCH_SIZE` if needed. The defaults are generally reasonable.

## Usage

Run the Colab notebook cells sequentially:

1.  **Cell 1: Setup and Initialization:**
    *   Installs required dependencies (`pip install ...`).
    *   Imports necessary libraries.
    *   Loads the OpenAI API key from Colab Secrets.
    *   Configures paths, models, and other parameters.
    *   Checks for GPU availability.
    *   Loads the embedding model and tokenizer.
    *   Initializes the OpenAI client.
    *   Initializes the ChromaDB client and collection.
    *   Prints status messages and warnings/errors if setup fails.
    *   **Verify:** Check the output for any critical errors (e.g., API key missing, model loading failed, DB connection failed).

2.  **Cell 2: Function Definitions:**
    *   Defines the core Python functions (`get_embeddings_batch`, `chunk_code_file`, `find_code_files`, `index_code_files`, `check_plagiarism`).
    *   This cell only defines the functions; it doesn't execute the main logic yet.

3.  **Cell 3: Run Indexing Process:**
    *   **Scans:** Finds eligible code files in the configured `DRIVE_REPO_PATHS`.
    *   **Clears DB:** *Important:* This step currently **deletes and recreates** the ChromaDB collection (`COLLECTION_NAME`) to ensure a fresh index. Any previously indexed data will be lost.
    *   **Indexes:** Chunks, embeds, and adds the code chunks to the database in batches.
    *   **Reports:** Prints progress, timings, and final statistics (number of files processed, chunks added, final DB count). This step can take time depending on the size of your repositories and whether a GPU is available.

4.  **Cell 4: Test Plagiarism Checker:**
    *   Provides example code snippets.
    *   Calls the `check_plagiarism` function for each test snippet.
    *   Prints the detailed JSON results from the plagiarism check.
    *   Use this cell as a template to test your own code snippets:
        ```python
        my_code = """
        // Paste your code snippet here
        function example() {
            console.log("Hello, world!");
        }
        """
        result = check_plagiarism(my_code)
        print(json.dumps(result, indent=2))
        ```

## Key Code Components

*   `find_code_files(...)`: Locates relevant source files in Google Drive.
*   `chunk_code_file(...)`: Splits file content into token-based chunks.
*   `get_embeddings_batch(...)`: Generates embeddings for text batches using the transformer model.
*   `index_code_files(...)`: Orchestrates finding, chunking, embedding, and storing files in ChromaDB.
*   `check_plagiarism(...)`: The main RAG function that takes a snippet, queries the DB, calls the LLM, and returns the plagiarism assessment.

## Limitations and Considerations

*   **Colab Dependency:** Relies heavily on the Google Colab environment, secrets management, and Google Drive integration. Running locally would require significant modifications.
*   **Database Persistence:** ChromaDB is configured as `PersistentClient`, storing data within the Colab instance's temporary storage (`/content/temp_chroma_db_drive`). This data will be lost when the Colab runtime disconnects unless saved/backed up explicitly (e.g., copying the DB directory to Drive).
*   **Indexing Time:** Indexing large codebases can be time-consuming, especially without a GPU.
*   **Database Clearing:** The current indexing process clears the entire collection before starting. Modify `index_code_files` if incremental updates are needed.
*   **Cost:** Uses the OpenAI API, which incurs costs based on token usage for both the prompt and the completion.
*   **Accuracy:** The accuracy depends on the quality of the embeddings, the effectiveness of the RAG retrieval, the capabilities of the LLM (GPT-4o), and the prompt design. It identifies *similarity*, which may or may not equate to plagiarism in all contexts. Human review is recommended.
*   **Token Limits:** Code is chunked based on token limits. Very large functions or files might be split across multiple chunks, potentially impacting context understanding.