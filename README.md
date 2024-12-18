# AI RAG Knowledge base app for Cyberbullying

## Installation

### Packages

- Please install the packages based on requirements.txt.

```bash
pip install -r requirements.txt
```
- Rename `env.example.txt` file to .env and add your `OPENAI_API_KEY`.


## Storing Data to Chroma DB for RAG
To extract the contents of the pdf file to store in the vector DB, run

```bash
python run rag_store.py
```

## Running the App
```bash
python run main.py
```

