import pandas as pd
import chromadb
import uuid

class Portfolio:
    def __init__(self, file_path="app/resource/portfolio.csv"):
        # Load the CSV file containing the portfolio data
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
        # Initialize a persistent ChromaDB client to store and query data
        self.chroma_client = chromadb.PersistentClient(path="vectorstore")
        
        # Create or get a collection named "portfolio" in ChromaDB
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        # Check if the collection is already populated
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                # Add documents (tech stack) and associated metadata (links) into the ChromaDB collection
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas=[{"links": row["Links"]}],  # Make sure metadatas is a list of dictionaries
                    ids=[str(uuid.uuid4())]  # Generate a unique ID for each entry
                )

    def query_links(self, skills):
        """
        Query the ChromaDB collection based on a list of skills and return matching portfolio links.
        
        :param skills: A list of skills related to the job requirements
        :return: A list of relevant portfolio links
        """
        try:
            # Perform a query in ChromaDB with the input skills and retrieve 2 most relevant links
            query_result = self.collection.query(query_texts=skills, n_results=2)
            
            # Debugging: Check the structure of query_result
            print("DEBUG: query_result:", query_result)

            # Ensure 'metadatas' is a list and extract 'links' from each item
            if 'metadatas' in query_result:
                return [metadata.get('links', 'No link available') for metadata in query_result['metadatas']]
            else:
                return []

        except Exception as e:
            print(f"Error in querying portfolio: {e}")
            return []
