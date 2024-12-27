import random
import requests

class SimpleAgent:
    """Simple agent to interact with WebShop simulation."""
    def __init__(self, base_url):
        self.base_url = base_url
        self.session_id = None
        self.score = 0

    def start_session(self):
        """Start a new session with the simulation."""
        response = requests.get(f"{self.base_url}/start")
        if response.status_code == 200:
            data = response.json()
            self.session_id = data['session_id']
            print(f"Started session: {self.session_id}")
        else:
            print("Failed to start session.")
            exit()

    def search(self, query):
        """Search for products in the simulation."""
        response = requests.get(f"{self.base_url}/search", params={'session_id': self.session_id, 'query': query})
        results = response.json()  # Get the search results as JSON
        print(f"Search results for '{query}':")
    
        for idx, item in enumerate(results['products'], 1):
            # Extract product and score from the response
            product = item['product']  # CHANGE: Extract product details
            score = item['score']      # CHANGE: Extract score

            # Get the pricing value
            price = product.get('pricing', 'N/A')  # Default to 'N/A' if pricing is not available
            if isinstance(price, list):  # If pricing is a list, take the first element
                price = price[0] if price else 'N/A'  # Use 'N/A' if the list is empty

            # Print the product details, including the score
            print(f"{idx}. {product['name']} (Price: {price}, Score: {score})")  # CHANGE: Added score to output
    
        # Return the full product list with scores for further use
        return results['products']



    def buy(self, product_name, product_score):
        """Simulate buying a product by its name and handle the score."""
        response = requests.post(
            f"{self.base_url}/buy",
            json={
                'session_id': self.session_id,
                'product_name': product_name,
                'product_score': product_score  # Send the pre-calculated score
            }
        )
        if response.status_code == 200:
            result = response.json()  # Get the server's response
            # Ensure the server's score matches the agent's expectations
            server_score = result.get('score')
            if server_score != product_score:
                print(f"Warning: Server returned a different score ({server_score}) than expected ({product_score}).")
            # Update the cumulative score
            self.score += product_score
            print(f"Bought product '{product_name}' (Score: {product_score}). Current cumulative score: {self.score}")
        else:
            print(f"Failed to buy the product '{product_name}'.")


    def run(self, queries):
        """Run the agent with a sequence of queries."""
        self.start_session()
        for query in queries:
            products = self.search(query)
            if products:
                # Select the product with the highest score (first in the sorted list)
                chosen_product = products[0]
                product_name = chosen_product['product']['name']
                product_score = chosen_product['score']
                self.buy(product_name, product_score)  # Pass name and score to buy
        print(f"Final cumulative score: {self.score}")

# Example usage
if __name__ == "__main__":
    agent = SimpleAgent(base_url="http://127.0.0.1:3000")
    queries = ["anti aging", "console tables", "tempered glass", "smartwatch"]
    agent.run(queries)
