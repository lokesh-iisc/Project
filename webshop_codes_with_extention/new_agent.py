import requests
import json
from bs4 import BeautifulSoup

class SimpleAgent:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session_id = None
        self.score = 0

    def start_session(self):
        """Start a new session."""
        # Use a fixed session ID, as required by app.py
        self.session_id = "abc"

        # Initialize the session on the server
        response = requests.get(f"{self.base_url}/{self.session_id}")
        if response.status_code == 200:
            print(f"Started session with ID: {self.session_id}")
        else:
            print(f"Failed to initialize session with ID: {self.session_id}. Server responded with: {response.status_code}")


    def search(self, query):
        """Perform a search query."""
        keywords = query.lower().split(" ")
        
        response = requests.get(
            f"{self.base_url}/search_results/{self.session_id}/{keywords}/1"
        )

        if response.status_code != 200:
            print(f"Search request failed: {response.status_code}")
            return []

        try:
            html = response.text
            print(f"Search results for '{query}': Retrieved HTML")
            return self.parse_products_from_html(html)
        except Exception as e:
            print(f"Error parsing search results: {e}")
            return []

    

    def parse_products_from_html(self, html):
        """Parse products from the search results HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        products = []

        # Find all product items in the HTML
        product_elements = soup.find_all('div', class_='list-group-item')

        for element in product_elements:
            try:
                # Extract ASIN from the product link
                asin = element.find('h4', class_='product-asin').a.text.strip()

                # Extract product title
                title = element.find('h4', class_='product-title').text.strip()

                try:
                    # Extract price and handle ranges
                    price_text = element.find('h5', class_='product-price').text.strip()
                    if "to" in price_text:
                    # Split range and take the average price
                        lower, upper = map(lambda x: float(x.replace('$', '').replace(',', '')), price_text.split("to"))
                        price = (lower + upper) / 2
                    else:
                    # Single price
                        price = float(price_text.replace('$', '').replace(',', ''))
                except ValueError:
                    print(f"Error parsing price: {price_text}")
                    price = None

                # Extract main image URL
                image_url = element.find('img', class_='result-img')['src']

                # Append the product details to the list
                products.append({
                    "asin": asin,
                    "name": title,
                    "price": price,
                    "image": image_url,
                })
            except AttributeError as e:
                print(f"Error parsing product element: {e}")

        return products

    def view_product(self, asin, query, page, options):
        """View product details."""
        response = requests.get(
            f"{self.base_url}/item_page/{self.session_id}/{asin}/{query}/{page}/{options}"
        )
        if response.status_code != 200:
            print(f"Failed to view product: {response.status_code}")
            return None

        html = response.text
        print(f"Viewed product details for ASIN: {asin}")
        return self.parse_product_details_from_html(html)

    def parse_product_details_from_html(self, html):
        """Parse product details from the item page HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        try:
            # Extract product image
            image_url = soup.find('img', id='product-image')['src']

            # Extract product title
            title = soup.find('h2').text.strip()

            try:
                # Extract price and handle ranges
                price_text = soup.find('h4', text=lambda x: x and "Price:" in x).text.strip()
                price_text = price_text.replace('Price:', '').strip()
                if "to" in price_text:
                    # Split range and take the average price
                    lower, upper = map(lambda x: float(x.replace('$', '').replace(',', '')), price_text.split("to"))
                    price = (lower + upper) / 2
                else:
                    # Single price
                    price = float(price_text.replace('$', '').replace(',', ''))
            except ValueError:
                print(f"Error parsing price: {price_text}")
                price = None

            # Extract product rating
            try:
                rating_text = soup.find('h4', text=lambda x: x and "Rating:" in x).text.strip()
                rating = float(rating_text.replace('Rating:', '').strip())
            except (ValueError, AttributeError):
                print(f"Error parsing rating: {rating_text}")
                rating = None  # Default to None if not available


            # Extract product options
            options = {}
            for option_section in soup.find_all('div', class_='radio-toolbar'):
                option_name = option_section.find_previous('h4').text.strip()
                option_values = [
                    label.text.strip()
                    for label in option_section.find_all('label')
                ]
                options[option_name] = option_values

            return {
                "image_url": image_url,
                "title": title,
                "price": price,
                "rating": rating,
                "options": options,
            }
        except AttributeError as e:
            print(f"Error parsing product details: {e}")
            return None

        
    def view_item_sub_page(self, asin, query, page, sub_page, options):
        """View a sub-page of the product."""
        response = requests.get(
            f"{self.base_url}/item_sub_page/{self.session_id}/{asin}/{query}/{page}/{sub_page}/{options}"
        )
        if response.status_code != 200:
            print(f"Failed to view sub-page: {response.status_code}")
            return None

        html = response.text
        print(f"Viewed sub-page {sub_page} for ASIN: {asin}")

        # Determine which parser to use based on the sub_page
        if sub_page == "Attributes":
            return self.parse_attributes_page(html)
        elif sub_page == "Features":
            return self.parse_features_page(html)
        elif sub_page == "Reviews":
            return self.parse_review_page(html)
        elif sub_page == "Description":
            return self.parse_description_page(html)
        else:
            print(f"Unknown sub-page type: {sub_page}")
            return None

    def parse_attributes_page(self, html):
        """Parse attributes from attributes_page.html."""
        soup = BeautifulSoup(html, 'html.parser')
        try:
            attributes = [
                attribute.text.strip()
                for attribute in soup.find_all('p', class_='attribute')
            ]
            category = soup.find('h5', class_='product-category').text.strip()
            query = soup.find('h5', class_='product-query').text.strip()
            product_category = soup.find('h5', class_='product-product_category').text.strip()

            return {
                "type": "Attributes",
                "attributes": attributes,
                "category": category,
                "query": query,
                "product_category": product_category
            }
        except AttributeError as e:
            print(f"Error parsing attributes page: {e}")
            return None
        
    def parse_features_page(self, html):
        """Parse features from features_page.html."""
        soup = BeautifulSoup(html, 'html.parser')
        try:
            features = [
                feature.text.strip()
                for feature in soup.find_all('p', class_='product-info')
            ]
            return {"type": "Features", "features": features}
        except AttributeError as e:
            print(f"Error parsing features page: {e}")
            return None


    def parse_review_page(self, html):
        """Parse reviews from review_page.html."""
        soup = BeautifulSoup(html, 'html.parser')
        try:
            reviews = []
            for review_card in soup.find_all('div', class_='card'):
                title = review_card.find('h4', class_='blue-text').text.strip()
                score = int(review_card.find('span').text.strip())
                body = review_card.find('p', class_='content').text.strip()
                reviews.append({"title": title, "score": score, "body": body})

            return {"type": "Reviews", "reviews": reviews}
        except AttributeError as e:
            print(f"Error parsing review page: {e}")
            return None

    def parse_description_page(self, html):
        """Parse description from description_page.html."""
        soup = BeautifulSoup(html, 'html.parser')
        try:
            description = soup.find('p', class_='product-info').text.strip()
            return {"type": "Description", "description": description}
        except AttributeError as e:
            print(f"Error parsing description page: {e}")
            return None

    

    def buy_product(self, asin, options):
        """Simulate buying a product."""
        response = requests.get(
            f"{self.base_url}/done/{self.session_id}/{asin}/{options}"
        )
        if response.status_code == 200:
            html = response.text
            print(f"Bought product with ASIN: {asin}")
            return self.parse_done_page(html)
        else:
            print(f"Failed to buy product with ASIN: {asin}")
            return None
        
    def parse_done_page(self, html):
        """Parse the confirmation details from the done_page HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        try:
            # Extract thank-you message
            thank_you_message = soup.find('h1', id='thankyou')
            thank_you_message = thank_you_message.text.strip() if thank_you_message else "Thank you!"

            # Extract MTurk code
            mturk_code = soup.find('pre')
            mturk_code = mturk_code.text.strip() if mturk_code else "N/A"

            # Extract reward score
            reward_text = soup.find('h3', id='reward')
            reward_text = reward_text.find('pre').text.strip() if reward_text and reward_text.find('pre') else "0"
            reward = float(reward_text) if reward_text.replace('.', '', 1).isdigit() else 0

            # Extract product details (if visible)
            asin = soup.find('h4', id='asin')
            asin = asin.find('pre').text.strip() if asin and asin.find('pre') else "Unknown"

            options = soup.find('h4', id='options')
            options = options.find('pre').text.strip() if options and options.find('pre') else "{}"

            purchased_attrs = soup.find('h4', id='purchased_attrs')
            purchased_attrs = purchased_attrs.find('pre').text.strip() if purchased_attrs and purchased_attrs.find('pre') else "N/A"

            category = soup.find('h4', id='purchased-category')
            category = category.find('pre').text.strip() if category and category.find('pre') else "N/A"

            query = soup.find('h4', id='purchased-query')
            query = query.find('pre').text.strip() if query and query.find('pre') else "N/A"

            product_category = soup.find('h4', id='purchased-pc')
            product_category = product_category.find('pre').text.strip() if product_category and product_category.find('pre') else "N/A"

            return {
                "thank_you_message": thank_you_message,
                "mturk_code": mturk_code,
                "reward": reward,
                "product_details": {
                    "asin": asin,
                    "options": options,
                    "purchased_attrs": purchased_attrs,
                    "category": category,
                    "query": query,
                    "product_category": product_category,
                }
            }
        except Exception as e:
            print(f"Error parsing done page: {e}")
            return None



    def run(self, queries):
        """Run the agent with a sequence of queries."""
        self.start_session()

        for query in queries:
            # Step 1: Search for products
            products = self.search(query)
            if products:
                # Step 2: Select the top product
                top_product = products[0]  # Pick the top product
                asin = top_product["asin"]
                product_name = top_product.get("name", "Unknown Product")
                product_price = top_product.get("price", "N/A")
                options = top_product.get("options", {})  # Extract options if available
                print(f"Selected product: {product_name} (ASIN: {asin}, Price: {product_price})")
                print(f"Options: {options}")

                # Step 3: View product details
                product_details = self.view_product(asin, query, 1,options)
                if product_details:
                    options = product_details.get("options", {})
                    print(f"Product details: {product_details}")

                    # Step 4: View a sub-page (e.g., Description)
                    sub_page = "Description"  # Example sub-page
                    sub_page_details = self.view_item_sub_page(asin, query, 1, sub_page, options)
                    print(f"Sub-page details ({sub_page}): {sub_page_details}")

                    # Step 5: Purchase the product
                    print(f"Attempting to buy product: {product_name} (ASIN: {asin}) with options: {options}")
                    purchase_result = self.buy_product(asin, options)
                    if purchase_result:
                        print(f"Purchase successful: {purchase_result}")
                    else:
                        print(f"Purchase failed for product: {product_name} (ASIN: {asin})")

        print("Agent finished all tasks.")

if __name__ == "__main__":
    agent = SimpleAgent(base_url="http://10.217.49.21:3000")  # Update base_url
    queries = [
        #"Queen bed",
        "i am looking for a queen sized bed that is black, and price lower than 140.00 dollars",
        #"Looking for a white ceramic dinner set under $200",
        #"Find me a silver smartphone case above $25 with a minimalist design",
    ]
    agent.run(queries)





# ############################################
# import requests
# import json


# class SimpleAgent:
#     def __init__(self, base_url):
#         self.base_url = base_url
#         self.session_id = None
#         self.score = 0

#     def start_session(self):
#         """Start a new session."""
#         self.session_id = "abc"  # Fixed session ID as per the server logic
#         print(f"Started session with ID: {self.session_id}")

#     def search(self, query):
#         """Perform a search query."""
#         keywords = query.lower().split(" ")
#         response = requests.get(
#             f"{self.base_url}/search_results/{self.session_id}/{keywords}/1"
#         )

#         if response.status_code != 200:
#             print(f"Search request failed: {response.status_code}")
#             return []

#         try:
#             html = response.text
#             print(f"Search results for '{query}': Retrieved HTML")
#             return self.parse_products_from_html(html)
#         except Exception as e:
#             print(f"Error parsing search results: {e}")
#             return []

#     def parse_products_from_html(self, html):
#         """Parse products from the HTML (placeholder)."""
#         # Implement this function based on your HTML structure
#         return [
#             {"asin": "B01N5IB20Q", "name": "Product 1", "price": 50, "score": 10},
#             {"asin": "B07R8N4L8B", "name": "Product 2", "price": 100, "score": 8},
#         ]

#     def view_product(self, asin, query, page):
#         """View product details."""
#         response = requests.get(
#             f"{self.base_url}/item_page/{self.session_id}/{asin}/{query}/{page}/[]"
#         )
#         if response.status_code != 200:
#             print(f"Failed to view product: {response.status_code}")
#             return None

#         html = response.text
#         print(f"Viewed product details for ASIN: {asin}")
#         return self.parse_product_details_from_html(html)

#     def parse_product_details_from_html(self, html):
#         """Parse product details from the HTML (placeholder)."""
#         # Parse the HTML to extract product details
#         return {"asin": "B01N5IB20Q", "options": ["option1", "option2"]}

#     def view_item_sub_page(self, asin, query, page, sub_page, options):
#         """View a sub-page of the product."""
#         response = requests.get(
#             f"{self.base_url}/item_sub_page/{self.session_id}/{asin}/{query}/{page}/{sub_page}/{options}"
#         )
#         if response.status_code != 200:
#             print(f"Failed to view sub-page: {response.status_code}")
#             return None

#         html = response.text
#         print(f"Viewed sub-page {sub_page} for ASIN: {asin}")
#         return self.parse_sub_page_details_from_html(html)

#     def parse_sub_page_details_from_html(self, html):
#         """Parse details from the sub-page HTML (placeholder)."""
#         # Parse the HTML to extract sub-page details
#         return {"sub_page_details": "Sample details from sub-page"}

#     def buy_product(self, asin, options):
#         """Simulate buying a product."""
#         response = requests.get(
#             f"{self.base_url}/done/{self.session_id}/{asin}/{options}"
#         )
#         if response.status_code == 200:
#             html = response.text
#             print(f"Bought product with ASIN: {asin}")
#             return self.parse_done_page(html)
#         else:
#             print(f"Failed to buy product with ASIN: {asin}")
#             return None
        
#     def parse_done_page(self, html):
#         """Parse the confirmation details from the done_page HTML."""
#         soup = BeautifulSoup(html, 'html.parser')
#         try:
#             # Extract thank-you message
#             thank_you_message = soup.find('h1', id='thankyou').text.strip()

#             # Extract MTurk code
#             mturk_code = soup.find('pre').text.strip()

#             # Extract reward score
#             reward_text = soup.find('h3', id='reward').find('pre').text.strip()
#             reward = float(reward_text)

#             # Extract product details (if visible)
#             asin = soup.find('h4', id='asin').find('pre').text.strip()
#             options = soup.find('h4', id='options').find('pre').text.strip()
#             purchased_attrs = soup.find('h4', id='purchased_attrs').find('pre').text.strip()
#             category = soup.find('h4', id='purchased-category').find('pre').text.strip()
#             query = soup.find('h4', id='purchased-query').find('pre').text.strip()
#             product_category = soup.find('h4', id='purchased-pc').find('pre').text.strip()

#             return {
#                 "thank_you_message": thank_you_message,
#                 "mturk_code": mturk_code,
#                 "reward": reward,
#                 "product_details": {
#                     "asin": asin,
#                     "options": options,
#                     "purchased_attrs": purchased_attrs,
#                     "category": category,
#                     "query": query,
#                     "product_category": product_category,
#                 }
#             }
#         except AttributeError as e:
#             print(f"Error parsing done page: {e}")
#             return None



#     def run(self, queries):
#         """Run the agent with a sequence of queries."""
#         self.start_session()

#         for query in queries:
#             products = self.search(query)
#             if products:
#                 top_product = products[0]  # Pick the top product
#                 asin = top_product["asin"]
#                 print(f"Viewing product: {asin}")
#                 product_details = self.view_product(asin, query, 1)
#                 if product_details:
#                     options = product_details.get("options", [])
#                     sub_page = "details"  # Example sub-page name
#                     print(f"Viewing sub-page {sub_page} for ASIN: {asin}")
#                     sub_page_details = self.view_item_sub_page(
#                         asin, query, 1, sub_page, options
#                     )
#                     print(f"Sub-page details: {sub_page_details}")
#                     print(f"Buying product: {asin} with options {options}")
#                     self.buy_product(asin, options)

#         print("Agent finished all tasks.")


# if __name__ == "__main__":
#     agent = SimpleAgent(base_url="http://10.217.49.21:3000")  # Update base_url
#     queries = [
#         "Looking for a white ceramic dinner set under $200",
#         "Find me a silver smartphone case above $25 with a minimalist design",
#     ]
#     agent.run(queries)


# ####################################################################

# from bs4 import BeautifulSoup

# def parse_products_from_html(self, html):
#     """Parse products from the search results HTML."""
#     soup = BeautifulSoup(html, 'html.parser')
#     products = []

#     # Find all product items in the HTML
#     product_elements = soup.find_all('div', class_='list-group-item')

#     for element in product_elements:
#         try:
#             # Extract ASIN from the product link
#             asin = element.find('h4', class_='product-asin').a.text.strip()

#             # Extract product title
#             title = element.find('h4', class_='product-title').text.strip()

#             # Extract price and convert to float
#             price_text = element.find('h5', class_='product-price').text.strip()
#             price = float(price_text.replace('$', '').replace(',', ''))

#             # Extract main image URL
#             image_url = element.find('img', class_='result-img')['src']

#             # Append the product details to the list
#             products.append({
#                 "asin": asin,
#                 "name": title,
#                 "price": price,
#                 "image": image_url,
#             })
#         except AttributeError as e:
#             print(f"Error parsing product element: {e}")

#     return products


# ##########################################################################


# from bs4 import BeautifulSoup

# def parse_product_details_from_html(self, html):
#     """Parse product details from the item page HTML."""
#     soup = BeautifulSoup(html, 'html.parser')
#     try:
#         # Extract product image
#         image_url = soup.find('img', id='product-image')['src']

#         # Extract product title
#         title = soup.find('h2').text.strip()

#         # Extract product price
#         price_text = soup.find('h4', text=lambda x: x and "Price:" in x).text.strip()
#         price = float(price_text.replace('Price:', '').strip().replace('$', ''))

#         # Extract product rating
#         rating_text = soup.find('h4', text=lambda x: x and "Rating:" in x).text.strip()
#         rating = float(rating_text.replace('Rating:', '').strip())

#         # Extract product options
#         options = {}
#         for option_section in soup.find_all('div', class_='radio-toolbar'):
#             option_name = option_section.find_previous('h4').text.strip()
#             option_values = [
#                 label.text.strip()
#                 for label in option_section.find_all('label')
#             ]
#             options[option_name] = option_values

#         return {
#             "image_url": image_url,
#             "title": title,
#             "price": price,
#             "rating": rating,
#             "options": options,
#         }
#     except AttributeError as e:
#         print(f"Error parsing product details: {e}")
#         return None

# ########################################################################


# def view_item_sub_page(self, asin, query, page, sub_page, options):
#     """View a sub-page of the product."""
#     response = requests.get(
#         f"{self.base_url}/item_sub_page/{self.session_id}/{asin}/{query}/{page}/{sub_page}/{options}"
#     )
#     if response.status_code != 200:
#         print(f"Failed to view sub-page: {response.status_code}")
#         return None

#     html = response.text
#     print(f"Viewed sub-page {sub_page} for ASIN: {asin}")

#     # Determine which parser to use based on the sub_page
#     if sub_page == "Attributes":
#         return self.parse_attributes_page(html)
#     elif sub_page == "Features":
#         return self.parse_features_page(html)
#     elif sub_page == "Reviews":
#         return self.parse_review_page(html)
#     elif sub_page == "Description":
#         return self.parse_description_page(html)
#     else:
#         print(f"Unknown sub-page type: {sub_page}")
#         return None


# def parse_attributes_page(self, html):
#     """Parse attributes from attributes_page.html."""
#     soup = BeautifulSoup(html, 'html.parser')
#     try:
#         attributes = [
#             attribute.text.strip()
#             for attribute in soup.find_all('p', class_='attribute')
#         ]
#         category = soup.find('h5', class_='product-category').text.strip()
#         query = soup.find('h5', class_='product-query').text.strip()
#         product_category = soup.find('h5', class_='product-product_category').text.strip()

#         return {
#             "type": "Attributes",
#             "attributes": attributes,
#             "category": category,
#             "query": query,
#             "product_category": product_category
#         }
#     except AttributeError as e:
#         print(f"Error parsing attributes page: {e}")
#         return None

# def parse_features_page(self, html):
#     """Parse features from features_page.html."""
#     soup = BeautifulSoup(html, 'html.parser')
#     try:
#         features = [
#             feature.text.strip()
#             for feature in soup.find_all('p', class_='product-info')
#         ]
#         return {"type": "Features", "features": features}
#     except AttributeError as e:
#         print(f"Error parsing features page: {e}")
#         return None


# def parse_review_page(self, html):
#     """Parse reviews from review_page.html."""
#     soup = BeautifulSoup(html, 'html.parser')
#     try:
#         reviews = []
#         for review_card in soup.find_all('div', class_='card'):
#             title = review_card.find('h4', class_='blue-text').text.strip()
#             score = int(review_card.find('span').text.strip())
#             body = review_card.find('p', class_='content').text.strip()
#             reviews.append({"title": title, "score": score, "body": body})

#         return {"type": "Reviews", "reviews": reviews}
#     except AttributeError as e:
#         print(f"Error parsing review page: {e}")
#         return None

# def parse_description_page(self, html):
#     """Parse description from description_page.html."""
#     soup = BeautifulSoup(html, 'html.parser')
#     try:
#         description = soup.find('p', class_='product-info').text.strip()
#         return {"type": "Description", "description": description}
#     except AttributeError as e:
#         print(f"Error parsing description page: {e}")
#         return None

# ######################################################################################

# def buy_product(self, asin, options):
#     """Simulate buying a product."""
#     response = requests.get(
#         f"{self.base_url}/done/{self.session_id}/{asin}/{options}"
#     )
#     if response.status_code == 200:
#         html = response.text
#         print(f"Bought product with ASIN: {asin}")
#         return self.parse_done_page(html)
#     else:
#         print(f"Failed to buy product with ASIN: {asin}")
#         return None

# def parse_done_page(self, html):
#     """Parse the confirmation details from the done_page HTML."""
#     soup = BeautifulSoup(html, 'html.parser')
#     try:
#         # Extract thank-you message
#         thank_you_message = soup.find('h1', id='thankyou').text.strip()

#         # Extract MTurk code
#         mturk_code = soup.find('pre').text.strip()

#         # Extract reward score
#         reward_text = soup.find('h3', id='reward').find('pre').text.strip()
#         reward = float(reward_text)

#         # Extract product details (if visible)
#         asin = soup.find('h4', id='asin').find('pre').text.strip()
#         options = soup.find('h4', id='options').find('pre').text.strip()
#         purchased_attrs = soup.find('h4', id='purchased_attrs').find('pre').text.strip()
#         category = soup.find('h4', id='purchased-category').find('pre').text.strip()
#         query = soup.find('h4', id='purchased-query').find('pre').text.strip()
#         product_category = soup.find('h4', id='purchased-pc').find('pre').text.strip()

#         return {
#             "thank_you_message": thank_you_message,
#             "mturk_code": mturk_code,
#             "reward": reward,
#             "product_details": {
#                 "asin": asin,
#                 "options": options,
#                 "purchased_attrs": purchased_attrs,
#                 "category": category,
#                 "query": query,
#                 "product_category": product_category,
#             }
#         }
#     except AttributeError as e:
#         print(f"Error parsing done page: {e}")
#         return None

# #####################################################################

# def run(self, queries):
#     """Run the agent with a sequence of queries."""
#     self.start_session()

#     for query in queries:
#         # Step 1: Search for products
#         products = self.search(query)
#         if products:
#             # Step 2: Select the top product
#             top_product = products[0]  # Pick the top product
#             asin = top_product["asin"]
#             product_name = top_product.get("name", "Unknown Product")
#             product_price = top_product.get("price", "N/A")
#             print(f"Selected product: {product_name} (ASIN: {asin}, Price: {product_price})")

#             # Step 3: View product details
#             product_details = self.view_product(asin, query, 1)
#             if product_details:
#                 options = product_details.get("options", {})
#                 print(f"Product details: {product_details}")

#                 # Step 4: View a sub-page (e.g., Description)
#                 sub_page = "Description"  # Example sub-page
#                 sub_page_details = self.view_item_sub_page(asin, query, 1, sub_page, options)
#                 print(f"Sub-page details ({sub_page}): {sub_page_details}")

#                 # Step 5: Purchase the product
#                 print(f"Attempting to buy product: {product_name} (ASIN: {asin}) with options: {options}")
#                 purchase_result = self.buy_product(asin, options)
#                 if purchase_result:
#                     print(f"Purchase successful: {purchase_result}")
#                 else:
#                     print(f"Purchase failed for product: {product_name} (ASIN: {asin})")

#     print("Agent finished all tasks.")
