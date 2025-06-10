import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Product Catalog Setup (Expanded with More Categories) ---
data = {
    'id': [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, # Original Electronics, Home, Fitness, Office
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25,             # More Electronics, Home, Fitness, Office
        26, 27, 28,                                         # Books
        29, 30, 31,                                         # Clothing
        32, 33,                                             # Toys & Games
        34, 35,                                             # Beauty & Personal Care
        36, 37, 38                                          # Sports & Outdoors
    ],
    'name': [
        "Pro Laptop X1", "Budget Laptop Y2", "Gaming Laptop Z3",
        "Ultra HD Smart TV", "Portable Bluetooth Speaker", "Noise-Cancelling Headphones",
        "Espresso Machine", "Drip Coffee Maker", "Blender Pro",
        "Yoga Mat Premium", "Dumbbell Set (5-25lbs)", "Resistance Bands Kit",
        "Leather Office Chair", "Standing Desk Converter", "Ergonomic Mouse",
        "Smartwatch Series 7", "Wireless Earbuds Pro", "Air Fryer XL",
        "Electric Kettle Go", "Running Shoes Flex", "Fitness Tracker Band",
        "Dual Monitor Arm", "Premium Fountain Pen", "LED Desk Lamp", "Portable SSD 1TB",
        "The Midnight Library", "Sapiens: A Brief History", "Dune (Sci-Fi Classic)",
        "Classic Cotton T-Shirt", "Slim Fit Denim Jeans", "Lightweight Windbreaker",
        "Catan Board Game", "Superhero Action Figure",
        "Hydrating Face Moisturizer", "Organic Shampoo & Conditioner Set",
        "Insulated Water Bottle", "2-Person Camping Tent", "All-Season Sleeping Bag"
    ],
    'category': [
        "Electronics", "Electronics", "Electronics", "Electronics", "Electronics", "Electronics",
        "Home Appliances", "Home Appliances", "Home Appliances",
        "Fitness", "Fitness", "Fitness",
        "Office", "Office", "Office",
        "Electronics", "Electronics", "Home Appliances", "Home Appliances", "Fitness", "Fitness",
        "Office", "Office", "Office", "Electronics",
        "Books", "Books", "Books",
        "Clothing", "Clothing", "Clothing",
        "Toys & Games", "Toys & Games",
        "Beauty & Personal Care", "Beauty & Personal Care",
        "Sports & Outdoors", "Sports & Outdoors", "Sports & Outdoors"
    ],
    'price': [
        1200, 550, 1800, 700, 80, 250, 300, 60, 120, 40, 150, 25, 280, 180, 35,
        399, 149, 110, 45, 130, 89, 75, 55, 40, 120,
        15, 20, 12,
        25, 60, 45,
        50, 20,
        30, 40,
        28, 90, 70
    ],
    'description': [
        "High-performance laptop for professionals, fast SSD, long battery life, lightweight.", "Affordable laptop for students and basic tasks, good battery, reliable.", "Top-tier gaming laptop with dedicated graphics card, high refresh rate screen, RGB keyboard.",
        "55-inch 4K Smart TV with vibrant colors and smart features.", "Compact and powerful Bluetooth speaker with excellent sound quality, water-resistant.", "Over-ear headphones with active noise cancellation, superior comfort, long playtime.",
        "Premium espresso machine for barista-quality coffee at home, milk frother included.", "Simple and effective drip coffee maker, programmable, large carafe.", "Powerful blender for smoothies, soups, and more, multiple speed settings.",
        "Eco-friendly, non-slip yoga mat for all types of practice.", "Adjustable dumbbell set, perfect for home workouts, space-saving.", "Versatile resistance bands for strength training and stretching.",
        "Comfortable high-back leather office chair with lumbar support.", "Adjustable standing desk converter, improve posture and productivity.", "Vertical ergonomic mouse to reduce wrist strain.",
        "Advanced smartwatch with GPS, heart rate monitor, and various apps.", "True wireless earbuds with noise cancellation and long battery life.", "Large capacity air fryer for healthier cooking with less oil.",
        "Fast-boiling electric kettle with auto shut-off and stainless steel body.", "Comfortable and durable running shoes for daily training and marathons.", "Sleek fitness tracker to monitor steps, sleep, and workouts.",
        "Adjustable dual monitor arm for ergonomic screen positioning and desk space saving.", "Elegant fountain pen for a smooth writing experience, perfect for gifting.", "Modern LED desk lamp with adjustable brightness and color temperature.", "Fast and compact portable Solid State Drive with 1TB storage for backups and transfers.",
        "A captivating novel about choices and regrets. International bestseller.", "An exploration of human history from the Stone Age to the present. Highly acclaimed.", "Classic science fiction epic set on the desert planet Arrakis. Hugo Award winner.",
        "Comfortable and durable 100% cotton t-shirt, available in various colors.", "Stylish slim fit jeans made with stretch denim for maximum comfort and mobility.", "Water-resistant and windproof jacket, perfect for outdoor activities and travel.",
        "Strategy board game of trading and building. Fun for the whole family or game nights.", "Detailed 6-inch action figure of a popular superhero, great for collectors.",
        "Lightweight daily moisturizer with hyaluronic acid and natural extracts for all skin types.", "Sulfate-free organic shampoo and conditioner set for healthy, shiny hair.",
        "Double-wall insulated stainless steel water bottle, keeps drinks cold for 24h or hot for 12h.", "Lightweight and easy-to-set-up dome tent for 2 people, ideal for backpacking and camping.", "Comfortable sleeping bag suitable for 3-season camping, rated for moderate temperatures."
    ],
    'tags': [
        "laptop, work, professional, high-performance, ssd", "laptop, student, budget, affordable, reliable", "laptop, gaming, graphics, high-refresh, powerful",
        "tv, smart-tv, 4k, entertainment, living-room", "speaker, bluetooth, portable, audio, music", "headphones, audio, noise-cancelling, travel, music",
        "coffee, espresso, kitchen, appliance, morning", "coffee, drip, kitchen, appliance, programmable", "blender, kitchen, smoothie, appliance, health",
        "yoga, fitness, exercise, wellness, eco-friendly", "weights, fitness, strength, workout, home-gym", "fitness, resistance, workout, stretch, portable",
        "office, chair, furniture, ergonomic, comfort", "office, desk, ergonomic, health, productivity", "office, mouse, ergonomic, computer, accessory",
        "smartwatch, wearable, health, fitness, gps, apple-watch-alternative", "earbuds, wireless, audio, music, noise-cancelling, airpods-alternative", "airfryer, kitchen, cooking, healthy, appliance",
        "kettle, electric, tea, coffee, kitchen, appliance", "shoes, running, fitness, sport, marathon, training", "fitnesstracker, wearable, health, steps, sleep, fitbit-alternative",
        "monitor, arm, office, ergonomic, desk, setup", "pen, fountain, writing, luxury, gift, office, stationery", "lamp, led, desk, office, lighting, study, reading", "ssd, storage, portable, backup, harddrive, fast, external",
        "fiction, novel, fantasy, contemporary, best-seller, book, reading", "non-fiction, history, anthropology, science, book, reading, educational", "sci-fi, science-fiction, classic, space-opera, book, reading, novel",
        "t-shirt, cotton, casual, apparel, basic, top, clothing", "jeans, denim, pants, slim-fit, apparel, bottoms, clothing", "jacket, windbreaker, outdoor, apparel, lightweight, clothing, outerwear",
        "board-game, strategy, family-game, settlers-of-catan, tabletop, game", "action-figure, toy, superhero, collectible, kids, game",
        "skincare, moisturizer, face-cream, hyaluronic-acid, beauty, cosmetic, personal-care", "haircare, shampoo, conditioner, organic, beauty, cosmetic, personal-care, hair",
        "water-bottle, hydration, outdoor, fitness, travel, sports, insulated, bottle", "tent, camping, outdoor, backpacking, hiking, sports, shelter", "sleeping-bag, camping, outdoor, hiking, sports, shelter, warmth"
    ],
    'complementary_ids': [
        [14, 15, 25], [15, 25], [15, 22], [5, 6, 17], [], [17], [8], [7], [],
        [11, 12, 20, 21, 36], [10, 12, 20, 21, 36], [10, 11, 20, 21, 36], [14, 15, 22, 24], [13, 15, 22, 24], [13, 14, 24, 25],
        [21, 17], [6, 25], [9], [7], [10, 11, 12, 21, 31, 36], [16, 20, 36],
        [13, 14, 24], [24, 26, 27, 28], [23, 26, 27, 28], [1, 2, 3], # End of original + some cross-category links
        [24, 23], [24, 23], [24],                                     # Books
        [30], [29], [20, 36],                                       # Clothing
        [], [],                                                      # Toys & Games
        [35], [34],                                                  # Beauty & Personal Care
        [10,11,12,20,21,31,37,38], [36,38], [36,37]                  # Sports & Outdoors
    ]
}
products_df = pd.DataFrame(data)
products_df.set_index('id', inplace=True)


# --- 2. Content-Based Recommender (ML Component) ---
class ContentBasedRecommender:
    def __init__(self, products_df): # CORRECTED from _init_
        self.products_df = products_df.copy()
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._build_model()

    def _build_model(self):
        self.products_df['combined_features'] = self.products_df['name'] + ' ' + \
                                                self.products_df['description'] + ' ' + \
                                                self.products_df['category'] + ' ' + \
                                                self.products_df['tags']
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products_df['combined_features'])

    def get_similar_products(self, product_id, top_n=3):
        if product_id not in self.products_df.index:
            return pd.DataFrame()

        idx = self.products_df.index.get_loc(product_id)
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        similar_product_indices = []
        for i, score in sim_scores[1:]: # Start from 1 to exclude the product itself
            if len(similar_product_indices) < top_n:
                # i is the original index from enumerate, which corresponds to the row in tfidf_matrix
                # and also to the position in products_df if it wasn't filtered/reordered
                # Since we used products_df.index.get_loc(product_id), 'i' can directly be used
                # to get the actual product ID from the DataFrame's index
                actual_product_id = self.products_df.index[i]
                if actual_product_id != product_id: # Ensure we don't add the same product
                     similar_product_indices.append(actual_product_id)
            else:
                break
        
        if not similar_product_indices:
            return pd.DataFrame()

        return self.products_df.loc[similar_product_indices]
11

# --- 3. Shopping Bot Logic ---
class ShoppingBot:
    def __init__(self, products_df, recommender): # CORRECTED from _init_
        self.products_df = products_df
        self.recommender = recommender
        self.user_preferences = {}
        self.current_selection = None

    def greet(self):
        print("Welcome to your Personal Shopping Assistant!")
        print("I can help you find products based on category and budget.")

    def ask_initial_preferences(self):
        print("\nLet's find what you're looking for.")
        self.user_preferences = {}

        categories = sorted(self.products_df['category'].unique().tolist()) # Sort categories for display
        print("Available categories:")
        for i, cat in enumerate(categories):
            print(f"{i+1}. {cat}")
        while True:
            try:
                choice_str = input(f"Which category are you interested in? (1-{len(categories)} or skip): ").strip()
                if not choice_str:
                    break
                choice = int(choice_str) - 1
                if 0 <= choice < len(categories):
                    self.user_preferences['category'] = categories[choice]
                    break
                else:
                    print("Invalid choice. Please select a number from the list or press Enter to skip.")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter to skip.")

        budget_known = input("Do you have a budget in mind? (yes/no, or skip): ").lower().strip()
        if budget_known == 'yes':
            while True:
                try:
                    min_budget_str = input("What's your minimum budget? (e.g., 50, or 0, or skip): ").strip()
                    max_budget_str = input("What's your maximum budget? (e.g., 1000, or skip): ").strip()

                    min_budget = float(min_budget_str) if min_budget_str else 0.0
                    max_budget = float(max_budget_str) if max_budget_str else float('inf')

                    if min_budget < 0 or max_budget < min_budget:
                        print("Invalid budget range. Max budget must be >= min budget, and min budget >= 0.")
                    else:
                        if min_budget_str: self.user_preferences['min_budget'] = min_budget
                        if max_budget_str: self.user_preferences['max_budget'] = max_budget
                        break
                except ValueError:
                    print("Invalid input. Please enter a number for budget or leave blank.")

        print("\nThanks! Searching for products based on your preferences...")

    def filter_products(self):
        filtered_df = self.products_df.copy()

        if 'category' in self.user_preferences:
            filtered_df = filtered_df[filtered_df['category'] == self.user_preferences['category']]

        if 'min_budget' in self.user_preferences:
            filtered_df = filtered_df[filtered_df['price'] >= self.user_preferences['min_budget']]

        if 'max_budget' in self.user_preferences:
             filtered_df = filtered_df[filtered_df['price'] <= self.user_preferences['max_budget']]

        return filtered_df

    def display_products_tabular(self, df_to_display, title="Matching Products"):
        if df_to_display.empty:
            print(f"\n--- {title} ---")
            print("Sorry, no products match your current criteria.")
            return False

        print(f"\n--- {title} ({len(df_to_display)}) ---")
        display_df = df_to_display[['name', 'price', 'category']].copy()
        display_df.index.name = 'ID'
        print(display_df.to_string()) # .to_string() helps display full DataFrame in console
        return True

    def suggest_complementary_products(self, product_id):
        if product_id not in self.products_df.index:
            return

        product_name = self.products_df.loc[product_id, 'name']
        complementary_ids_list = self.products_df.loc[product_id, 'complementary_ids']

        if not isinstance(complementary_ids_list, list) or not complementary_ids_list:
            return

        valid_complementary_ids = [cid for cid in complementary_ids_list if cid in self.products_df.index]
        if not valid_complementary_ids:
            return

        complementary_prods_df = self.products_df.loc[valid_complementary_ids]
        if not complementary_prods_df.empty:
            self.display_products_tabular(complementary_prods_df, title=f"Complementary Products for {product_name}")


    def suggest_similar_by_content(self, product_id):
        product_name = self.products_df.loc[product_id, 'name']
        similar_df = self.recommender.get_similar_products(product_id, top_n=3)
        if not similar_df.empty:
            self.display_products_tabular(similar_df, title=f"You Might Also Like (Similar to {product_name})")

    def run(self):
        self.greet()

        while True:
            self.ask_initial_preferences()
            matched_products = self.filter_products()

            if not self.display_products_tabular(matched_products):
                refine = input("Would you like to refine your search or exit? (refine/exit): ").lower().strip()
                if refine != 'refine':
                    break
                else:
                    continue

            while True: # Inner loop for actions on the current product list
                choice = input("\nEnter Product ID for details & suggestions, 'refine' search, or 'exit': ").lower().strip()
                if choice == 'exit':
                    print("Thank you for using the Shopping Assistant!")
                    return
                if choice == 'refine':
                    break # Breaks inner loop to go back to ask_initial_preferences

                try:
                    product_id_choice = int(choice)
                    if product_id_choice in self.products_df.index:
                        self.current_selection = product_id_choice
                        if product_id_choice not in matched_products.index:
                             print(f"\nNote: Product ID {product_id_choice} is valid but was not in your last search results. Showing details anyway.")
                    else:
                        print("Invalid Product ID. Please choose from the list or type 'refine'/'exit'.")
                        continue

                    selected_product = self.products_df.loc[self.current_selection]
                    print(f"\n--- Details for: {selected_product['name']} (ID: {self.current_selection}) ---")
                    print(f"Price: ${selected_product['price']:.2f}")
                    print(f"Category: {selected_product['category']}")
                    print(f"Description: {selected_product['description']}")
                    print(f"Tags: {selected_product['tags']}")

                    self.suggest_complementary_products(self.current_selection)
                    self.suggest_similar_by_content(self.current_selection)
                    print("\nWhat would you like to do next with the current search results list?")

                except ValueError:
                    print("Invalid input. Please enter a valid Product ID number, 'refine', or 'exit'.")
                except KeyError: # Should be rare if product_id_choice is in self.products_df.index
                    print(f"Error: Product ID {product_id_choice} not found in the catalog.")

        print("Goodbye!")


# --- 4. Running the Bot ---
if __name__ == "__main__": # CORRECTED from _main_
    recommender = ContentBasedRecommender(products_df)
    bot = ShoppingBot(products_df, recommender)
    bot.run()