import pandas as pd
import numpy as np
from faker import Faker
import random
import streamlit as st
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Streamlit app setup
st.title("📊 Shopify Reviews Dataset Generator")
st.markdown("Generate a custom dataset of 2,000 customer reviews")

# Product and category data
products = {
    "Apparel": ["Classic White T-Shirt", "Black Jeans", "Hoodie", "Summer Dress"],
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Fitness Tracker"],
    "Home": ["Ceramic Coffee Mug", "Yoga Mat", "Desk Lamp", "Throw Pillow"],
    "Beauty": ["Acne Face Wash", "Sunscreen SPF 50", "Lipstick Set", "Hair Dryer"]
}

# Generate data function
@st.cache_data
def generate_data(rows=2000):
    data = []
    for review_id in range(1000, 1000 + rows):
        category = random.choice(list(products.keys()))
        product = random.choice(products[category])
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.3, 0.35])
        review_content = (
            # 5-star reviews (40+ variations)
            random.choice([
                "Absolutely love this product! Exceeded my expectations.",
                "Perfect in every way - will definitely buy again!",
                "Worth every penny. The quality is outstanding.",
                "Better than I imagined. Fast shipping too!",
                "My new favorite product. Works perfectly!",
                "Exceptional quality. Couldn't be happier!",
                "This product changed my daily routine for the better.",
                "100% satisfied. Met all my requirements.",
                "Packaging was eco-friendly and product arrived safely.",
                "The attention to detail is remarkable.",
                "Exactly as described, if not better!",
                "Customer service was excellent when I had questions.",
                "I've recommended this to all my friends!",
                "The photos don't do it justice - it's even better in person.",
                "Arrived early and in perfect condition.",
                "I'm blown away by how good this is!",
                "Simple to use and works flawlessly.",
                "The perfect gift - recipient loved it!",
                "I use this every day and it still looks new.",
                "Solved a problem I've had for years.",
                "Worth the wait - absolutely perfect!",
                "The gold standard of products in this category.",
                "I'm so impressed I bought a second one!",
                "Beautiful craftsmanship and durable materials.",
                "Five stars isn't enough - it's that good!",
                "The instructions were clear and setup was easy.",
                "I was skeptical but this delivered beyond expectations.",
                "Perfect size and weight - very comfortable.",
                "The colors are even more vibrant in person.",
                "I've tried many brands and this is the best by far.",
                "My kids love it too - family approved!",
                "Great for beginners and experts alike.",
                "The company stands behind their product 100%.",
                "I can't imagine my life without this now!",
                "The perfect balance of form and function.",
                "Arrived carefully packaged with no damage.",
                "Exactly what I needed - perfect fit!",
                "The quality rivals products twice the price.",
                "I'm so glad I chose this over competitors.",
                "The perfect addition to my collection."
            ]) if rating == 5 else
            
            # 4-star reviews (40+ variations)
            random.choice([
                "Very good, but has some minor flaws.",
                "Great product overall with room for small improvements.",
                "Works well for the price point.",
                "Almost perfect - just one small issue.",
                "Good quality but could be more durable.",
                "Does what it promises with minor drawbacks.",
                "I'm happy with my purchase but it's not flawless.",
                "Better than average but not exceptional.",
                "Good value for money with some limitations.",
                "Met most of my expectations.",
                "Solid product that does the job well.",
                "Would buy again but hope for small improvements.",
                "The color is slightly different than pictured.",
                "Took longer to arrive than expected but good quality.",
                "Works as advertised with minor quirks.",
                "Good for beginners but pros might want more features.",
                "Comfortable but could use better padding.",
                "Effective but the instructions could be clearer.",
                "Looks great but shows fingerprints easily.",
                "Powerful but a bit noisy during operation.",
                "Stylish design but the material could be better.",
                "Functional but the interface could be more intuitive.",
                "Good performance but battery life could be longer.",
                "Comfortable fit but runs slightly large.",
                "Nice product but packaging was damaged.",
                "Works well but took some time to set up properly.",
                "Great concept but execution could be refined.",
                "Good for everyday use but not heavy-duty.",
                "Attractive design but not as durable as hoped.",
                "Effective but could use more size options.",
                "Pleasant experience with minor drawbacks.",
                "Does the job but could be more user-friendly.",
                "Nice quality but the color fades after washing.",
                "Good value but shipping was slower than competitors.",
                "Reliable but lacks some premium features.",
                "Comfortable but not for extended wear.",
                "Works as intended but not exceptional.",
                "Decent product but customer service was slow.",
                "Good but not worth the full retail price.",
                "Satisfactory with room for improvement."
            ]) if rating == 4 else
            
            # 3-star reviews (40+ variations)
            random.choice([
                "It's okay. Not great, not terrible.",
                "Average product - does the job but nothing special.",
                "Meets basic requirements but lacks wow factor.",
                "Exactly what you'd expect for the price.",
                "Functional but unremarkable.",
                "Neither impressed nor disappointed.",
                "Middle of the road - neither good nor bad.",
                "Adequate quality for occasional use.",
                "Does what it's supposed to but not exceptionally.",
                "Fair product at a fair price.",
                "Not bad, but I've seen better.",
                "Acceptable but won't buy again.",
                "Mediocre performance overall.",
                "It works, but just barely.",
                "Basic functionality - nothing more.",
                "Standard quality - nothing stands out.",
                "Gets the job done but not elegantly.",
                "Okay for beginners but professionals will want more.",
                "Average at best - expected better.",
                "Not terrible but not great either.",
                "Sufficient but not impressive.",
                "Middling quality - you get what you pay for.",
                "Works but feels cheaply made.",
                "Does the minimum required.",
                "Passable but forgettable product.",
                "Neither the best nor worst I've tried.",
                "Satisfactory but leaves room for improvement.",
                "Decent but the competition does it better.",
                "Acceptable quality with some flaws.",
                "Not worth returning but won't repurchase.",
                "Just okay - nothing to write home about.",
                "Fairly standard - no surprises good or bad.",
                "Meh. It exists and functions.",
                "Competent but uninspiring.",
                "Doesn't excel in any particular area.",
                "Basic product with basic performance.",
                "Not offensive but not appealing either.",
                "Forgettable experience overall.",
                "Wouldn't recommend but wouldn't warn against.",
                "Perfectly average in every way."
            ]) if rating == 3 else
            
            # 2-star reviews (40+ variations)
            random.choice([
                "Disappointed with the quality.",
                "Not worth the money in my opinion.",
                "Poor construction for the price.",
                "Several issues right out of the box.",
                "Lasted less time than expected.",
                "Looks cheap compared to competitors.",
                "Flimsy materials - won't last long.",
                "Difficult to use as intended.",
                "Arrived damaged or defective.",
                "Doesn't work as advertised.",
                "Underwhelming performance overall.",
                "Wouldn't buy again at this price.",
                "Fell apart after minimal use.",
                "Missing parts when delivered.",
                "Much smaller/cheaper looking than photos.",
                "Uncomfortable to use for long periods.",
                "Poor value for money spent.",
                "Instructions were unclear or missing.",
                "Not what I expected based on description.",
                "Quality control issues apparent.",
                "Broke within days of normal use.",
                "Customer service was unhelpful.",
                "Design flaws make it frustrating to use.",
                "Materials feel subpar for the category.",
                "Overpriced for what you get.",
                "Shipped late without notification.",
                "Safety concerns with how it functions.",
                "Difficult to assemble properly.",
                "Doesn't work with other standard products.",
                "Poor packaging led to damage.",
                "Not suitable for intended purpose.",
                "Malfunctioned almost immediately.",
                "Unpleasant smell/taste out of package.",
                "Colors ran during first wash.",
                "Missing important features competitors have.",
                "Unstable or wobbly during use.",
                "Parts don't fit together correctly.",
                "Loud or annoying during operation.",
                "Requires constant adjustments to work.",
                "Generally unsatisfied with purchase."
            ]) if rating == 2 else
            
            # 1-star reviews (40+ variations)
            random.choice([
                "Terrible product! Would not recommend.",
                "Complete waste of money.",
                "Broke immediately upon first use.",
                "Dangerous defect - could have caused injury.",
                "False advertising - nothing like description.",
                "Absolute junk - don't bother buying.",
                "Worst purchase I've made this year.",
                "Arrived used or dirty despite being 'new'.",
                "Missing critical components to function.",
                "Company refused refund for defective item.",
                "Hazardous materials or construction.",
                "Cheap knockoff of name brand product.",
                "Didn't work at all right out of box.",
                "Scam - product doesn't match photos.",
                "Customer service was rude and unhelpful.",
                "Fell apart within hours of normal use.",
                "Manufacturing defects throughout.",
                "Smelled toxic or chemically when opened.",
                "Not safe for intended age group.",
                "Literally caught fire during normal use.",
                "Stolen design from better companies.",
                "Shipping took months with no updates.",
                "Counterfeit item despite 'official' listing.",
                "Leaked dangerous substances during use.",
                "Complete failure at basic functionality.",
                "Wrong item shipped and no return label.",
                "Used my credit card info fraudulently.",
                "Violates basic safety standards.",
                "Product photos were deliberately misleading.",
                "Multiple attempts to contact seller failed.",
                "Item was clearly used/returned before.",
                "Missing safety features competitors include.",
                "Instructions were in wrong language.",
                "Charged twice and no refund processed.",
                "Product recalled after I purchased it.",
                "Expired product sold as new.",
                "Burned out electrical components immediately.",
                "Mold or mildew present in packaging.",
                "Sharp edges that could cut users.",
                "Total scam - avoid this seller completely."
            ]) if rating == 1 else ""
        )
        timestamp = fake.date_time_between(start_date="-1y", end_date="now")
        email = fake.email()
        order_value = round(random.uniform(10, 200), 2)
        status = random.choice(["fulfilled", "fulfilled", "returned", "refunded"])
        country = fake.country()

        data.append([
            review_id, product, rating, review_content, 
            timestamp, email, category, order_value, status, country
        ])
    
    df = pd.DataFrame(data, columns=[
        "Review ID", "Product Name", "Rating", "Review Content", 
        "Timestamp", "Customer Email", "Product Category", 
        "Order Value", "Fulfillment Status", "Shipping Country"
    ])
    return df

# Generate the dataset
if st.button("Generate Dataset (2,000 Reviews)"):
    with st.spinner("Creating your dataset..."):
        df = generate_data(2000)
        st.success("Dataset created successfully!")
        
        # Show preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="shopify_reviews_2000.csv",
            mime="text/csv",
            help="Click to download the generated dataset"
        )
        
        # Show stats
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(df))
        col2.metric("Average Rating", f"{df['Rating'].mean():.1f} ⭐")
        col3.metric("Countries", df['Shipping Country'].nunique())
        
        # Rating distribution chart
        st.bar_chart(df['Rating'].value_counts().sort_index())
else:
    st.info("Click the button above to generate your dataset")

# Instructions
st.sidebar.markdown("### Instructions")
st.sidebar.write("""
1. Click **Generate Dataset**
2. Preview the data
3. Download the CSV file
4. Use for analysis/testing
""")