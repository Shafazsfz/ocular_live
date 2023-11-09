import os
import streamlit as st
from streamlit_chat import message
import csv
import sqlite3
import pandas as pd
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from langchain.prompts import MessagesPlaceholder


api_key = st.sidebar.text_input("Enter your API key:", type="password")

csv_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

tablename = 'csv_file.name'

prefix_text = st.sidebar.text_area(
    "Enter prefix to pass:",
    """Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
    note: Use `ILIKE %keyword%` in your SQL query to perform fuzzy text matching within text  or string datatype columns.
    
    Use the following for output format:
    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"
    You should return the SQL query along with the result or response.

    Your output for user input should be like this format-

    ```     Question: "How many products are there?"
            SQLQuery: "Select count (distinct products) from orders"
            SQLResult: "1112"
            Answer: "There are 1112 products."
    ```""",height=600
)

st.sidebar.write(f'You wrote {len(prefix_text)} characters.')

# Load CSV into DataFrame and create SQLite database
def load_data(csv_file, tablename):
    tablename = csv_file.name()
    tablename = tablename.replace(" ", "_")
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect('db2.db')
    df.to_sql(tablename, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()



# Initialize the LLMPredictor and other necessary components
def initialize_llm_predictor():

    custom_table_info = {
    "fact_order_item": f"""CREATE TABLE "fact_order_item" (
                        order_id INTEGER PRIMARY KEY,
                        order_item_id,
                        marketplace,
                        order_date,
                        quantity,
                        price,
                        discounts,
                        shipping_pincode,
                        state,
                        city,
                        sku_name,
                        asin,
                        amazon_parent_sku,
                        fabric,
                        product_name,
                        collection,
                        master_collection,
                        product_type,
                        weigh_slab,
                        awb,
                        shipping_status,
                        simplified_status,
                        fulfillment_channel,
                        inner_consumption,
                        outer_consumption,
                        consumption_cost,
                        cost
                    )

    /*
    Note for fact_order_item: 
    1. Order_id is the unique identifier of every order placed in alphanumeric format
    2. Order_item_id is the unique identifier of every item within an order in alphanumeric format. An order can have multiple items within
    3. Marketplace is the sales channel for the order
    4. Order_date is the date and time at which an order was placed
    5. quantity is the quantity of the product ordered in that order for that order item in integer format
    6. Price is numeric format of the INR price for one unit of the product
    7. Discounts is numeric format of the INR discount to price for one unit of the product.
    8. Shipping_pincode is an alphanumeric pincode of the location
    9. State is the state within a country from which the order is placed
    10. City is the city within a state from which the order is placed
    11. Sku_name is the specific alphanumeric identifier for a product
    12. Fabric is an alphanumeric to describe the type of fabric being sold
    13. Product_name is the name of the specific product being sold in alphanumeric
    14. Collection is the collection or category of which the product_type is a part
    15. Master_collection is a master collection of which the collection is a part
    16. Product_type is a product type of which the product is a part
    17. Weigh_slab is the weight slab of the specific product
    18. shipping_status is a string field that describes the shipping status of an order, returned, cancelled orders should not be considered in revenue calculations
    19. Awb is the unique shipping ID given to that product
    20. Cost is the INR cost of the product in the row
    21. Fulfillment_channel is a string field that describes the channel through which final fulfillment / shipping was done
    22. Use `ILIKE %keyword% in your SQL query to perform fuzzy text matching within text  or string datatype columns.
    */""",

    "fact_ads_sales_spend": f"""CREATE TABLE "fact_ads_sales_spend" (
                        sku_name,
                        date,
                        asin,
                        cost,
                        attributed_sales,
                        total_sales,
                        total_orders,
                        impressions,
                        clicks,
                        cpc,
                        cpm,
                        acos,
                        roas,
                        ctr,
                        sessions,
                        unitsOrdered,
                        unitSessionPercentage,
                        pageViews,
                        pageViewsPercentage
                    )

    /* 
    Note for fact_ads_sales_spend:                  
    1. Sku_name is the specific alphanumeric identifier for a product
    2. Date is the date on which the ad campaign was run in DD/MM/YYYY format
    3. Cost is the cost in INR of running the ad campaign
    4. Attributed_sales is the sales attributed in INR to that specific ad campaign, attributed_sales / cost is return on ad spend (roas)
    5. Total_sales is the total sales in INR against that ad campaign
    6. Total_orders is the total orders in units against that ad campaign
    7. Impressions is the number of impressions generated by that ad campaign
    8. Clicks is the total number of clicks against that ad campaign
    9. Cpc is the cost per click in INR against that ad campaign
    10. Cpm is the cost per 1000 impressions in INR against that ad campaign
    11. Acos is the advertising cost of sales
    12. Roas is the return on ad spends (Attributed_sales / cost)
    13. Ctr is the percentage click through rate of the campaign (clicks/impressions)
    14. Sessions is the number of sessions on website against that ad spend
    15. Use `ILIKE %keyword% in your SQL query to perform fuzzy text matching within text  or string datatype columns.
    */"""
    }

    db = SQLDatabase.from_uri("sqlite:///db2.db", custom_table_info=custom_table_info)
    #llm = OpenAI(model_name= 'gpt-2.5-turbo', temperature=0, verbose=True)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)

    agent = create_sql_agent(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0),
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=prefix_text,
    )
    return agent



if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.title("")

    if csv_file:
        load_data(csv_file)
        st.write("CSV file uploaded successfully!")

        # Initialize LLMPredictor
        query_engine = initialize_llm_predictor()

        # def conversational_chat(query):
        #     result = query_engine.query(query)
        #     st.session_state['history'].append((query, result.response, result.metadata['sql_query']))
        #     return result.response, result.metadata['sql_query']
        
        def conversational_chat(query):
            result = query_engine.run(query)
            #sql_query = result.metadata['sql_query']
            #combined_response = f"{result.response}\n \nSQL Query: \n{sql_query}"
            st.session_state['history'].append((query, result))
            return result

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + csv_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Ask question here", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
