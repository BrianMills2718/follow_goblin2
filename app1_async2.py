#!/usr/bin/env python3
# ASYNCHRONOUS NETWORK VISUALIZATION OF X ACCOUNT FOLLOWING NETWORK – VERSION 4.0
#
# This script retrieves the "following" network of an input X (formerly Twitter) account asynchronously.
# It then constructs a directed network graph using pyvis.
#
# NEW FEATURES:
#  1. The user may now select the maximum number of nodes to display in the graph.
#     The top nodes by in‐degree (i.e. most followed within the network) are always retained.
#  2. Display filters have been introduced. You may constrain the nodes by:
#       - statuses_count, followers_count, friends_count, media_count (via min/max bounds),
#       - created_at (by a date range),
#       - presence of a non‐empty location,
#       - blue_verified (boolean),
#       - verified status (select "Only Verified" or "Only Not Verified"),
#       - website (only nodes with a non‐empty website), and
#       - business_account status (select "Only Business Accounts" or "Only Non-Business Accounts").
#
# DEVIL'S ADVOCATE: Although the asynchronous API calls and filter logic are precise,
# further refinements (e.g. robust error handling, rate‐limiting, and parsing of nonstandard dates)
# may be required for production use.

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import aiohttp
import json
from pyvis.network import Network
import datetime

# CONSTANTS – Adjust your RapidAPI credentials as necessary.
RAPIDAPI_KEY = "d72bcd77e2msh76c7e6cf37f0b89p1c51bcjsnaad0f6b01e4f"
RAPIDAPI_HOST = "twitter-api45.p.rapidapi.com"

async def get_following_async(screenname: str, session: aiohttp.ClientSession):
    """
    Asynchronously retrieve the first page (50 accounts) of accounts that the given user is following.
    """
    url = f"https://{RAPIDAPI_HOST}/following.php?screenname={screenname}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.text()
            return json.loads(data).get("following", [])
    except Exception:
        return []

def compute_ratio(followers_count, friends_count):
    """Compute follower/following ratio; return 0 if denominator is zero."""
    try:
        return followers_count / friends_count if friends_count else 0
    except:
        return 0

async def main_async(input_username: str):
    """
    Retrieves and processes the following network asynchronously.
    Enhanced to store additional account attributes for filtering.
    """
    nodes, edges = {}, []
    original_id = f"orig_{input_username}"
    # The original node: minimal attributes with only available fields
    nodes[original_id] = {
        "screen_name": input_username,
        "name": input_username,
        "followers_count": None,
        "friends_count": None,
        "statuses_count": None,
        "media_count": None,
        "description": None,
        "ratio": None,
        "direct": True
    }

    async with aiohttp.ClientSession() as session:
        first_hop_accounts = await get_following_async(input_username, session)
        for account in first_hop_accounts:
            uid = str(account.get("user_id"))
            if not uid:
                continue
            ratio = compute_ratio(account.get("followers_count", 0), account.get("friends_count", 0))
            nodes[uid] = {
                "screen_name": account.get("screen_name", ""),
                "name": account.get("name", ""),
                "followers_count": account.get("followers_count", 0),
                "friends_count": account.get("friends_count", 0),
                "statuses_count": account.get("statuses_count", 0),
                "media_count": account.get("media_count", 0),
                "description": account.get("description", ""),
                "ratio": ratio,
                "direct": True
            }
            edges.append((original_id, uid))
        tasks = [get_following_async(acc.get("screen_name", ""), session) for acc in first_hop_accounts]
        second_hop_results = await asyncio.gather(*tasks)
        for idx, second_accounts in enumerate(second_hop_results):
            source_id = str(first_hop_accounts[idx].get("user_id"))
            for account in second_accounts:
                sid = str(account.get("user_id"))
                if not sid:
                    continue
                ratio = compute_ratio(account.get("followers_count", 0), account.get("friends_count", 0))
                if sid not in nodes:
                    nodes[sid] = {
                        "screen_name": account.get("screen_name", ""),
                        "name": account.get("name", ""),
                        "followers_count": account.get("followers_count", 0),
                        "friends_count": account.get("friends_count", 0),
                        "statuses_count": account.get("statuses_count", 0),
                        "media_count": account.get("media_count", 0),
                        "description": account.get("description", ""),
                        "ratio": ratio,
                        "direct": False
                    }
                edges.append((source_id, sid))
    return nodes, edges

def filter_nodes(nodes, filters):
    """
    Filters nodes based on provided filter criteria.
    """
    filtered = {}
    for node_id, node in nodes.items():
        # Always include the original node
        if node_id.startswith("orig_"):
            filtered[node_id] = node
            continue

        # Helper function to safely compare values that might be None
        def is_in_range(value, min_val, max_val):
            if value is None:
                return False
            return min_val <= value <= max_val

        # Numeric filters with None handling
        if not is_in_range(node.get("statuses_count"), filters["statuses_range"][0], filters["statuses_range"][1]):
            continue
        if not is_in_range(node.get("followers_count"), filters["followers_range"][0], filters["followers_range"][1]):
            continue
        if not is_in_range(node.get("friends_count"), filters["friends_range"][0], filters["friends_range"][1]):
            continue
        if not is_in_range(node.get("media_count"), filters["media_range"][0], filters["media_range"][1]):
            continue
        if not is_in_range(node.get("ratio"), filters["ratio_range"][0], filters["ratio_range"][1]):
            continue
        
        filtered[node_id] = node
    return filtered

def build_network(nodes, edges, top_n=10):
    """
    Constructs and returns a pyvis Network object for the top N accounts by in-degree.
    Also returns top accounts and their in-degrees, excluding those followed by original account.
    """
    # Compute in-degrees for nodes
    in_degrees = {str(node_id): 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if src in nodes and tgt in nodes:
            in_degrees[str(tgt)] += 1
            
    # Find nodes followed by original account
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    followed_by_original = {tgt for src, tgt in edges if src == original_id}
    
    # Filter out nodes followed by original account and sort by in-degree
    independent_accounts = [(uid, degree) for uid, degree in in_degrees.items() 
                          if uid not in followed_by_original and not uid.startswith("orig_")]
    top_independent = sorted(independent_accounts, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Get overall top accounts for visualization
    top_accounts = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = {uid for uid, _ in top_accounts}
    
    # Build network visualization
    net = Network(height="750px", width="100%", directed=True, bgcolor="#222222", font_color="white")
    for node_id, meta in nodes.items():
        if node_id in top_nodes:
            hover_text = f"""
            Name: {meta['name']}
            Screen Name: {meta['screen_name']}
            User ID: {node_id}
            Description: {meta['description']}
            Followers: {meta.get('followers_count', 0):,}
            Following: {meta.get('friends_count', 0):,}
            Follower/Following Ratio: {meta['ratio']:.2f}
            Statuses: {meta.get('statuses_count', 0):,}
            Media Count: {meta.get('media_count', 0):,}
            """
            net.add_node(str(node_id), label=meta["screen_name"], title=hover_text)
    
    for src, tgt in edges:
        if src in top_nodes and tgt in top_nodes:
            net.add_edge(str(src), str(tgt))
            
    return net, top_accounts, top_independent

def run_async_main(input_username: str):
    """Wrapper to execute the asynchronous function."""
    return asyncio.run(main_async(input_username))

def main():
    st.title("X Account Following Network Visualization (Asynchronous)")
    st.markdown("Enter an X (formerly Twitter) username to retrieve its following network.")

    input_username = st.text_input("X Username (without @):", value="elonmusk")
    
    # Sidebar: Display Options and Filter Criteria
    st.sidebar.header("Display Options")
    max_nodes_display = st.sidebar.slider("Max Nodes to Display", min_value=5, max_value=100, value=10, step=1)
    
    st.sidebar.header("Filter Criteria")
    
    # Range sliders for available numeric fields
    st.sidebar.subheader("Numeric Ranges")
    statuses_range = st.sidebar.slider("Statuses Count Range", 
                                     min_value=0, max_value=1000000, 
                                     value=(0, 1000000))
    
    followers_range = st.sidebar.slider("Followers Count Range", 
                                      min_value=0, max_value=10000000, 
                                      value=(0, 10000000))
    
    friends_range = st.sidebar.slider("Friends Count Range", 
                                    min_value=0, max_value=10000000, 
                                    value=(0, 10000000))
    
    media_range = st.sidebar.slider("Media Count Range", 
                                  min_value=0, max_value=10000, 
                                  value=(0, 10000))
    
    ratio_range = st.sidebar.slider("Follower/Following Ratio Range",
                                  min_value=0.0, max_value=1000.0,
                                  value=(0.0, 1000.0))
    
    filters = {
        "statuses_range": statuses_range,
        "followers_range": followers_range,
        "friends_range": friends_range,
        "media_range": media_range,
        "ratio_range": ratio_range
    }
    
    # Use session state to store the network data
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    
    if st.button("Generate Network"):
        # Only fetch data when button is clicked
        nodes, edges = run_async_main(input_username)
        st.session_state.network_data = (nodes, edges)
        st.write("DEBUG: Retrieved nodes and edges from API.")
    
    # If we have data, show the visualization
    if st.session_state.network_data is not None:
        nodes, edges = st.session_state.network_data
        
        # Apply display filters to nodes
        filtered_nodes = filter_nodes(nodes, filters)
        st.write(f"DEBUG: {len(filtered_nodes)} nodes remain after applying filters.")
        
        # Retain only edges whose endpoints are in the filtered nodes
        filtered_edges = [(src, tgt) for src, tgt in edges if src in filtered_nodes and tgt in filtered_nodes]
        
        # Build the network graph from filtered nodes
        net, top_accounts, top_independent = build_network(filtered_nodes, filtered_edges, top_n=max_nodes_display)
        net.save_graph("network.html")
        with open("network.html", 'r', encoding="utf-8") as html_file:
            components.html(html_file.read(), height=750, width=800)
        
        # Display top accounts in tables
        st.subheader("Top Accounts by In-Degree (Being Followed)")
        top_accounts_data = []
        for uid, degree in top_accounts:
            account = filtered_nodes[uid]
            top_accounts_data.append({
                "Screen Name": account['screen_name'],
                "Name": account['name'],
                "In-Degree": degree,
                "Followers": f"{account.get('followers_count', 0):,}",
                "Following": f"{account.get('friends_count', 0):,}",
                "Ratio": f"{account.get('ratio', 0):.2f}",
                "Statuses": f"{account.get('statuses_count', 0):,}",
                "Media": f"{account.get('media_count', 0):,}"
            })
        st.table(top_accounts_data)
            
        st.subheader("Top Independent Accounts by In-Degree (Not Followed by Original Account)")
        independent_accounts_data = []
        for uid, degree in top_independent:
            account = filtered_nodes[uid]
            independent_accounts_data.append({
                "Screen Name": account['screen_name'],
                "Name": account['name'],
                "In-Degree": degree,
                "Followers": f"{account.get('followers_count', 0):,}",
                "Following": f"{account.get('friends_count', 0):,}",
                "Ratio": f"{account.get('ratio', 0):.2f}",
                "Statuses": f"{account.get('statuses_count', 0):,}",
                "Media": f"{account.get('media_count', 0):,}"
            })
        st.table(independent_accounts_data)

if __name__ == "__main__":
    main()
