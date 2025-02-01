#!/usr/bin/env python3
# ASYNCHRONOUS 3D NETWORK VISUALIZATION OF X ACCOUNT FOLLOWING NETWORK
#
# This script retrieves the "following" network of an input X (formerly Twitter) account asynchronously
# and visualizes it in an interactive 3D graph with permanent labels and detailed information tables.
#
# FEATURES:
#  1. 3D Force-Directed Graph visualization with:
#     - Permanent node labels that scale with node size
#     - Detailed hover information
#     - Interactive camera controls and node focusing
#  2. Node importance calculated using either PageRank or In-Degree
#  3. Configurable node and label sizes
#  4. Display filters for:
#     - statuses_count, followers_count, friends_count, media_count
#     - created_at date range
#     - location (with search)
#     - verification status
#     - website presence
#     - business account status
#  5. Paginated tables showing:
#     - Top accounts by importance (PageRank/In-Degree)
#     - Top independent accounts (not followed by original account)


import streamlit as st
import streamlit.components.v1 as components
import asyncio
import aiohttp
import json
from pyvis.network import Network
import datetime
import numpy as np
from scipy import sparse
from openai import OpenAI
from typing import List, Dict
import colorsys
import random
from tqdm import tqdm  # For progress tracking
from dotenv import load_dotenv
import os

# CONSTANTS
RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
RAPIDAPI_HOST = "twitter-api45.p.rapidapi.com"

# Add these constants after the existing RAPIDAPI constants
OPENROUTER_API_KEY = "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"
COMMUNITY_COLORS = {}  # Will be populated dynamically

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
    return followers_count / friends_count if friends_count else 0

def compute_pagerank(nodes, edges, damping=0.85, epsilon=1e-8, max_iter=100):
    """
    Compute PageRank for each node in the network.
    """
    # Create node index mapping
    node_to_index = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
    n = len(nodes)
    
    # Create adjacency matrix
    rows, cols = [], []
    for src, tgt in edges:
        if src in node_to_index and tgt in node_to_index:
            rows.append(node_to_index[src])
            cols.append(node_to_index[tgt])
    
    # Create sparse matrix
    data = np.ones_like(rows)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Normalize adjacency matrix
    out_degree = np.array(A.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1  # Avoid division by zero
    A = sparse.diags(1/out_degree) @ A
    
    # Initialize PageRank
    pr = np.ones(n) / n
    
    # Power iteration
    for _ in range(max_iter):
        pr_next = (1 - damping) / n + damping * A.T @ pr
        if np.sum(np.abs(pr_next - pr)) < epsilon:
            break
        pr = pr_next
    
    # Convert back to dictionary
    return {node_id: pr[idx] for node_id, idx in node_to_index.items()}

async def main_async(input_username: str):
    """
    Retrieves and processes the following network asynchronously.
    Enhanced to store additional account attributes for filtering.
    """
    nodes, edges = {}, []
    original_id = f"orig_{input_username}"
    # The original node: minimal attributes.
    nodes[original_id] = {
        "screen_name": input_username,
        "name": input_username,
        "followers_count": None,
        "friends_count": None,
        "statuses_count": None,
        "media_count": None,
        "created_at": None,
        "location": None,
        "blue_verified": None,
        "verified": None,
        "website": None,
        "business_account": None,
        "ratio": None,
        "description": "",  # Add empty description for original node
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
                "created_at": account.get("created_at", ""),
                "location": account.get("location", ""),
                "blue_verified": account.get("blue_verified", False),
                "verified": account.get("verified", False),
                "website": account.get("website", ""),
                "business_account": account.get("business_account", False),
                "description": account.get("description", ""),  # Add description field
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
                        "created_at": account.get("created_at", ""),
                        "location": account.get("location", ""),
                        "blue_verified": account.get("blue_verified", False),
                        "verified": account.get("verified", False),
                        "website": account.get("website", ""),
                        "business_account": account.get("business_account", False),
                        "description": account.get("description", ""),  # Add description field
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
        # Always include the original node.
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

        # Location filters
        location = node.get("location")
        if filters["selected_locations"]:
            if location is not None and isinstance(location, str) and location.strip():
                location = location.strip().lower()
                if not any(loc.lower() in location for loc in filters["selected_locations"]):
                    continue
            else:
                continue
        elif filters["require_location"]:
            if not location or not isinstance(location, str) or not location.strip():
                continue

        # Blue verified filter.
        if filters["require_blue_verified"]:
            if not node.get("blue_verified", False):
                continue

        # Verified filter.
        if filters["verified_option"] == "Only Verified":
            if not node.get("verified", False):
                continue
        elif filters["verified_option"] == "Only Not Verified":
            if node.get("verified", False):
                continue

        # Website filter.
        if filters["require_website"]:
            if not node.get("website", "").strip():
                continue

        # Business account filter.
        if filters["business_account_option"] == "Only Business Accounts":
            if not node.get("business_account", False):
                continue
        elif filters["business_account_option"] == "Only Non-Business Accounts":
            if node.get("business_account", False):
                continue
        
        filtered[node_id] = node
    return filtered

# ---------------------------------------------------------------------
# NEW: Build a 3D network visualization using ForceGraph3D.
# ---------------------------------------------------------------------
def get_openai_client():
    """Initialize OpenAI client."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

async def get_community_labels(accounts: List[Dict], num_communities: int) -> List[str]:
    """Get community labels from GPT-4o-mini."""
    client = get_openai_client()
    
    # Create prompt with account information
    account_info = "\n".join([
        f"Username: {acc['screen_name']}, Description: {acc['description']}"
        for acc in accounts[:20]  # Use first 20 accounts as examples
    ])
    
    prompt = f"""Based on these X/Twitter accounts and their descriptions:

{account_info}

Generate exactly {num_communities} distinct community labels that best categorize these and similar accounts.
Include an "Other" category as one of the {num_communities} labels. Return only the labels, one per line.
Focus on professional/topical communities rather than demographic categories."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes social media accounts into communities."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    
    # Parse response into list of labels
    labels = [label.strip() for label in response.choices[0].message.content.split('\n') if label.strip()]
    return labels

async def classify_accounts(accounts: List[Dict], labels: List[str], batch_size=50) -> Dict[str, str]:
    """Classify accounts into communities in parallel batches."""
    client = get_openai_client()
    results = {}
    
    # Create a container for progress indicators
    progress_container = st.container()
    with progress_container:
        st.write("### Community Classification Progress")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        batch_status = st.empty()
        
        # Show initial stats
        total_batches = (len(accounts) + batch_size - 1) // batch_size
        st.write(f"Total accounts to process: {len(accounts)}")
        st.write(f"Number of batches: {total_batches}")
        
        # Create a placeholder for batch status updates
        batch_updates = st.empty()
        completed_batches = set()

    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(10)  # Process up to 10 batches concurrently
    
    async def process_batch(batch, batch_num):
        async with semaphore:  # Limit concurrent processing
            accounts_info = "\n".join([
                f"Username: {acc['screen_name']}, Description: {acc['description']}"
                for acc in batch
            ])
            
            labels_str = "\n".join(labels)
            prompt = f"""Given these community labels:
{labels_str}

Classify each of these accounts into exactly one of the above communities.
Only use the exact community labels provided above, do not create new ones.
If unsure, use the 'Other' category.

Accounts to classify:
{accounts_info}

Return in format:
username: community_label"""

            try:
                # Update status before processing
                with batch_status:
                    st.write(f"Processing batch {batch_num + 1}...")
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that classifies social media accounts into predefined communities. Only use the exact community labels provided."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                )
                
                # Parse response into dictionary
                classifications = {}
                for line in response.choices[0].message.content.split('\n'):
                    if ':' in line:
                        username, label = line.split(':', 1)
                        label = label.strip()
                        # Only store valid community labels
                        if label in labels:
                            classifications[username.strip()] = label
                        else:
                            # If invalid label, use "Other"
                            classifications[username.strip()] = "Other"
                
                completed_batches.add(batch_num)
                
                # Update batch status
                with batch_updates:
                    st.write(f"✅ Completed batch {batch_num + 1}/{total_batches}")
                
                return batch_num, classifications
            except Exception as e:
                with batch_updates:
                    st.error(f"❌ Error in batch {batch_num + 1}: {str(e)}")
                return batch_num, {}

    # Create batches with larger batch size
    batches = []
    for i in range(0, len(accounts), batch_size):
        batch = accounts[i:i + batch_size]
        batches.append((batch, i // batch_size))
    
    # Process all batches in parallel with high concurrency
    tasks = [process_batch(batch, batch_num) for batch, batch_num in batches]
    batch_results = await asyncio.gather(*tasks)
    
    # Update progress as results come in
    for batch_num, classifications in sorted(batch_results, key=lambda x: x[0]):
        results.update(classifications)
        progress = (batch_num + 1) / total_batches
        progress_text.text(f"Overall Progress: {progress:.1%}")
        progress_bar.progress(progress)
    
    # Show final status
    with progress_container:
        if len(completed_batches) == total_batches:
            st.success(f"✅ Classification complete! Processed {len(accounts)} accounts in {total_batches} batches.")
        else:
            st.warning(f"⚠️ Classification partially complete. {len(completed_batches)}/{total_batches} batches processed.")
    
    return results

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.uniform(-0.2, 0.2)
        value = 0.9 + random.uniform(-0.2, 0.2)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def build_network_3d(nodes, edges, max_nodes=10, size_factors=None, use_pagerank=False):
    """
    Constructs a 3D ForceGraph visualization with permanent labels and hover info.
    Updated to apply max_nodes selection similar to the build_network_2d function.
    """
    # Set default size factors if None
    if size_factors is None:
        size_factors = {
            'base_size': 5,
            'importance_factor': 3.0,
            'label_size_factor': 1.0
        }

    # Determine node importance
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
            
    pagerank = compute_pagerank(nodes, edges)
    importance = pagerank if use_pagerank else in_degrees

    # Identify the original node
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    followed_by_original = {tgt for src, tgt in edges if src == original_id}

    # Select top nodes based on the importance score
    top_overall = sorted(
        [(nid, score) for nid, score in importance.items() if not nid.startswith("orig_")],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes // 2]

    top_independent = sorted(
        [(nid, score) for nid, score in importance.items()
         if not nid.startswith("orig_") and nid not in followed_by_original],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes // 2]

    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}

    # Filter nodes and edges to only include selected nodes
    nodes = {node_id: meta for node_id, meta in nodes.items() if node_id in selected_nodes}
    edges = [(src, tgt) for src, tgt in edges if src in selected_nodes and tgt in selected_nodes]

    nodes_data = []
    links_data = []

    # Convert edges to proper format
    links_data = [{"source": str(src), "target": str(tgt)} for src, tgt in edges]

    # Convert nodes to proper format with additional info
    for node_id, meta in nodes.items():
        try:
            base_size = float(size_factors.get('base_size', 5))
            importance_factor = float(size_factors.get('importance_factor', 3.0))
            
            # Handle None values for followers_count
            followers_count = meta.get("followers_count")
            if followers_count is None:
                followers_count = 0 if node_id.startswith("orig_") else 1000  # Default value for non-original nodes
            
            # Calculate node size with type checking
            followers_factor = float(followers_count) / 1000.0
            node_size = base_size + followers_factor * importance_factor
            
            # Ensure node_size is positive
            node_size = max(1.0, node_size)
            
            # Handle None values for other metrics
            following_count = meta.get("friends_count", 0)
            if following_count is None:
                following_count = 0
                
            ratio = meta.get("ratio", 0.0)
            if ratio is None:
                ratio = 0.0
                
            # Get community color if it exists
            community_color = meta.get("community_color", "#6ca6cd")  # Use default color if no community
            
            # Get community information
            username = meta.get("screen_name", "")
            community = "N/A"
            if ('node_communities' in st.session_state and 
                st.session_state.node_communities and 
                username in st.session_state.node_communities):
                community = st.session_state.node_communities[username]
            
            nodes_data.append({
                "id": str(node_id),
                "name": str(meta.get("screen_name", "")),
                "community": community,
                "followers": int(followers_count),
                "following": int(following_count),
                "ratio": float(ratio),
                "size": float(node_size),
                "description": str(meta.get("description", "")),
                "pagerank": float(importance.get(node_id, 0)),
                "indegree": int(in_degrees.get(node_id, 0)),
                "color": community_color
            })
        except Exception as e:
            st.write(f"Warning: Error processing node {node_id}: {str(e)}")
            nodes_data.append({
                "id": str(node_id),
                "name": str(meta.get("screen_name", "")),
                "community": "N/A",
                "followers": 0,
                "following": 0,
                "ratio": 0.0,
                "size": float(size_factors.get('base_size', 5)),
                "description": "",
                "pagerank": 0.0,
                "indegree": 0,
                "color": "#6ca6cd"
            })

    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)

    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <script src="https://unpkg.com/three@0.149.0/build/three.min.js"></script>
        <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
        <script src="https://unpkg.com/three-spritetext"></script>
        <style>
          #graph {{ width: 100%; height: 750px; }}
          .node-tooltip {{
              font-family: Arial;
              padding: 8px;
              border-radius: 4px;
              background-color: rgba(0,0,0,0.8);
              color: white;
              white-space: pre-line;
              font-size: 14px;
          }}
        </style>
      </head>
      <body>
        <div id="graph"></div>
        <script>
          const data = {{
            nodes: {nodes_json},
            links: {links_json}
          }};
          
          console.log("Graph data:", data);
          
          const Graph = ForceGraph3D()
            (document.getElementById('graph'))
            .graphData(data)
            .nodeColor(node => node.color)
            .nodeRelSize(6)
            .nodeVal(node => node.size)
            .nodeThreeObject(node => {{
                const group = new THREE.Group();
                
                const sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(Math.cbrt(node.size)),
                    new THREE.MeshLambertMaterial({{
                        color: node.color,
                        transparent: true,
                        opacity: 0.75
                    }})
                );
                group.add(sphere);
                
                const sprite = new SpriteText(node.name);
                sprite.textHeight = Math.max(4, Math.min(12, 8 * Math.cbrt(node.size / 10))) * 
                                  {size_factors.get('label_size_factor', 1.0)};
                sprite.color = 'white';
                sprite.backgroundColor = 'rgba(0,0,0,0.6)';
                sprite.padding = 2;
                sprite.borderRadius = 3;
                sprite.position.y = Math.cbrt(node.size) + 1;
                group.add(sprite);
                
                return group;
            }})
            .nodeLabel(node => {{
                return `<div class="node-tooltip">
                    <b>@${{node.name}}</b><br/>
                    Community: ${{node.community}}<br/>
                    Followers: ${{node.followers.toLocaleString()}}<br/>
                    Following: ${{node.following.toLocaleString()}}<br/>
                    Ratio: ${{node.ratio.toFixed(2)}}<br/>
                    Description: ${{node.description}}<br/>
                    PageRank: ${{node.pagerank.toFixed(4)}}<br/>
                    In-Degree: ${{node.indegree}}
                    </div>`;
            }})
            .linkDirectionalParticles(1)
            .linkDirectionalParticleSpeed(0.006)
            .backgroundColor("#101020");

          // Set initial camera position
          Graph.cameraPosition({{ x: 150, y: 150, z: 150 }});

          // Adjust force parameters for better layout
          Graph.d3Force('charge').strength(-120);
          
          // Add node click behavior for camera focus
          Graph.onNodeClick(node => {{
              const distance = 40;
              const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
              Graph.cameraPosition(
                  {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
                  node,
                  2000
              );
          }});
        </script>
      </body>
    </html>
    """
    return html_code


# ---------------------------------------------------------------------
# End new 3D visualization code
# ---------------------------------------------------------------------

def build_network_2d(nodes, edges, max_nodes=10, size_factors=None, use_pagerank=False):
    """
    Constructs a 2D network visualization using pyvis.
    Uses the same parameters as build_network_3d for consistency.
    """
    # Use same importance calculation as 3D version
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
    
    pagerank = compute_pagerank(nodes, edges)
    importance = pagerank if use_pagerank else in_degrees
    
    # Find original node and followed nodes
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    followed_by_original = {tgt for src, tgt in edges if src == original_id}
    
    # Select nodes same way as 3D version
    top_overall = sorted(
        [(nid, score) for nid, score in importance.items() 
         if not nid.startswith("orig_")],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]
    
    top_independent = sorted(
        [(nid, score) for nid, score in importance.items() 
         if not nid.startswith("orig_") and nid not in followed_by_original],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]
    
    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}

    # Create pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Normalize importance scores
    max_importance = max(importance.values())
    normalized_importance = {nid: score/max_importance for nid, score in importance.items()}
    
    # Add nodes
    for node_id in selected_nodes:
        size = (size_factors['base_size'] +
                normalized_importance[node_id] * size_factors['importance_factor'] * 20)
        
        # Safely format numeric values by checking for None
        followers = nodes[node_id].get('followers_count')
        followers_str = f"{followers:,}" if isinstance(followers, int) else "0"
        friends = nodes[node_id].get('friends_count')
        friends_str = f"{friends:,}" if isinstance(friends, int) else "0"
        ratio = nodes[node_id].get('ratio')
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "0.00"
        description = nodes[node_id].get('description') or ""
        title = (f"Followers: {followers_str}\n"
                 f"Following: {friends_str}\n"
                 f"Ratio: {ratio_str}\n"
                 f"Description: {description}")

        color = nodes[node_id].get("community_color", "#6ca6cd")

        net.add_node(
            node_id,
            label=nodes[node_id]["screen_name"],
            title=title,
            size=size,
            color=color
        )
    
    # Add edges between selected nodes
    for src, tgt in edges:
        if src in selected_nodes and tgt in selected_nodes:
            net.add_edge(src, tgt)
    
    return net

def create_account_table(accounts_data, start_idx=0, page_size=10):
    """
    Create a table for displaying account information.
    accounts_data should be a list of tuples: (node_id, score, node_data)
    """
    if not accounts_data:
        return st.write("No accounts to display")
    
    # Update table columns to include description
    table_data = {
        "Rank": [],
        "Username": [],
        "Score": [],
        "Followers": [],
        "Following": [],
        "F/F Ratio": [],
        "Description": []  # Include description in the table
    }
    
    end_idx = min(start_idx + page_size, len(accounts_data))
    for idx, (_, score, node) in enumerate(accounts_data[start_idx:end_idx], start=start_idx + 1):
        table_data["Rank"].append(idx)
        table_data["Username"].append(node["screen_name"])
        table_data["Score"].append(f"{score:.4f}")
        table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
        table_data["Following"].append(f"{node.get('friends_count', 0):,}")
        table_data["F/F Ratio"].append(f"{node.get('ratio', 0):.2f}")
        table_data["Description"].append(node.get("description", ""))  # Add description to the table
    
    st.table(table_data)
    
    return end_idx < len(accounts_data)

def run_async_main(input_username: str):
    """Wrapper to execute the asynchronous function."""
    return asyncio.run(main_async(input_username))

def get_top_accounts_by_community(nodes: Dict, node_communities: Dict, importance_scores: Dict, top_n: int = 10) -> Dict[str, List]:
    """Get top accounts for each community based on importance scores."""
    community_accounts = {}
    
    # Group accounts by community
    for node_id, node in nodes.items():
        if node_id.startswith("orig_"):
            continue
            
        username = node["screen_name"]
        if username in node_communities:
            community = node_communities[username]
            if community not in community_accounts:
                community_accounts[community] = []
            community_accounts[community].append((node_id, node, importance_scores.get(node_id, 0)))
    
    # Sort accounts within each community by importance
    top_accounts = {}
    for community, accounts in community_accounts.items():
        sorted_accounts = sorted(accounts, key=lambda x: x[2], reverse=True)[:top_n]
        top_accounts[community] = sorted_accounts
    
    return top_accounts

def display_community_tables(top_accounts: Dict[str, List], community_colors: Dict[str, str]):
    """Display tables of top accounts for each community."""
    for community, accounts in top_accounts.items():
        color = community_colors.get(community, "#6ca6cd")
        
        st.markdown(
            f'<div style="background-color: {color}; padding: 10px; '
            f'border-radius: 5px; margin: 10px 0; color: black;">'
            f'<h3>{community}</h3></div>',
            unsafe_allow_html=True
        )
        
        if not accounts:
            st.write("No accounts in this community")
            continue
            
        table_data = {
            "Username": [],
            "Followers": [],
            "Following": [],
            "Score": []
        }
        
        for _, node, score in accounts:
            table_data["Username"].append(node["screen_name"])
            table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
            table_data["Following"].append(f"{node.get('friends_count', 0):,}")
            table_data["Score"].append(f"{score:.4f}")
        
        st.table(table_data)

def display_top_accounts_table(nodes: Dict, edges: List, importance_scores: Dict, original_id: str, 
                             exclude_first_degree: bool = False, top_n: int = 20):
    """Display table of top accounts overall."""
    # Get first degree connections if needed for exclusion
    first_degree = {tgt for src, tgt in edges if src == original_id} if exclude_first_degree else set()
    
    # Sort accounts by importance score
    sorted_accounts = []
    for node_id, node in nodes.items():
        if node_id.startswith("orig_"):
            continue
        if exclude_first_degree and node_id in first_degree:
            continue
        sorted_accounts.append((node_id, node, importance_scores.get(node_id, 0)))
    
    sorted_accounts.sort(key=lambda x: x[2], reverse=True)
    top_accounts = sorted_accounts[:top_n]
    
    # Create and display table
    st.subheader(f"Top {top_n} Accounts Overall{'(Excluding First Degree)' if exclude_first_degree else ''}")
    
    table_data = {
        "Rank": [],
        "Username": [],
        "Community": [],  # Add community column
        "Score": [],
        "Followers": [],
        "Following": [],
        "F/F Ratio": []
    }
    
    for idx, (node_id, node, score) in enumerate(top_accounts, 1):
        username = node["screen_name"]
        community = "N/A"
        # Check if node_communities exists in session state before accessing it
        if 'node_communities' in st.session_state and st.session_state.node_communities:
            if username in st.session_state.node_communities:
                community = st.session_state.node_communities[username]
        
        table_data["Rank"].append(idx)
        table_data["Username"].append(username)
        table_data["Community"].append(community)
        table_data["Score"].append(f"{score:.4f}")
        table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
        table_data["Following"].append(f"{node.get('friends_count', 0):,}")
        table_data["F/F Ratio"].append(f"{node.get('ratio', 0):.2f}")
    
    st.table(table_data)

def main():
    # Initialize session state variables if they don't exist
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    if 'community_labels' not in st.session_state:
        st.session_state.community_labels = None
    if 'community_colors' not in st.session_state:
        st.session_state.community_colors = None
    if 'node_communities' not in st.session_state:
        st.session_state.node_communities = None
    if 'use_3d' not in st.session_state:
        st.session_state.use_3d = True

    st.title("X Account Following Network Visualization")
    st.markdown("Enter an X (formerly Twitter) username to retrieve its following network.")

    input_username = st.text_input("X Username (without @):", value="elonmusk")
    
    # Sidebar: Display Options and Filter Criteria
    st.sidebar.header("Display Options")
    
    # Node and Label Size Controls
    st.sidebar.subheader("Size Controls")
    use_pagerank = st.sidebar.checkbox("Use PageRank for Importance", value=False)
    base_size = st.sidebar.slider("Base Node Size", 
                                min_value=1, 
                                max_value=20, 
                                value=5)
    
    importance_factor = st.sidebar.slider(
        "PageRank Factor" if use_pagerank else "In-Degree Factor", 
        min_value=0.1, 
        max_value=10.0, 
        value=3.0
    )
    
    # Move label_size_factor control here and make it conditional on use_3d
    label_size_factor = 1.0  # Default value
    if st.session_state.network_data is not None:  # Only show after network is generated
        st.session_state.use_3d = st.checkbox("Use 3D Visualization", value=st.session_state.use_3d)
        if st.session_state.use_3d:  # Show label size control only for 3D visualization
            label_size_factor = st.sidebar.slider("Label Size Factor", 
                                                min_value=0.1, 
                                                max_value=5.0, 
                                                value=1.0)
    
    max_nodes_display = st.sidebar.slider("Max Nodes to Display", 
                                        min_value=5, 
                                        max_value=1000, 
                                        value=50,
                                        step=5)
    
    st.sidebar.header("Filter Criteria")
    
    # Keep only the numeric filters
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
    
    filters = {
        "statuses_range": statuses_range,
        "followers_range": followers_range,
        "friends_range": friends_range,
        "media_range": media_range,
        "created_range": (datetime.date(2000, 1, 1), datetime.date(2100, 1, 1)),
        "require_location": False,
        "selected_locations": [],
        "require_blue_verified": False,
        "verified_option": "Any",
        "require_website": False,
        "business_account_option": "Any"
    }
    
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    
    if st.button("Generate Network"):
        nodes, edges = run_async_main(input_username)
        # Add debug information
        st.write(f"DEBUG: Retrieved {len(nodes)} nodes and {len(edges)} edges from API.")
        # Check for None values in followers_count
        none_followers = [node_id for node_id, data in nodes.items() 
                         if data.get('followers_count') is None]
        if none_followers:
            st.write(f"WARNING: Found {len(none_followers)} nodes with None followers_count")
        
        st.session_state.network_data = (nodes, edges)
    
    if st.session_state.network_data is not None:
        nodes, edges = st.session_state.network_data
        filtered_nodes = filter_nodes(nodes, filters)
        
        # Add degree filtering
        st.sidebar.subheader("Node Degree Filtering")
        show_original = st.sidebar.checkbox("Show Original Node", value=True)
        show_first_degree = st.sidebar.checkbox("Show First Degree Connections", value=True)
        show_second_degree = st.sidebar.checkbox("Show Second Degree Connections", value=True)
        
        # Filter nodes by degree
        original_id = next(id for id in filtered_nodes.keys() if id.startswith("orig_"))
        first_degree = {tgt for src, tgt in edges if src == original_id}
        second_degree = {tgt for src, tgt in edges if src in first_degree} - first_degree - {original_id}
        
        degree_filtered_nodes = {}
        for node_id, node in filtered_nodes.items():
            if node_id == original_id and show_original:
                degree_filtered_nodes[node_id] = node
            elif node_id in first_degree and show_first_degree:
                degree_filtered_nodes[node_id] = node
            elif node_id in second_degree and show_second_degree:
                degree_filtered_nodes[node_id] = node
        
        # Add community filtering
        if st.session_state.community_labels and st.session_state.community_colors:
            st.sidebar.subheader("Community Filtering")
            selected_communities = {}
            for label, color in st.session_state.community_colors.items():
                selected_communities[label] = st.sidebar.checkbox(
                    label,
                    value=True,
                    key=f"community_{label}"
                )
            
            # Filter nodes by selected communities
            community_filtered_nodes = {}
            for node_id, node in degree_filtered_nodes.items():
                if node_id.startswith("orig_"):
                    community_filtered_nodes[node_id] = node
                    continue
                    
                username = node["screen_name"]
                if username in st.session_state.node_communities:
                    community = st.session_state.node_communities[username]
                    if selected_communities.get(community, True):
                        community_filtered_nodes[node_id] = node
            
            filtered_nodes = community_filtered_nodes
        else:
            filtered_nodes = degree_filtered_nodes
        
        filtered_edges = [(src, tgt) for src, tgt in edges 
                         if src in filtered_nodes and tgt in filtered_nodes]
        
        # Update node colors based on communities
        if st.session_state.community_labels and st.session_state.community_colors:
            for node_id in filtered_nodes:
                username = filtered_nodes[node_id]["screen_name"]
                if username in st.session_state.node_communities:
                    community = st.session_state.node_communities[username]
                    # Add debug check
                    if community not in st.session_state.community_colors:
                        st.warning(f"Invalid community label found: {community}")
                        community = "Other"
                    filtered_nodes[node_id]["community_color"] = st.session_state.community_colors[community]
        
        # Update size_factors dictionary
        size_factors = {
            'base_size': float(base_size),
            'importance_factor': float(importance_factor),
            'label_size_factor': float(label_size_factor)
        }
        
        # Display the graph
        if st.session_state.use_3d:
            html_code = build_network_3d(
                filtered_nodes, 
                filtered_edges,
                max_nodes=max_nodes_display,
                size_factors=size_factors,
                use_pagerank=use_pagerank
            )
            st.write("Debug: About to render 3D graph")
            components.html(html_code, height=750, width=800)
            st.write("Debug: Finished rendering")
        else:
            net = build_network_2d(
                filtered_nodes, 
                filtered_edges,
                max_nodes=max_nodes_display,
                size_factors=size_factors,
                use_pagerank=use_pagerank
            )
            net.save_graph("network.html")
            with open("network.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=750, width=800)

        # Move community color key here, right after the graph
        if st.session_state.community_labels and st.session_state.community_colors:
            st.subheader("Community Color Key")
            cols = st.columns(len(st.session_state.community_colors))
            for idx, (label, color) in enumerate(st.session_state.community_colors.items()):
                with cols[idx]:
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 10px; '
                        f'border-radius: 5px; margin: 2px 0; color: black; '
                        f'text-align: center;">{label}</div>',
                        unsafe_allow_html=True
                    )

        # Add community detection controls
        st.header("Community Detection")
        num_communities = st.number_input(
            "Number of Communities (including 'Other')", 
            min_value=2, 
            max_value=10, 
            value=6
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Community Labels"):
                with st.spinner("Generating community labels..."):
                    # Get community labels
                    community_labels = asyncio.run(get_community_labels(
                        [node for node in filtered_nodes.values()],
                        num_communities
                    ))
                    
                    # Store labels in session state
                    st.session_state.community_labels = community_labels
                    
                    # Display the generated labels
                    st.write("Generated Community Labels:")
                    for i, label in enumerate(community_labels, 1):
                        st.write(f"{i}. {label}")
        
        with col2:
            # Only show assign button if labels exist
            if st.session_state.community_labels:
                if st.button("Assign Accounts to Communities"):
                    with st.spinner("Assigning accounts to communities..."):
                        # Generate colors for communities
                        colors = generate_distinct_colors(len(st.session_state.community_labels))
                        community_colors = dict(zip(st.session_state.community_labels, colors))
                        
                        # Classify accounts
                        node_communities = asyncio.run(classify_accounts(
                            list(filtered_nodes.values()),
                            st.session_state.community_labels
                        ))
                        
                        # Store results in session state
                        st.session_state.community_colors = community_colors
                        st.session_state.node_communities = node_communities
                        
                        # Force a rerun to update the visualization
                        st.rerun()

        # Display overall analysis section
        st.header("Network Analysis")
        
        # Calculate importance scores
        importance_scores = compute_pagerank(filtered_nodes, filtered_edges) if use_pagerank else {
            node_id: sum(1 for _, tgt in filtered_edges if tgt == node_id)
            for node_id in filtered_nodes
        }
        
        # Add toggle for excluding first-degree follows
        exclude_first_degree = st.checkbox("Exclude First Degree Follows from Top Accounts", value=False)
        
        # Display top accounts table
        display_top_accounts_table(
            filtered_nodes,
            filtered_edges,
            importance_scores,
            original_id,
            exclude_first_degree
        )
        
        # Only show community tables if communities have been assigned
        if ('node_communities' in st.session_state and 
            st.session_state.node_communities and 
            'community_colors' in st.session_state and 
            st.session_state.community_colors):
            
            st.header("Community Analysis")
            # Get and display top accounts by community
            top_accounts = get_top_accounts_by_community(
                filtered_nodes,
                st.session_state.node_communities,
                importance_scores
            )
            
            display_community_tables(top_accounts, st.session_state.community_colors)

if __name__ == "__main__":
    main()
