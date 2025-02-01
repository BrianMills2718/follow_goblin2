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
def build_network_3d(nodes, edges, max_nodes=10, size_factors=None, use_pagerank=False):
    """
    Constructs a 3D ForceGraph visualization.
    size_factors is a dict containing:
        - base_size: base size for nodes
        - importance_factor: how much in-degree/pagerank affects size
        - label_size_factor: how much to scale the label size
    """
    if size_factors is None:
        size_factors = {
            'base_size': 5,
            'importance_factor': 3,
            'label_size_factor': 1
        }

    # Compute both in-degrees and PageRank
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
    
    pagerank = compute_pagerank(nodes, edges)
    
    # Determine importance metric based on selection
    importance = pagerank if use_pagerank else in_degrees
    
    # Find original node
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    
    # Find nodes followed by original account
    followed_by_original = {tgt for src, tgt in edges if src == original_id}
    
    # Get top N overall nodes (excluding original)
    top_overall = sorted(
        [(nid, score) for nid, score in importance.items() 
         if not nid.startswith("orig_")],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]  # Use half the max nodes for each category
    
    # Get top N independent nodes (not followed by original)
    top_independent = sorted(
        [(nid, score) for nid, score in importance.items() 
         if not nid.startswith("orig_") and nid not in followed_by_original],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]  # Use half the max nodes for each category
    
    # Combine selected nodes
    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}

    # Normalize importance scores for sizing
    max_importance = max(importance.values())
    normalized_importance = {
        nid: score/max_importance 
        for nid, score in importance.items()
    }

    # Build nodes array for 3D visualization
    nodes_data = []
    for node_id, meta in nodes.items():
        if node_id in selected_nodes:
            # Calculate node size using the normalized importance
            size = (size_factors['base_size'] + 
                   normalized_importance[node_id] * size_factors['importance_factor'] * 20)
            nodes_data.append({
                "id": node_id,
                "name": meta["screen_name"],
                "size": size,
                "followers": meta.get("followers_count", 0),
                "following": meta.get("friends_count", 0),
                "ratio": meta.get("ratio", 0),
                "importance": importance[node_id],
                "isFollowed": node_id in followed_by_original,
                "isOriginal": node_id.startswith("orig_")
            })

    links_data = [
        {"source": src, "target": tgt} 
        for src, tgt in edges 
        if src in selected_nodes and tgt in selected_nodes
    ]

    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)
    label_scale = size_factors['label_size_factor']

    html_code = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>3D Force Graph</title>
    <script src="https://unpkg.com/three@0.149.0/build/three.min.js"></script>
    <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
    <script src="https://unpkg.com/three-spritetext"></script>
    <style>
      body {{ margin: 0; }}
      #graph {{ width: 100%; height: 750px; }}
      .node-label {{
          font-family: Arial;
          padding: 4px 8px;
          border-radius: 4px;
          background-color: rgba(0,0,0,0.6);
          color: white;
          white-space: nowrap;
      }}
    </style>
  </head>
  <body>
    <div id="graph"></div>
    <script>
      const graphData = {{
        nodes: {nodes_json},
        links: {links_json}
      }};

      const Graph = ForceGraph3D()
        (document.getElementById('graph'))
        .graphData(graphData)
        .nodeAutoColorBy('id')
        .nodeThreeObject(node => {{
            const group = new THREE.Group();
            
            // Create node sphere
            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(node.size / 2),
                new THREE.MeshLambertMaterial({{
                    color: node.color || 0x6ca6cd,
                    transparent: true,
                    opacity: 0.75
                }})
            );
            group.add(sphere);

            // Create label sprite
            const sprite = new SpriteText(node.name);
            sprite.textHeight = node.size * {label_scale};
            sprite.color = 'white';
            sprite.backgroundColor = 'rgba(0,0,0,0.6)';
            sprite.padding = 2;
            sprite.borderRadius = 3;
            sprite.position.y = node.size / 2 + 2;
            group.add(sprite);
            
            return group;
        }})
        .nodeVal('size')
        .nodeLabel(node => 
            `<div class="node-label">
                ${{node.name}}<br>
                Followers: ${{node.followers.toLocaleString()}}<br>
                Following: ${{node.following.toLocaleString()}}<br>
                Ratio: ${{node.ratio.toFixed(2)}}
            </div>`
        )
        .linkDirectionalParticles(2)
        .linkDirectionalParticleSpeed(0.006)
        .backgroundColor("#101020");

      // Set initial camera position
      Graph.cameraPosition({{ x: 100, y: 100, z: 100 }});
      
      // Adjust force parameters
      Graph.d3Force('charge').strength(-120);
      Graph.d3Force('link').distance(50);

      // Add node click behavior for camera focus
      Graph.onNodeClick(node => {{
          const distance = 80;
          const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
          Graph.cameraPosition(
              {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
              node,
              3000
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

def create_account_table(accounts_data, start_idx=0, page_size=10):
    """
    Create a table for displaying account information.
    accounts_data should be a list of tuples: (node_id, score, node_data)
    """
    if not accounts_data:
        return st.write("No accounts to display")
    
    # Create the table data
    table_data = {
        "Rank": [],
        "Username": [],
        "Score": [],
        "Followers": [],
        "Following": [],
        "F/F Ratio": [],
        "Verified": [],
        "Location": []
    }
    
    end_idx = min(start_idx + page_size, len(accounts_data))
    for idx, (_, score, node) in enumerate(accounts_data[start_idx:end_idx], start=start_idx + 1):
        table_data["Rank"].append(idx)
        table_data["Username"].append(node["screen_name"])
        table_data["Score"].append(f"{score:.4f}")
        table_data["Followers"].append(f"{node['followers_count']:,}")
        table_data["Following"].append(f"{node['friends_count']:,}")
        table_data["F/F Ratio"].append(f"{node.get('ratio', 0):.2f}")
        table_data["Verified"].append("✓" if node.get("verified", False) else "")
        table_data["Location"].append(node.get("location", ""))
    
    st.table(table_data)
    
    return end_idx < len(accounts_data)

def run_async_main(input_username: str):
    """Wrapper to execute the asynchronous function."""
    return asyncio.run(main_async(input_username))

def main():
    st.title("X Account Following Network Visualization (Asynchronous – 3D)")
    st.markdown("Enter an X (formerly Twitter) username to retrieve its following network. The resulting network is displayed in 3D.")

    input_username = st.text_input("X Username (without @):", value="elonmusk")
    
    # Sidebar: Display Options and Filter Criteria.
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
    
    # Replace individual min/max inputs with range sliders
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
    
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
    created_range = (start_date, end_date)
    
    st.sidebar.subheader("Location Filters")
    require_location = st.sidebar.checkbox("Only accounts with non-empty location", value=False)
    
    if 'network_data' in st.session_state and st.session_state.network_data is not None:
        nodes, _ = st.session_state.network_data
        all_locations = set()
        location_map = {}
        for node in nodes.values():
            location = node.get("location")
            if location is not None and isinstance(location, str):
                loc = location.strip()
                if loc:
                    normalized = loc.lower()
                    all_locations.add(normalized)
                    location_map[normalized] = loc

        location_search = st.sidebar.text_input("Search locations", "")
        filtered_locations = []
        if location_search:
            search_term = location_search.lower()
            filtered_locations = [location_map[loc] for loc in all_locations if search_term in loc]
        else:
            filtered_locations = [location_map[loc] for loc in all_locations]

        selected_locations = st.sidebar.multiselect(
            "Select locations",
            options=sorted(filtered_locations),
            help="Select one or more locations to filter nodes. Type above to search."
        )
    else:
        selected_locations = []
    
    st.sidebar.subheader("Other Filters")
    require_blue_verified = st.sidebar.checkbox("Only blue verified accounts", value=False)
    verified_option = st.sidebar.selectbox("Verified Status", options=["Any", "Only Verified", "Only Not Verified"])
    require_website = st.sidebar.checkbox("Only accounts with website", value=False)
    business_account_option = st.sidebar.selectbox("Business Account", options=["Any", "Only Business Accounts", "Only Non-Business Accounts"])
    
    filters = {
        "statuses_range": statuses_range,
        "followers_range": followers_range,
        "friends_range": friends_range,
        "media_range": media_range,
        "created_range": created_range,
        "require_location": require_location,
        "selected_locations": selected_locations,
        "require_blue_verified": require_blue_verified,
        "verified_option": verified_option,
        "require_website": require_website,
        "business_account_option": business_account_option
    }
    
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    
    if st.button("Generate Network"):
        nodes, edges = run_async_main(input_username)
        st.session_state.network_data = (nodes, edges)
        st.write("DEBUG: Retrieved nodes and edges from API.")
    
    if st.session_state.network_data is not None:
        nodes, edges = st.session_state.network_data
        
        filtered_nodes = filter_nodes(nodes, filters)
        st.write(f"DEBUG: {len(filtered_nodes)} nodes remain after applying filters.")
        filtered_edges = [(src, tgt) for src, tgt in edges if src in filtered_nodes and tgt in filtered_nodes]
        
        size_factors = {
            'base_size': base_size,
            'importance_factor': importance_factor,
            'label_size_factor': label_size_factor
        }
        
        html_code = build_network_3d(
            filtered_nodes, 
            filtered_edges,
            max_nodes=max_nodes_display,  # Pass max_nodes_display here
            size_factors=size_factors,
            use_pagerank=use_pagerank
        )
        components.html(html_code, height=750, width=800)
        
        # Update the display of top accounts to use max_nodes_display
        importance = compute_pagerank(filtered_nodes, filtered_edges) if use_pagerank else {
            node_id: sum(1 for _, tgt in filtered_edges if tgt == node_id)
            for node_id in filtered_nodes
        }
        
        original_id = next(id for id in filtered_nodes.keys() if id.startswith("orig_"))
        followed_by_original = {tgt for src, tgt in filtered_edges if src == original_id}
        
        # Initialize session state for pagination if not exists
        if 'overall_page' not in st.session_state:
            st.session_state.overall_page = 0
        if 'independent_page' not in st.session_state:
            st.session_state.independent_page = 0
        
        # Prepare account data with all relevant information
        overall_accounts = [
            (nid, score, filtered_nodes[nid]) 
            for nid, score in sorted(
                [(nid, score) for nid, score in importance.items() if not nid.startswith("orig_")],
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        independent_accounts = [
            (nid, score, filtered_nodes[nid]) 
            for nid, score in sorted(
                [(nid, score) for nid, score in importance.items() 
                 if not nid.startswith("orig_") and nid not in followed_by_original],
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        # Display overall top accounts
        st.subheader(f"Top Accounts by {'PageRank' if use_pagerank else 'In-Degree'}")
        col1, col2 = st.columns([4, 1])
        with col1:
            has_more_overall = create_account_table(
                overall_accounts, 
                start_idx=st.session_state.overall_page * 10
            )
        with col2:
            if has_more_overall:
                if st.button("Next 10 ▶", key="next_overall"):
                    st.session_state.overall_page += 1
            if st.session_state.overall_page > 0:
                if st.button("◀ Previous", key="prev_overall"):
                    st.session_state.overall_page -= 1
        
        # Display independent top accounts
        st.subheader("Top Independent Accounts (Not Followed by Original)")
        col1, col2 = st.columns([4, 1])
        with col1:
            has_more_independent = create_account_table(
                independent_accounts, 
                start_idx=st.session_state.independent_page * 10
            )
        with col2:
            if has_more_independent:
                if st.button("Next 10 ▶", key="next_independent"):
                    st.session_state.independent_page += 1
            if st.session_state.independent_page > 0:
                if st.button("◀ Previous", key="prev_independent"):
                    st.session_state.independent_page -= 1

        # Add reset button for pagination
        if st.button("Reset Tables"):
            st.session_state.overall_page = 0
            st.session_state.independent_page = 0
            st.experimental_rerun()

if __name__ == "__main__":
    main()
