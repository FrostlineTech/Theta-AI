"""
Knowledge Graph Implementation for Theta AI

This module creates specialized knowledge graphs for technical domains to
enhance Theta's understanding of relationships between concepts.
"""

import json
import logging
import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Implements knowledge graphs for technical domains."""
    
    def __init__(self, datasets_dir: Path):
        """
        Initialize the knowledge graph system.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = datasets_dir
        
        # Create knowledge graph directory
        self.graph_dir = datasets_dir / "knowledge_graphs"
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Initialize domain-specific graphs
        self.graphs = {}
        self.entity_index = {}  # Maps entities to their properties
    
    def load_knowledge_graph(self, domain: str) -> bool:
        """
        Load a knowledge graph for a specific domain.
        
        Args:
            domain: Technical domain to load graph for
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.graph_dir / f"{domain}_knowledge_graph.json"
        
        if not file_path.exists():
            logger.warning(f"Knowledge graph for domain '{domain}' not found")
            return False
        
        try:
            # Load graph data with UTF-8 encoding and error handling
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    graph_data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try a more aggressive approach with binary reading
                logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Replace or remove non-UTF-8 bytes
                cleaned_content = b''
                for i in range(0, len(content)):
                    byte = content[i:i+1]
                    try:
                        byte.decode('utf-8')
                        cleaned_content += byte
                    except UnicodeDecodeError:
                        # Replace problematic bytes
                        cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                
                # Try parsing the cleaned content
                graph_data = json.loads(cleaned_content.decode('utf-8'))
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add entities (nodes)
            for entity_id, entity in graph_data.get("entities", {}).items():
                G.add_node(entity_id, **entity)
                self.entity_index[entity_id] = entity
            
            # Add relations (edges)
            for relation in graph_data.get("relations", []):
                source = relation.get("source")
                target = relation.get("target")
                relation_type = relation.get("type")
                weight = relation.get("weight", 1.0)
                
                if source and target:
                    G.add_edge(source, target, type=relation_type, weight=weight)
            
            # Store graph
            self.graphs[domain] = G
            
            logger.info(f"Loaded knowledge graph for domain '{domain}' with {len(G.nodes)} nodes and {len(G.edges)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph for domain '{domain}': {e}")
            return False
    
    def create_graph_from_qa(self, domain: str, qa_data: List[Dict]) -> nx.DiGraph:
        """
        Create a knowledge graph from QA data.
        
        Args:
            domain: Domain for the graph
            qa_data: List of QA pairs
            
        Returns:
            NetworkX graph
        """
        logger.info(f"Creating knowledge graph for domain '{domain}' from {len(qa_data)} QA pairs")
        
        # Create new graph
        G = nx.DiGraph()
        entities = set()
        relations = []
        
        # Process QA pairs
        for qa_pair in qa_data:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            # Extract entities and relations from Q&A
            extracted = self._extract_entities_relations(question, answer, domain)
            
            # Add to graph
            for entity in extracted["entities"]:
                entity_id = entity.get("id")
                if entity_id and entity_id not in entities:
                    entities.add(entity_id)
                    G.add_node(entity_id, **entity)
                    self.entity_index[entity_id] = entity
            
            # Collect relations
            relations.extend(extracted["relations"])
        
        # Add all relations to graph
        for relation in relations:
            source = relation.get("source")
            target = relation.get("target")
            rel_type = relation.get("type")
            weight = relation.get("weight", 1.0)
            
            if source in entities and target in entities:
                G.add_edge(source, target, type=rel_type, weight=weight)
        
        # Store graph
        self.graphs[domain] = G
        
        # Save to file
        self.save_knowledge_graph(domain)
        
        logger.info(f"Created knowledge graph for domain '{domain}' with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    
    def _extract_entities_relations(self, question: str, answer: str, domain: str) -> Dict:
        """
        Extract entities and relations from text.
        
        Args:
            question: Question text
            answer: Answer text
            domain: Knowledge domain
            
        Returns:
            Dictionary with entities and relations
        """
        # This is a simplified implementation - a real system would use NLP
        # Detect entities based on domain knowledge
        
        entities = []
        relations = []
        
        # Combine text for processing
        text = f"{question} {answer}"
        text_lower = text.lower()
        
        # Domain-specific entity detection
        if domain == "cybersecurity":
            # Detect security concepts
            for concept in ["encryption", "authentication", "authorization", "firewall", 
                           "malware", "phishing", "ransomware", "zero-day", "vulnerability"]:
                if concept in text_lower:
                    entity_id = f"concept_{concept.replace('-', '_')}"
                    entities.append({
                        "id": entity_id,
                        "name": concept,
                        "type": "concept"
                    })
                    
                    # Detect relations to attacks if appropriate
                    if concept in ["vulnerability", "zero-day"]:
                        for attack in ["exploit", "attack", "malware"]:
                            if attack in text_lower:
                                attack_id = f"concept_{attack}"
                                relations.append({
                                    "source": attack_id,
                                    "target": entity_id,
                                    "type": "exploits",
                                    "weight": 1.0
                                })
        
        elif domain == "programming":
            # Detect programming languages
            for language in ["python", "javascript", "java", "c++", "go", "rust"]:
                if language in text_lower:
                    entity_id = f"language_{language}"
                    entities.append({
                        "id": entity_id,
                        "name": language,
                        "type": "language"
                    })
                    
                    # Detect paradigms
                    for paradigm in ["object-oriented", "functional", "procedural"]:
                        if paradigm in text_lower:
                            paradigm_id = f"paradigm_{paradigm.replace('-', '_')}"
                            entities.append({
                                "id": paradigm_id,
                                "name": paradigm,
                                "type": "paradigm"
                            })
                            
                            relations.append({
                                "source": entity_id,
                                "target": paradigm_id,
                                "type": "supports",
                                "weight": 1.0
                            })
        
        elif domain == "networking":
            # Detect protocols
            for protocol in ["tcp", "udp", "http", "https", "dns", "dhcp", "smtp"]:
                if protocol.lower() in text_lower:
                    entity_id = f"protocol_{protocol}"
                    entities.append({
                        "id": entity_id,
                        "name": protocol.upper(),
                        "type": "protocol"
                    })
                    
                    # Detect OSI layers
                    for layer, layer_name in enumerate(["physical", "datalink", "network", "transport", "session", "presentation", "application"]):
                        layer_num = layer + 1
                        if f"layer {layer_num}" in text_lower or f"{layer_name} layer" in text_lower:
                            layer_id = f"layer_{layer_num}"
                            entities.append({
                                "id": layer_id,
                                "name": f"Layer {layer_num} ({layer_name})",
                                "type": "layer"
                            })
                            
                            relations.append({
                                "source": entity_id,
                                "target": layer_id,
                                "type": "operates_at",
                                "weight": 1.0
                            })
        
        return {"entities": entities, "relations": relations}
    
    def save_knowledge_graph(self, domain: str) -> bool:
        """
        Save a knowledge graph to disk.
        
        Args:
            domain: Domain to save graph for
            
        Returns:
            True if successful, False otherwise
        """
        if domain not in self.graphs:
            logger.warning(f"No knowledge graph found for domain '{domain}'")
            return False
        
        try:
            # Get graph
            G = self.graphs[domain]
            
            # Convert to serializable format
            graph_data = {
                "domain": domain,
                "entities": {},
                "relations": []
            }
            
            # Add entities (nodes)
            for node, data in G.nodes(data=True):
                graph_data["entities"][node] = data
            
            # Add relations (edges)
            for source, target, data in G.edges(data=True):
                relation = {
                    "source": source,
                    "target": target,
                    "type": data.get("type", "related_to"),
                    "weight": data.get("weight", 1.0)
                }
                graph_data["relations"].append(relation)
            
            # Save to file with UTF-8 encoding
            file_path = self.graph_dir / f"{domain}_knowledge_graph.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved knowledge graph for domain '{domain}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph for domain '{domain}': {e}")
            return False
    
    def visualize_graph(self, domain: str, output_path: Optional[Path] = None) -> bool:
        """
        Visualize a knowledge graph.
        
        Args:
            domain: Domain to visualize graph for
            output_path: Path to save visualization to (if None, use default path)
            
        Returns:
            True if successful, False otherwise
        """
        if domain not in self.graphs:
            logger.warning(f"No knowledge graph found for domain '{domain}'")
            return False
        
        try:
            # Get graph
            G = self.graphs[domain]
            
            # Set up visualization
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Draw nodes by type
            node_types = set(nx.get_node_attributes(G, 'type').values())
            colors = plt.cm.tab10(range(len(node_types)))
            color_map = dict(zip(node_types, colors))
            
            for node_type in node_types:
                nodes = [node for node, data in G.nodes(data=True) if data.get('type') == node_type]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[color_map[node_type]], 
                                      node_size=100, alpha=0.8, label=node_type)
            
            # Draw edges by type
            edge_types = set(nx.get_edge_attributes(G, 'type').values())
            
            for edge_type in edge_types:
                edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('type') == edge_type]
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, alpha=0.5, 
                                      edge_color='gray', label=edge_type)
            
            # Draw labels
            labels = {node: data.get('name', node) for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
            
            # Add legend
            plt.legend(scatterpoints=1, loc='lower right')
            
            # Set title and layout
            plt.title(f"Knowledge Graph: {domain.capitalize()}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save or show
            if output_path is None:
                output_path = self.graph_dir / f"{domain}_knowledge_graph.png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved knowledge graph visualization to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing knowledge graph for domain '{domain}': {e}")
            return False
    
    def query_knowledge_graph(self, domain: str, query: str) -> List[Dict]:
        """
        Query the knowledge graph for relevant information.
        
        Args:
            domain: Domain to query
            query: Query string
            
        Returns:
            List of relevant entities and relations
        """
        if domain not in self.graphs:
            # Try to load it
            if not self.load_knowledge_graph(domain):
                logger.warning(f"No knowledge graph found for domain '{domain}'")
                return []
        
        # Get graph
        G = self.graphs[domain]
        
        # Simple keyword matching (in a real system, use NLP or vector similarity)
        query_lower = query.lower()
        results = []
        
        # Find matching entities
        matching_nodes = []
        for node, data in G.nodes(data=True):
            name = data.get('name', '').lower()
            if name in query_lower or query_lower in name:
                matching_nodes.append(node)
        
        # For each matching node, get related information
        for node in matching_nodes:
            # Get entity data
            entity = G.nodes[node]
            
            # Get direct relations
            relations = []
            
            # Outgoing edges
            for _, target, data in G.out_edges(node, data=True):
                target_entity = G.nodes[target]
                relations.append({
                    "type": data.get('type', 'related_to'),
                    "direction": "outgoing",
                    "entity": target_entity.get('name', target),
                    "entity_id": target
                })
            
            # Incoming edges
            for source, _, data in G.in_edges(node, data=True):
                source_entity = G.nodes[source]
                relations.append({
                    "type": data.get('type', 'related_to'),
                    "direction": "incoming",
                    "entity": source_entity.get('name', source),
                    "entity_id": source
                })
            
            # Add to results
            results.append({
                "entity": entity.get('name', node),
                "type": entity.get('type', 'unknown'),
                "relations": relations
            })
        
        return results
    
    def format_graph_results(self, results: List[Dict]) -> str:
        """
        Format knowledge graph query results for human readability.
        
        Args:
            results: Query results
            
        Returns:
            Formatted string
        """
        if not results:
            return "No relevant information found in knowledge graph."
        
        formatted = "Knowledge Graph Information:\n\n"
        
        for item in results:
            formatted += f"Entity: {item['entity']} (Type: {item['type']})\n"
            formatted += "Relations:\n"
            
            if not item['relations']:
                formatted += "  - No direct relations found\n"
            else:
                for relation in item['relations']:
                    direction = "→" if relation['direction'] == "outgoing" else "←"
                    formatted += f"  - {direction} {relation['type']} {relation['entity']}\n"
            
            formatted += "\n"
        
        return formatted

def main():
    """Main function to test the knowledge graph implementation."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create knowledge graph system
    kg = KnowledgeGraph(datasets_dir)
    
    # Try to load existing graphs
    domains = ["cybersecurity", "programming", "networking"]
    
    for domain in domains:
        if kg.load_knowledge_graph(domain):
            logger.info(f"Successfully loaded {domain} knowledge graph")
            
            # Test query
            query = f"What is related to TCP" if domain == "networking" else f"What is related to Python" if domain == "programming" else "What is related to encryption"
            results = kg.query_knowledge_graph(domain, query)
            
            print(f"\nQuery: {query}")
            print(kg.format_graph_results(results))
            
            # Visualize
            kg.visualize_graph(domain)
        else:
            logger.info(f"No existing {domain} knowledge graph found. Need to create one.")

if __name__ == "__main__":
    main()
