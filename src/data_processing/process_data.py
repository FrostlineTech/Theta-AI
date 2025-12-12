"""
Process data for Theta AI training.
Combines multiple datasets into a single processed file.
"""

import json
import os
import re
import glob
from pathlib import Path
import logging
from src.data_processing.create_conversation_dataset import main as create_conversation_datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_dataset(file_path):
    """Load a dataset from a JSON file."""
    try:
        try:
            # First try with UTF-8 encoding and error replacement
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If that fails, try a more aggressive approach with binary reading
            logger.warning(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
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
            data = json.loads(cleaned_content.decode('utf-8'))
        
        # Validate structure
        if isinstance(data, list):
            # Standard QA format
            return [item for item in data if isinstance(item, dict) and 'question' in item and 'answer' in item]
        elif isinstance(data, dict):
            # Try to convert nested structure to QA format
            try:
                # Import only when needed to avoid circular imports
                from src.data_processing.structured_data_adapter import nested_to_qa
                qa_data = nested_to_qa(data)
                if qa_data:
                    print(f"Converted nested structure from {file_path} to {len(qa_data)} QA pairs")
                    return qa_data
            except ImportError:
                print(f"Warning: structured_data_adapter not available for {file_path}")
            except Exception as e:
                print(f"Warning: Could not convert nested structure in {file_path}: {e}")
        
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def enhance_with_cybersecurity_examples():
    """Add cybersecurity examples to enhance the dataset."""
    return [
        {
            "question": "What is defense in depth?",
            "answer": "Defense in depth is a cybersecurity strategy that employs multiple layers of security controls throughout an IT system. Rather than relying on a single security measure, it uses various mechanisms at different layers to protect assets. If one security control fails, others still provide protection. Examples include firewalls, IDS/IPS, antivirus, access controls, encryption, and security awareness training."
        },
        {
            "question": "What is a zero-day vulnerability?",
            "answer": "A zero-day vulnerability is a software security flaw that is unknown to those who should be interested in mitigating the vulnerability (including the vendor of the target software). Until the vulnerability is mitigated, hackers can exploit it to adversely affect computer programs, data, additional computers or a network. Zero-day exploits are particularly dangerous because they can be used before developers have the opportunity to develop and release patches."
        },
        {
            "question": "What is the difference between symmetric and asymmetric encryption?",
            "answer": "Symmetric encryption uses the same key for both encryption and decryption. It's fast but requires secure key exchange. Examples include AES and 3DES. Asymmetric encryption uses a public key for encryption and a private key for decryption. It's slower but solves the key distribution problem. Examples include RSA and ECC. Often, both methods are used together: asymmetric for secure key exchange and symmetric for efficient data encryption."
        },
        {
            "question": "What is a phishing attack?",
            "answer": "Phishing is a type of social engineering attack where attackers deceive people into revealing sensitive information or installing malware by disguising themselves as trustworthy entities in digital communication. Common forms include email phishing with fake websites, spear phishing targeting specific individuals, whaling targeting executives, vishing via phone calls, and smishing via text messages. Protection includes employee training, email filtering, multi-factor authentication, and keeping software updated."
        },
        {
            "question": "What is a buffer overflow?",
            "answer": "A buffer overflow occurs when a program writes data beyond the allocated memory buffer boundaries, potentially allowing attackers to execute arbitrary code or crash the system. It happens when input validation is inadequate. Types include stack-based (overwriting return addresses) and heap-based (corrupting dynamic memory). Prevention methods include input validation, bounds checking, using memory-safe languages, address space layout randomization (ASLR), and data execution prevention (DEP)."
        },
        {
            "question": "What is the principle of least privilege?",
            "answer": "The principle of least privilege is a computer security concept that limits user account and process permissions to only those absolutely required to perform authorized functions. This reduces the attack surface by ensuring that users have minimal access necessary to complete their job functions, limiting the damage that can occur from accidents, errors, or malicious attacks. Implementation involves role-based access control, regular permission reviews, and default-deny policies."
        },
        {
            "question": "What is the CIA triad in cybersecurity?",
            "answer": "The CIA triad refers to three core principles of information security: Confidentiality, Integrity, and Availability. Confidentiality ensures that information is accessible only to authorized parties. Integrity guarantees that data remains accurate and unaltered by unauthorized modifications. Availability ensures that information systems are functioning and accessible when needed by authorized users. These principles form the foundation for developing security policies and evaluating security measures within organizations."
        },
        {
            "question": "What is a man-in-the-middle attack?",
            "answer": "A man-in-the-middle (MITM) attack occurs when an attacker secretly intercepts and possibly alters communications between two parties who believe they are directly communicating with each other. The attacker can eavesdrop on sensitive information or manipulate data in transit. Common MITM techniques include ARP spoofing, DNS spoofing, HTTPS spoofing, and Wi-Fi eavesdropping. Protection measures include using encryption (HTTPS, VPN), certificate pinning, secure protocols, and public key infrastructure (PKI)."
        },
        {
            "question": "What is multi-factor authentication?",
            "answer": "Multi-factor authentication (MFA) is a security mechanism that requires users to provide two or more verification factors to gain access to a system or application. These factors fall into three categories: something you know (password, PIN), something you have (smartphone, security token), and something you are (fingerprint, facial recognition). By requiring multiple authentication factors, MFA significantly enhances security compared to password-only approaches, as attackers would need to compromise multiple authentication methods simultaneously."
        },
        {
            "question": "What is a security operations center (SOC)?",
            "answer": "A Security Operations Center (SOC) is a centralized unit that deals with security issues on an organizational and technical level. It employs people, processes, and technology to continuously monitor and improve an organization's security posture while preventing, detecting, analyzing, and responding to cybersecurity incidents. SOC responsibilities include 24/7 monitoring, alert investigation, incident response coordination, threat intelligence implementation, compliance reporting, and security improvement recommendations."
        }
    ]

def process_data(output_path=None, include_new_datasets=True):
    """
    Process data for Theta AI training.
    
    Args:
        output_path: Path to save the processed data. If None, a default path is used.
        
    Returns:
        Path to the processed data file.
    """
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Set default output path if none provided
    if output_path is None:
        output_path = datasets_dir / "processed_data.json"
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Generate conversation datasets if they don't exist yet
    conversation_flows_path = datasets_dir / "conversation_flows.json"
    multi_turn_dialogues_path = datasets_dir / "multi_turn_dialogues.json"
    
    if not conversation_flows_path.exists() or not multi_turn_dialogues_path.exists():
        logger.info("Creating conversation datasets...")
        conversation_paths = create_conversation_datasets()
        logger.info(f"Created conversation datasets at: {conversation_paths}")
    
    # Process Frostline data
    frostline_path = datasets_dir / "frostlinedata.json"
    frostline_qa = load_json_dataset(frostline_path) # Use the JSON loader instead
    
    # Load cybersecurity examples
    cybersecurity_qa = enhance_with_cybersecurity_examples()
    
    # Load additional JSON datasets
    software_dev_path = datasets_dir / "software_development.json"
    it_support_path = datasets_dir / "it_support.json"
    hardware_path = datasets_dir / "hardware_knowledge.json"
    network_security_path = datasets_dir / "network_security.json"
    programming_concepts_path = datasets_dir / "programming_concepts.json"
    cloud_computing_path = datasets_dir / "cloud_computing.json"
    advanced_cybersecurity_path = datasets_dir / "advanced_cybersecurity.json"
    advanced_programming_path = datasets_dir / "advanced_programming.json"
    advanced_cloud_path = datasets_dir / "advanced_cloud.json"
    conversational_examples_path = datasets_dir / "conversational_examples.json"
    theta_info_path = datasets_dir / "theta_info.json"
    technical_qa_path = datasets_dir / "technical_qa.json"
    general_conversation_path = datasets_dir / "general_conversation_fixed.json"
    advanced_technical_path = datasets_dir / "advanced_technical.json"
    
    # Load newly created datasets
    programming_part1_path = datasets_dir / "programming_part1.json"
    programming_part2_path = datasets_dir / "programming_part2.json"
    programming_part3_path = datasets_dir / "programming_part3.json"
    cybersecurity_part1_path = datasets_dir / "cybersecurity_part1.json"
    cybersecurity_part2_path = datasets_dir / "cybersecurity_part2.json"
    data_science_part1_path = datasets_dir / "data_science_part1.json"
    data_science_part2_path = datasets_dir / "data_science_part2.json"
    general_conversation_extension_path = datasets_dir / "general_conversation_extension.json"
    it_troubleshooting_part1_path = datasets_dir / "it_troubleshooting_part1.json"
    it_troubleshooting_part2_path = datasets_dir / "it_troubleshooting_part2.json"
    incident_response_path = datasets_dir / "incident_response_scenarios.json"
    user_conversations_path = datasets_dir / "user_conversations.json"
    
    # Load personal dataset - using the reformatted QA version
    shy_dakota_path = datasets_dir / "Shy&dakota_qa.json"
    
    # Look for OpenWebText processed files
    openwebtext_files = list(datasets_dir.glob("openwebtext_processed*.json"))
    logger.info(f"Found {len(openwebtext_files)} OpenWebText processed files")
    
    # Load our new enhanced datasets
    enhanced_networking_path = datasets_dir / "enhanced_networking.json"
    enhanced_cloud_computing_advanced_path = datasets_dir / "enhanced_cloud_computing_advanced.json"
    enhanced_data_science_path = datasets_dir / "enhanced_data_science.json"
    general_domain_challenges_path = datasets_dir / "general_domain_challenges.json"
    
    software_dev_qa = load_json_dataset(software_dev_path)
    it_support_qa = load_json_dataset(it_support_path)
    hardware_qa = load_json_dataset(hardware_path)
    network_security_qa = load_json_dataset(network_security_path)
    programming_concepts_qa = load_json_dataset(programming_concepts_path)
    cloud_computing_qa = load_json_dataset(cloud_computing_path)
    advanced_cybersecurity_qa = load_json_dataset(advanced_cybersecurity_path)
    advanced_programming_qa = load_json_dataset(advanced_programming_path)
    advanced_cloud_qa = load_json_dataset(advanced_cloud_path)
    conversational_examples_qa = load_json_dataset(conversational_examples_path)
    theta_info_qa = load_json_dataset(theta_info_path)
    technical_qa = load_json_dataset(technical_qa_path)
    general_conversation = load_json_dataset(general_conversation_path)
    advanced_technical = load_json_dataset(advanced_technical_path)
    
    # Load data from newly created datasets
    programming_part1 = load_json_dataset(programming_part1_path)
    programming_part2 = load_json_dataset(programming_part2_path)
    programming_part3 = load_json_dataset(programming_part3_path)
    cybersecurity_part1 = load_json_dataset(cybersecurity_part1_path)
    cybersecurity_part2 = load_json_dataset(cybersecurity_part2_path)
    data_science_part1 = load_json_dataset(data_science_part1_path)
    data_science_part2 = load_json_dataset(data_science_part2_path)
    general_conversation_extension = load_json_dataset(general_conversation_extension_path)
    it_troubleshooting_part1 = load_json_dataset(it_troubleshooting_part1_path)
    it_troubleshooting_part2 = load_json_dataset(it_troubleshooting_part2_path)
    incident_response = load_json_dataset(incident_response_path)
    user_conversations = load_json_dataset(user_conversations_path)
    
    # Load personal dataset
    shy_dakota = load_json_dataset(shy_dakota_path)
    
    # Load our new enhanced datasets
    enhanced_networking = load_json_dataset(enhanced_networking_path)
    enhanced_cloud_computing_advanced = load_json_dataset(enhanced_cloud_computing_advanced_path)
    enhanced_data_science = load_json_dataset(enhanced_data_science_path)
    general_domain_challenges = load_json_dataset(general_domain_challenges_path)
    
    # Load OpenWebText data (if available)
    openwebtext_data = []
    openwebtext_count = 0
    
    for owt_file in openwebtext_files:
        try:
            data = load_json_dataset(owt_file)
            openwebtext_data.extend(data)
            openwebtext_count += len(data)
            logger.info(f"Loaded {len(data)} examples from {owt_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {owt_file}: {e}")
            
    # Cap OpenWebText data if it's too large (optional)
    max_openwebtext_examples = 10000  # Adjust this value as needed
    if openwebtext_count > max_openwebtext_examples:
        logger.info(f"Capping OpenWebText data from {len(openwebtext_data)} to {max_openwebtext_examples} examples")
        openwebtext_data = openwebtext_data[:max_openwebtext_examples]

    # Load conversation datasets
    conversation_flows = load_json_dataset(conversation_flows_path)
    multi_turn = load_json_dataset(multi_turn_dialogues_path)
    
    # Load enhanced datasets if they exist
    expanded_conversations_path = datasets_dir / "expanded_conversations.json"
    enhanced_conversations_path = datasets_dir / "enhanced_conversations.json"
    expanded_technical_qa_path = datasets_dir / "expanded_technical_qa.json"
    tutorials_path = datasets_dir / "tutorials.json"
    problem_solutions_path = datasets_dir / "problem_solutions.json"
    
    # Load advanced datasets if they exist
    stack_exchange_path = datasets_dir / "stackexchange_processed.json"
    github_conversations_path = datasets_dir / "github_conversations_qa.json"
    technical_documentation_path = datasets_dir / "technical_documentation.json"
    tutorial_dialogues_path = datasets_dir / "tutorial_qa.json"
    
    # Load newly created datasets from the Theta AI improvement project
    # These files contain high-quality examples to enhance the model
    test_set_path = datasets_dir / "test_set.json"
    additional_cloud_computing_path = datasets_dir / "additional_cloud_computing.json"
    challenging_examples_path = datasets_dir / "challenging_examples.json"
    shy_dakota_qa_path = datasets_dir / "Shy&dakota_qa.json"  # Properly formatted QA version
    
    # Check and load enhanced datasets
    expanded_conversations = load_json_dataset(expanded_conversations_path) if expanded_conversations_path.exists() else []
    enhanced_conversations = load_json_dataset(enhanced_conversations_path) if enhanced_conversations_path.exists() else []
    expanded_technical_qa = load_json_dataset(expanded_technical_qa_path) if expanded_technical_qa_path.exists() else []
    tutorials = load_json_dataset(tutorials_path) if tutorials_path.exists() else []
    problem_solutions = load_json_dataset(problem_solutions_path) if problem_solutions_path.exists() else []
    
    # Check and load advanced datasets
    stack_exchange = load_json_dataset(stack_exchange_path) if stack_exchange_path.exists() else []
    github_conversations = load_json_dataset(github_conversations_path) if github_conversations_path.exists() else []
    technical_documentation = load_json_dataset(technical_documentation_path) if technical_documentation_path.exists() else []
    tutorial_dialogues = load_json_dataset(tutorial_dialogues_path) if tutorial_dialogues_path.exists() else []
    
    # Load newly created datasets
    if include_new_datasets:
        logger.info("Including newly created high-quality datasets")
        test_set = load_json_dataset(test_set_path) if test_set_path.exists() else []
        additional_cloud_computing = load_json_dataset(additional_cloud_computing_path) if additional_cloud_computing_path.exists() else []
        challenging_examples = load_json_dataset(challenging_examples_path) if challenging_examples_path.exists() else []
        shy_dakota = load_json_dataset(shy_dakota_qa_path) if shy_dakota_qa_path.exists() else []
        
        # Log counts of loaded items
        logger.info(f"Loaded {len(test_set)} test set examples")
        logger.info(f"Loaded {len(additional_cloud_computing)} additional cloud computing examples")
        logger.info(f"Loaded {len(challenging_examples)} challenging examples")
        logger.info(f"Loaded {len(shy_dakota)} Dakota & Shyanne QA examples")
    else:
        test_set = []
        additional_cloud_computing = []
        challenging_examples = []
        shy_dakota = load_json_dataset(shy_dakota_path) if shy_dakota_path.exists() else []
    
    # Combine all datasets
    all_qa_pairs = frostline_qa + cybersecurity_qa + software_dev_qa + it_support_qa + hardware_qa + \
                  network_security_qa + programming_concepts_qa + cloud_computing_qa + \
                  advanced_cybersecurity_qa + advanced_programming_qa + advanced_cloud_qa + \
                  conversational_examples_qa + theta_info_qa + technical_qa + \
                  general_conversation + advanced_technical + \
                  programming_part1 + programming_part2 + programming_part3 + \
                  cybersecurity_part1 + cybersecurity_part2 + \
                  data_science_part1 + data_science_part2 + \
                  general_conversation_extension + \
                  it_troubleshooting_part1 + it_troubleshooting_part2 + \
                  incident_response + user_conversations + conversation_flows + \
                  expanded_conversations + enhanced_conversations + expanded_technical_qa + \
                  tutorials + problem_solutions + \
                  stack_exchange + github_conversations + technical_documentation + tutorial_dialogues + \
                  openwebtext_data + \
                  shy_dakota + \
                  test_set + additional_cloud_computing + challenging_examples + \
                  enhanced_networking + enhanced_cloud_computing_advanced + enhanced_data_science + general_domain_challenges  # Include newly created datasets
    
    # Save processed data
    with open(output_path, 'w') as file:
        json.dump(all_qa_pairs, file, indent=2)
    print(f"Data saved to {output_path}")
    print(f"Processed {len(all_qa_pairs)} QA pairs:")
    print(f"- {len(frostline_qa)} from Frostline data")
    print(f"- {len(cybersecurity_qa)} cybersecurity examples")
    print(f"- {len(software_dev_qa)} software development examples")
    print(f"- {len(it_support_qa)} IT support examples")
    print(f"- {len(hardware_qa)} hardware knowledge examples")
    print(f"- {len(network_security_qa)} network security examples")
    print(f"- {len(programming_concepts_qa)} programming concepts examples")
    print(f"- {len(cloud_computing_qa)} cloud computing examples")
    print(f"- {len(advanced_cybersecurity_qa)} advanced cybersecurity examples")
    print(f"- {len(advanced_programming_qa)} advanced programming examples")
    print(f"- {len(advanced_cloud_qa)} advanced cloud examples")
    print(f"- {len(conversational_examples_qa)} conversational examples")
    print(f"- {len(theta_info_qa)} Theta AI info examples")
    print(f"- {len(technical_qa)} Technical QA examples")
    print(f"- {len(general_conversation)} General conversation examples")
    print(f"- {len(advanced_technical)} Advanced technical examples")
    print(f"- {len(programming_part1)} Programming part 1 examples")
    print(f"- {len(programming_part2)} Programming part 2 examples")
    print(f"- {len(programming_part3)} Programming part 3 examples")
    print(f"- {len(cybersecurity_part1)} Cybersecurity part 1 examples")
    print(f"- {len(cybersecurity_part2)} Cybersecurity part 2 examples")
    print(f"- {len(data_science_part1)} Data science part 1 examples")
    print(f"- {len(data_science_part2)} Data science part 2 examples")
    print(f"- {len(general_conversation_extension)} General conversation extension examples")
    print(f"- {len(it_troubleshooting_part1)} IT troubleshooting part 1 examples")
    print(f"- {len(it_troubleshooting_part2)} IT troubleshooting part 2 examples")
    print(f"- {len(incident_response)} Incident response scenarios examples")
    print(f"- {len(user_conversations)} User conversations examples")
    print(f"- {len(conversation_flows)} Conversation flows examples")
    print(f"- {len(expanded_conversations)} Expanded conversations examples")
    print(f"- {len(enhanced_conversations)} Enhanced multi-turn conversations examples")
    print(f"- {len(expanded_technical_qa)} Technical Q&A examples")
    print(f"- {len(tutorials)} Tutorial examples")
    print(f"- {len(problem_solutions)} Problem-solution examples")
    print(f"- {len(openwebtext_data)} OpenWebText examples")
    # Advanced datasets
    print(f"- {len(stack_exchange)} Stack Exchange examples")
    print(f"- {len(github_conversations)} GitHub conversations examples")
    print(f"- {len(technical_documentation)} Technical documentation examples")
    print(f"- {len(tutorial_dialogues)} Tutorial dialogue examples")
    
    # Personal dataset
    print(f"- {len(shy_dakota)} Dakota & Shyanne personal data examples")
    
    # Newly created datasets
    print(f"- {len(test_set)} Test set examples")
    print(f"- {len(additional_cloud_computing)} Additional cloud computing examples")
    print(f"- {len(challenging_examples)} Challenging examples across domains")
    
    # Print our new enhanced datasets
    print(f"- {len(enhanced_networking)} Enhanced networking examples")
    print(f"- {len(enhanced_cloud_computing_advanced)} Enhanced cloud computing advanced examples")
    print(f"- {len(enhanced_data_science)} Enhanced data science examples")
    print(f"- {len(general_domain_challenges)} General domain challenge examples")
    
    return output_path

def main():
    """Main function to process data."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process data for Theta AI training.')
    parser.add_argument('output_path', nargs='?', default=None, help='Path to save the processed data')
    parser.add_argument('--include_new_datasets', action='store_true', default=True, 
                        help='Include newly created datasets (test_set, additional_cloud_computing, etc.)')
    args = parser.parse_args()
    
    # Process data with command line arguments
    process_data(output_path=args.output_path, include_new_datasets=args.include_new_datasets)

if __name__ == "__main__":
    main()
