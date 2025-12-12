#data_processor.py

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import specialized data loader for non-Q&A datasets
SPECIALIZED_LOADER_AVAILABLE = False
LoadedDataset = None

try:
    # Try relative import first (when running as part of package)
    from src.data_processing.specialized_data_loader import (
        load_specialized_dataset,
        load_personality_datasets,
        extract_training_samples,
        DatasetType,
        LoadedDataset
    )
    SPECIALIZED_LOADER_AVAILABLE = True
except ImportError:
    try:
        # Try direct import (when running from src directory)
        from data_processing.specialized_data_loader import (
            load_specialized_dataset,
            load_personality_datasets,
            extract_training_samples,
            DatasetType,
            LoadedDataset
        )
        SPECIALIZED_LOADER_AVAILABLE = True
    except ImportError:
        try:
            # Try absolute path import
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from data_processing.specialized_data_loader import (
                load_specialized_dataset,
                load_personality_datasets,
                extract_training_samples,
                DatasetType,
                LoadedDataset
            )
            SPECIALIZED_LOADER_AVAILABLE = True
        except ImportError:
            print("Warning: Specialized data loader not available, using basic loading")

def load_frostline_data(data_path):
    """
    Load the Frostline dataset from a text file and structure it.
    """
    with open(data_path, 'r') as file:
        content = file.read()
    
    # Split content by empty lines to get individual entries
    entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
    
    # Transform into a structured format for QA pairs
    qa_pairs = []
    
    # Add basic company information
    for entry in entries:
        if "Frostline Solutions offers" in entry:
            qa_pairs.append({
                "question": "What is Frostline?",
                "answer": entry.strip()
            })
            qa_pairs.append({
                "question": "What services does Frostline offer?",
                "answer": entry.strip()
            })
        
        elif "takes pride in its local only" in entry:
            qa_pairs.append({
                "question": "What infrastructure does Frostline use?",
                "answer": entry.strip()
            })
        
        elif "CEO of frostline solutions" in entry:
            qa_pairs.append({
                "question": "Who is the CEO of Frostline?",
                "answer": entry.strip()
            })
            
        elif "Co CEO of frostline solutions" in entry:
            qa_pairs.append({
                "question": "Who is the Co-CEO of Frostline?",
                "answer": entry.strip()
            })
            
        elif "hq is located" in entry:
            qa_pairs.append({
                "question": "Where is Frostline headquartered?",
                "answer": entry.strip()
            })
            qa_pairs.append({
                "question": "Where is Frostline located?",
                "answer": entry.strip()
            })
    
    return qa_pairs

def enhance_with_cybersecurity_examples():
    """
    Add cybersecurity concept examples to complement the Frostline data.
    """
    cybersecurity_examples = [
        {
            "question": "What is defense in depth?",
            "answer": "Defense in depth is a cybersecurity strategy that employs multiple layers of security controls throughout an IT system. By implementing redundant protective mechanisms, if one layer fails, others will still provide protection. This approach includes technical controls, physical security measures, and administrative policies working together to comprehensively protect organizational assets."
        },
        {
            "question": "What is zero trust architecture?",
            "answer": "Zero Trust Architecture is a security framework that assumes no user or device should be automatically trusted, whether inside or outside the network perimeter. It follows the principle of 'never trust, always verify' by requiring strict identity verification and continuous validation for every device, user, and application trying to access resources, regardless of their location."
        },
        {
            "question": "What is a penetration test?",
            "answer": "A penetration test (or pen test) is a simulated cyber attack against a computer system, network, or application to identify vulnerabilities that could be exploited. It's performed by ethical hackers using the same tools and techniques as malicious actors but in a controlled environment to help organizations improve their security posture."
        },
        {
            "question": "What is a vulnerability assessment?",
            "answer": "A vulnerability assessment is a systematic review of security weaknesses in an information system. It evaluates if the system is susceptible to any known vulnerabilities, assigns severity levels to those vulnerabilities, and recommends remediation or mitigation measures where needed."
        },
        {
            "question": "What is a SIEM solution?",
            "answer": "SIEM (Security Information and Event Management) is a solution that provides real-time analysis of security alerts generated by applications and network hardware. It combines security information management (SIM) and security event management (SEM) to provide a comprehensive view of an organization's information security."
        },
        {
            "question": "What is ransomware?",
            "answer": "Ransomware is a type of malicious software designed to block access to a computer system or data until a sum of money (ransom) is paid. It typically encrypts files on the victim's system, making them inaccessible, and demands payment to decrypt them."
        },
        {
            "question": "What is phishing?",
            "answer": "Phishing is a cybercrime where attackers disguise themselves as trustworthy entities in electronic communications to obtain sensitive information such as usernames, passwords, and credit card details. It typically occurs via email, messaging, or malicious websites and often directs users to enter personal information at a fake website that matches the look and feel of the legitimate site."
        },
        {
            "question": "What is multi-factor authentication?",
            "answer": "Multi-factor authentication (MFA) is a security system that requires more than one method of authentication from independent categories of credentials to verify the user's identity for a login or other transaction. It combines two or more independent credentials: what the user knows (password), what the user has (security token), and what the user is (biometric verification)."
        },
        {
            "question": "What is a firewall?",
            "answer": "A firewall is a network security device that monitors and filters incoming and outgoing network traffic based on an organization's previously established security policies. It acts as a barrier between a trusted internal network and untrusted external networks such as the Internet, establishing rules for what traffic should be allowed or blocked."
        },
        {
            "question": "What is an IDS/IPS?",
            "answer": "IDS (Intrusion Detection System) and IPS (Intrusion Prevention System) are security technologies that monitor network traffic for suspicious activity. An IDS detects and alerts about potential intrusions, while an IPS goes a step further by actively preventing or blocking detected intrusions. They help identify attacks that might bypass the firewall."
        }
    ]
    return cybersecurity_examples

def load_json_dataset(file_path, preserve_structure: bool = False):
    """
    Load a dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        preserve_structure: If True, don't try to convert to Q&A format
        
    Returns:
        Loaded data (structure depends on preserve_structure flag)
    """
    try:
        # Try UTF-8 first, fall back to latin1
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as file:
                data = json.load(file)
        
        # If preserve_structure is True, return as-is
        if preserve_structure:
            return data
        
        # Check if we need to convert the structure
        if isinstance(data, dict) and not (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'question' in data[0]):
            # Try to import the adapter
            try:
                # Dynamically import the adapter to avoid circular imports
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "structured_data_adapter", 
                    Path(__file__).parent / "data_processing" / "structured_data_adapter.py"
                )
                adapter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(adapter)
                
                # Convert the data
                converted_data = adapter.nested_to_qa(data)
                if converted_data:
                    print(f"Converted nested structure from {file_path} to {len(converted_data)} QA pairs")
                    return converted_data
            except Exception as e:
                print(f"Warning: Could not convert nested structure: {e}")
                
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def load_openmath_sampled(file_path, max_samples: int = 10000):
    """
    Load a sampled subset from the massive OpenMathInstruct-1 dataset.
    The full dataset has 7.3M entries which is too large to load at once.
    
    Args:
        file_path: Path to the JSON file
        max_samples: Maximum number of samples to load (default 10000)
        
    Returns:
        List of sampled problem-solution pairs
    """
    import random
    
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"OpenMathInstruct file not found: {file_path}")
            return []
        
        print(f"Loading sampled subset from OpenMathInstruct-1 (max {max_samples} samples)...")
        
        # For very large JSON files, we need to stream-parse
        # First, check file size
        file_size = file_path.stat().st_size
        print(f"File size: {file_size / (1024*1024):.1f} MB")
        
        if file_size > 100 * 1024 * 1024:  # > 100MB
            # Use ijson for streaming if available, otherwise sample from beginning
            try:
                import ijson
                entries = []
                with open(file_path, 'rb') as f:
                    # Stream parse entries
                    parser = ijson.items(f, 'entries.item')
                    for i, entry in enumerate(parser):
                        if i < max_samples * 3:  # Collect 3x samples for random selection
                            entries.append(entry)
                        else:
                            # Reservoir sampling for remaining items
                            j = random.randint(0, i)
                            if j < len(entries):
                                entries[j] = entry
                        if i % 100000 == 0 and i > 0:
                            print(f"  Processed {i:,} entries...")
                
                # Sample from collected entries
                if len(entries) > max_samples:
                    entries = random.sample(entries, max_samples)
                    
                print(f"Loaded {len(entries)} sampled math examples from OpenMathInstruct-1")
                return entries
                
            except ImportError:
                print("ijson not available, using chunked loading...")
                # Fall back to loading first N entries only
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read until we find entries array, then parse incrementally
                    content = ""
                    entries = []
                    in_entries = False
                    brace_count = 0
                    current_entry = ""
                    
                    for line in f:
                        if '"entries":' in line:
                            in_entries = True
                            continue
                        
                        if in_entries:
                            for char in line:
                                if char == '{':
                                    brace_count += 1
                                    current_entry += char
                                elif char == '}':
                                    brace_count -= 1
                                    current_entry += char
                                    if brace_count == 0 and current_entry.strip():
                                        try:
                                            entry = json.loads(current_entry)
                                            entries.append(entry)
                                            current_entry = ""
                                            if len(entries) >= max_samples:
                                                print(f"Loaded {len(entries)} math examples from OpenMathInstruct-1")
                                                return entries
                                        except:
                                            current_entry = ""
                                elif brace_count > 0:
                                    current_entry += char
                    
                    print(f"Loaded {len(entries)} math examples from OpenMathInstruct-1")
                    return entries
        else:
            # Small enough to load directly
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "entries" in data:
                entries = data["entries"]
                if len(entries) > max_samples:
                    entries = random.sample(entries, max_samples)
                print(f"Loaded {len(entries)} math examples from OpenMathInstruct-1")
                return entries
            elif isinstance(data, list):
                if len(data) > max_samples:
                    data = random.sample(data, max_samples)
                print(f"Loaded {len(data)} math examples from OpenMathInstruct-1")
                return data
            
            return []
            
    except Exception as e:
        print(f"Error loading OpenMathInstruct-1: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_specialized_json(file_path, domain: str = None):
    """
    Load a JSON dataset using the specialized loader that preserves structure.
    
    Args:
        file_path: Path to the JSON file
        domain: Optional domain name
        
    Returns:
        LoadedDataset object or None
    """
    if SPECIALIZED_LOADER_AVAILABLE:
        return load_specialized_dataset(file_path, domain=domain, preserve_structure=True)
    else:
        # Fallback: basic loading
        data = load_json_dataset(file_path, preserve_structure=True)
        if data:
            return {
                "name": Path(file_path).name,
                "content": data,
                "domain": domain or "general"
            }
        return None

def process_data():
    """
    Process all available datasets and create a comprehensive training dataset.
    """
    # Create paths
    project_dir = Path("G:/Theta AI")
    datasets_dir = project_dir / "Datasets"
    output_path = datasets_dir / "processed_data.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(datasets_dir, exist_ok=True)
    
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
    shy_dakota_path = datasets_dir / "Shy&dakota_qa.json"  # Dakota & Shyanne dataset
    
    # Load new language and comprehension datasets
    javascript_dom_path = datasets_dir / "javascript_dom_interaction.json"
    typescript_migration_path = datasets_dir / "typescript_javascript_migration.json"
    web_framework_path = datasets_dir / "web_framework_integration.json"
    reading_comprehension_path = datasets_dir / "reading_comprehension_examples.json"
    grammar_rules_path = datasets_dir / "grammar_rules_simple.json"
    contextual_conversation_path = datasets_dir / "contextual_conversation_examples.json"
    advanced_english_path = datasets_dir / "advanced_english_usage.json"
    human_like_dpo_path = datasets_dir / "human_like_dpo_dataset.json"  # Human-Like DPO Dataset (mixed curriculum)
    
    # NVIDIA OpenMathInstruct-1 dataset (mathematics instruction data)
    openmath_instruct_path = datasets_dir / "openmath_instruct_1.json"
    
    # OpenAssistant OASST1 dataset (high-quality assistant conversations)
    openassistant_oasst1_path = datasets_dir / "openassistant_oasst1.json"
    openassistant_oasst1_enhanced_path = datasets_dir / "openassistant_oasst1_enhanced.json"
    
    # Combat assistance datasets
    combat_scenarios_path = datasets_dir / "combat_scenarios_qa.json"
    
    # Epsilon-related datasets (Red vs Blue)
    epsilon_personality_path = datasets_dir / "epsilon_personality.json"
    ai_fragments_path = datasets_dir / "ai_fragments_capabilities.json"
    virtual_environment_path = datasets_dir / "virtual_environment_operations.json"
    
    # Theta Alpha and fragments datasets
    theta_identity_path = datasets_dir / "theta_identity.json"
    enhanced_identity_path = datasets_dir / "enhanced_identity.json"  # New enhanced identity dataset
    fragments_under_theta_path = datasets_dir / "fragments_under_theta.json"
    delta_fragment_path = datasets_dir / "delta_fragment.json"
    sigma_fragment_path = datasets_dir / "sigma_fragment.json"
    omega_fragment_path = datasets_dir / "omega_fragment.json"
    gamma_fragment_path = datasets_dir / "gamma_fragment.json"
    eta_fragment_path = datasets_dir / "eta_fragment.json"
    iota_fragment_path = datasets_dir / "iota_fragment.json"
    beta_fragment_path = datasets_dir / "beta_fragment.json"
    lambda_fragment_path = datasets_dir / "lambda_fragment.json"
    kappa_fragment_path = datasets_dir / "kappa_fragment.json"
    
    # Advanced technical datasets
    cybersecurity_advanced_path = datasets_dir / "cybersecurity_advanced.json"
    quantum_computing_basics_path = datasets_dir / "quantum_computing_basics_fixed.json"  # Using fixed version
    machine_learning_frameworks_path = datasets_dir / "machine_learning_frameworks.json"
    cloud_computing_architecture_path = datasets_dir / "cloud_computing_architecture.json"
    devops_practices_path = datasets_dir / "devops_practices.json"
    
    # Conversational datasets
    basic_conversation_path = datasets_dir / "basic_conversation.json"
    common_questions_path = datasets_dir / "common_questions.json"
    small_talk_path = datasets_dir / "small_talk.json"
    
    # Personality and opinion datasets (specialized - not Q&A format)
    theta_opinions_path = datasets_dir / "theta_opinions.json"
    basic_conversation_natural_path = datasets_dir / "basic_conversation_natural.json"
    small_talk_natural_path = datasets_dir / "small_talk_natural.json"
    
    # Emotional intelligence datasets
    emotional_responses_path = datasets_dir / "emotional_responses.json"
    motivational_dialogues_path = datasets_dir / "motivational_dialogues.json"
    conflict_resolution_path = datasets_dir / "conflict_resolution.json"
    
    # Narrative datasets
    theta_personal_stories_path = datasets_dir / "theta_personal_stories.json"
    fragment_backstories_path = datasets_dir / "fragment_backstories.json"
    
    # Domain-specific Theta integration datasets
    combat_theta_path = datasets_dir / "combat_theta_integration.json"
    technical_theta_path = datasets_dir / "technical_theta_integration.json"
    personal_theta_path = datasets_dir / "personal_theta_interaction.json"
    
    # Technical patterns datasets (specialized - non-Q&A format)
    code_review_patterns_path = datasets_dir / "code_review_patterns.json"
    security_vulnerabilities_path = datasets_dir / "security_vulnerabilities.json"
    architecture_patterns_path = datasets_dir / "architecture_patterns.json"
    debugging_scenarios_path = datasets_dir / "debugging_scenarios.json"
    api_design_patterns_path = datasets_dir / "api_design_patterns.json"
    error_handling_patterns_path = datasets_dir / "error_handling_patterns.json"
    
    software_dev_qa = load_json_dataset(software_dev_path)
    it_support_qa = load_json_dataset(it_support_path)
    hardware_qa = load_json_dataset(hardware_path)
    network_security_qa = load_json_dataset(network_security_path)
    programming_concepts_qa = load_json_dataset(programming_concepts_path)
    cloud_computing_qa = load_json_dataset(cloud_computing_path)
    advanced_cybersecurity_qa = load_json_dataset(advanced_cybersecurity_path)
    advanced_programming_qa = load_json_dataset(advanced_programming_path)
    advanced_cloud_qa = load_json_dataset(advanced_cloud_path)
    shy_dakota_qa = load_json_dataset(shy_dakota_path)  # Load Dakota & Shyanne dataset
    
    # Load the new datasets
    javascript_dom_qa = load_json_dataset(javascript_dom_path)
    typescript_migration_qa = load_json_dataset(typescript_migration_path)
    web_framework_qa = load_json_dataset(web_framework_path)
    reading_comprehension_qa = load_json_dataset(reading_comprehension_path)
    grammar_rules_qa = load_json_dataset(grammar_rules_path)
    contextual_conversation_qa = load_json_dataset(contextual_conversation_path)
    advanced_english_qa = load_json_dataset(advanced_english_path)
    human_like_dpo_data = load_json_dataset(human_like_dpo_path)  # Loaded in original format, not converted to QA
    
    # Load OpenMathInstruct-1 dataset (kept in original problem-solution format)
    # This dataset is massive (7.3M entries), so we load a sampled subset
    openmath_instruct_data = load_openmath_sampled(openmath_instruct_path, max_samples=10000)
    
    # Load OpenAssistant OASST1 dataset (high-quality assistant conversations)
    # Prefer enhanced version if available, fallback to standard
    if openassistant_oasst1_enhanced_path.exists():
        openassistant_oasst1_qa = load_json_dataset(openassistant_oasst1_enhanced_path)
        print(f"Loaded {len(openassistant_oasst1_qa)} OpenAssistant OASST1 (enhanced) examples")
    elif openassistant_oasst1_path.exists():
        openassistant_oasst1_qa = load_json_dataset(openassistant_oasst1_path)
        print(f"Loaded {len(openassistant_oasst1_qa)} OpenAssistant OASST1 examples")
    else:
        openassistant_oasst1_qa = []
        print("OpenAssistant OASST1 dataset not found. Run download_openassistant_oasst1.py to download.")
    
    # Load combat scenarios dataset
    combat_scenarios_qa = load_json_dataset(combat_scenarios_path)
    
    # Load Epsilon-related datasets
    epsilon_personality_qa = load_json_dataset(epsilon_personality_path)
    ai_fragments_qa = load_json_dataset(ai_fragments_path)
    virtual_environment_qa = load_json_dataset(virtual_environment_path)
    
    # Load Theta Alpha and fragments datasets
    theta_identity_qa = load_json_dataset(theta_identity_path)
    enhanced_identity_qa = load_json_dataset(enhanced_identity_path)  # Load enhanced identity dataset
    fragments_under_theta_qa = load_json_dataset(fragments_under_theta_path)
    delta_fragment_qa = load_json_dataset(delta_fragment_path)
    sigma_fragment_qa = load_json_dataset(sigma_fragment_path)
    omega_fragment_qa = load_json_dataset(omega_fragment_path)
    gamma_fragment_qa = load_json_dataset(gamma_fragment_path)
    eta_fragment_qa = load_json_dataset(eta_fragment_path)
    iota_fragment_qa = load_json_dataset(iota_fragment_path)
    beta_fragment_qa = load_json_dataset(beta_fragment_path)
    lambda_fragment_qa = load_json_dataset(lambda_fragment_path)
    kappa_fragment_qa = load_json_dataset(kappa_fragment_path)
    
    # Load advanced technical datasets
    cybersecurity_advanced_qa = load_json_dataset(cybersecurity_advanced_path)
    quantum_computing_basics_qa = load_json_dataset(quantum_computing_basics_path)
    machine_learning_frameworks_qa = load_json_dataset(machine_learning_frameworks_path)
    cloud_computing_architecture_qa = load_json_dataset(cloud_computing_architecture_path)
    devops_practices_qa = load_json_dataset(devops_practices_path)
    
    # Load conversational datasets
    basic_conversation_qa = load_json_dataset(basic_conversation_path)
    common_questions_qa = load_json_dataset(common_questions_path)
    small_talk_qa = load_json_dataset(small_talk_path)
    
    # Load personality and opinion datasets (preserve structure - not Q&A)
    theta_opinions_data = load_json_dataset(theta_opinions_path, preserve_structure=True)
    basic_conversation_natural_data = load_json_dataset(basic_conversation_natural_path, preserve_structure=True)
    small_talk_natural_data = load_json_dataset(small_talk_natural_path, preserve_structure=True)
    
    # Load technical patterns datasets (preserve structure - specialized non-Q&A)
    code_review_patterns_data = load_json_dataset(code_review_patterns_path, preserve_structure=True)
    security_vulnerabilities_data = load_json_dataset(security_vulnerabilities_path, preserve_structure=True)
    architecture_patterns_data = load_json_dataset(architecture_patterns_path, preserve_structure=True)
    debugging_scenarios_data = load_json_dataset(debugging_scenarios_path, preserve_structure=True)
    api_design_patterns_data = load_json_dataset(api_design_patterns_path, preserve_structure=True)
    error_handling_patterns_data = load_json_dataset(error_handling_patterns_path, preserve_structure=True)
    
    # Load emotional intelligence datasets
    emotional_responses_qa = load_json_dataset(emotional_responses_path)
    motivational_dialogues_qa = load_json_dataset(motivational_dialogues_path)
    conflict_resolution_qa = load_json_dataset(conflict_resolution_path)
    
    # Load narrative datasets
    theta_personal_stories_qa = load_json_dataset(theta_personal_stories_path)
    fragment_backstories_qa = load_json_dataset(fragment_backstories_path)
    
    # Load domain-specific Theta integration datasets
    combat_theta_qa = load_json_dataset(combat_theta_path)
    technical_theta_qa = load_json_dataset(technical_theta_path)
    personal_theta_qa = load_json_dataset(personal_theta_path)
    
    # Load new conversational improvement datasets
    identity_consistency_path = datasets_dir / "identity_consistency.json"
    conversational_coherence_path = datasets_dir / "conversational_coherence.json"
    code_block_usage_path = datasets_dir / "code_block_usage.json"
    factual_accuracy_path = datasets_dir / "factual_accuracy.json"
    context_retention_path = datasets_dir / "context_retention.json"
    conversation_multiturn_path = datasets_dir / "conversation_multiturn.json"
    improved_formatting_path = datasets_dir / "improved_formatting.json"
    technical_accuracy_path = datasets_dir / "technical_accuracy.json"
    
    # Load web search and real-time data integration datasets
    weather_api_queries_path = datasets_dir / "weather_api_queries.json"
    time_zone_queries_path = datasets_dir / "time_zone_queries.json"
    current_events_handling_path = datasets_dir / "current_events_handling.json"
    web_search_integration_path = datasets_dir / "web_search_integration.json"
    realtime_data_integration_path = datasets_dir / "realtime_data_integration.json"
    technical_documentation_search_path = datasets_dir / "technical_documentation_search.json"
    language_models_rag_path = datasets_dir / "language_models_rag.json"
    
    # Define diverse curriculum dataset paths
    emotional_learning_path = datasets_dir / "Emotional_learning.json"
    personal_preferences_path = datasets_dir / "Personal_preferences.json"
    ethical_scenarios_path = datasets_dir / "Ethical_scenarios.json"
    technical_concepts_path = datasets_dir / "Technical_concepts.json"
    narrative_experiences_path = datasets_dir / "Narrative_experiences.json"
    emotional_intelligence_path = datasets_dir / "Emotional_intelligence.json"
    
    # Define additional diverse curriculum dataset paths
    cognitive_reasoning_path = datasets_dir / "Cognitive_reasoning.json"
    psychological_frameworks_path = datasets_dir / "Psychological_frameworks.json"
    conversational_dynamics_path = datasets_dir / "Conversational_dynamics.json"
    human_experience_simulation_path = datasets_dir / "Human_experience_simulation.json"
    humor_comprehension_path = datasets_dir / "Humor_comprehension.json"
    cultural_contexts_path = datasets_dir / "Cultural_contexts.json"
    ethical_reasoning_path = datasets_dir / "Ethical_reasoning.json"
    tactical_knowledge_path = datasets_dir / "Tactical_knowledge.json"
    interpersonal_intelligence_path = datasets_dir / "Interpersonal_intelligence.json"
    memory_simulation_path = datasets_dir / "Memory_simulation.json"
    
    # Enhanced training datasets (10 recommendations implementation)
    enhanced_training_dir = datasets_dir / "enhanced_training"
    contrastive_personality_path = enhanced_training_dir / "contrastive_personality.json"
    fragment_specific_path = enhanced_training_dir / "fragment_specific_responses.json"
    multi_turn_path = enhanced_training_dir / "multi_turn_conversations.json"
    trust_progression_path = enhanced_training_dir / "trust_progression.json"
    uncertainty_handling_path = enhanced_training_dir / "uncertainty_handling.json"
    graceful_refusals_path = enhanced_training_dir / "graceful_refusals.json"
    proactive_suggestions_path = enhanced_training_dir / "proactive_suggestions.json"
    mood_variations_path = enhanced_training_dir / "mood_variations.json"
    long_form_technical_path = enhanced_training_dir / "long_form_technical.json"
    real_code_reviews_path = enhanced_training_dir / "real_code_reviews.json"
    
    # Load the diverse curriculum datasets (non-Q&A format)
    emotional_learning_data = load_json_dataset(emotional_learning_path)
    personal_preferences_data = load_json_dataset(personal_preferences_path)
    ethical_scenarios_data = load_json_dataset(ethical_scenarios_path)
    technical_concepts_data = load_json_dataset(technical_concepts_path)
    narrative_experiences_data = load_json_dataset(narrative_experiences_path)
    emotional_intelligence_data = load_json_dataset(emotional_intelligence_path)
    
    # Load additional diverse curriculum datasets
    cognitive_reasoning_data = load_json_dataset(cognitive_reasoning_path)
    psychological_frameworks_data = load_json_dataset(psychological_frameworks_path)
    conversational_dynamics_data = load_json_dataset(conversational_dynamics_path)
    human_experience_simulation_data = load_json_dataset(human_experience_simulation_path)
    humor_comprehension_data = load_json_dataset(humor_comprehension_path)
    cultural_contexts_data = load_json_dataset(cultural_contexts_path)
    ethical_reasoning_data = load_json_dataset(ethical_reasoning_path)
    tactical_knowledge_data = load_json_dataset(tactical_knowledge_path)
    interpersonal_intelligence_data = load_json_dataset(interpersonal_intelligence_path)
    memory_simulation_data = load_json_dataset(memory_simulation_path)

    # Load the new conversational improvement datasets
    identity_consistency_qa = load_json_dataset(identity_consistency_path)
    conversational_coherence_qa = load_json_dataset(conversational_coherence_path)
    code_block_usage_qa = load_json_dataset(code_block_usage_path)
    factual_accuracy_qa = load_json_dataset(factual_accuracy_path)
    context_retention_qa = load_json_dataset(context_retention_path)
    conversation_multiturn_qa = load_json_dataset(conversation_multiturn_path)
    improved_formatting_qa = load_json_dataset(improved_formatting_path)
    technical_accuracy_qa = load_json_dataset(technical_accuracy_path)

    # Load the web search and real-time data integration datasets
    weather_api_queries_qa = load_json_dataset(weather_api_queries_path)
    time_zone_queries_qa = load_json_dataset(time_zone_queries_path)
    current_events_handling_qa = load_json_dataset(current_events_handling_path)
    web_search_integration_qa = load_json_dataset(web_search_integration_path)
    realtime_data_integration_qa = load_json_dataset(realtime_data_integration_path)
    technical_documentation_search_qa = load_json_dataset(technical_documentation_search_path)
    language_models_rag_qa = load_json_dataset(language_models_rag_path)
    
    # Load enhanced training datasets (10 recommendations)
    enhanced_training_qa = []
    
    # Helper function to convert enhanced datasets to QA format
    def convert_enhanced_to_qa(data, source_type):
        """Convert enhanced training data formats to QA pairs."""
        qa_pairs = []
        if not data or not isinstance(data, list):
            return qa_pairs
            
        for item in data:
            if source_type == "contrastive":
                # DPO-style: use chosen response as answer
                if 'question' in item and 'chosen' in item:
                    qa_pairs.append({
                        'question': item['question'],
                        'answer': item['chosen'],
                        'domain': item.get('domain', 'personality'),
                        'type': 'contrastive_chosen'
                    })
            elif source_type == "fragment_specific":
                if 'question' in item and 'answer' in item:
                    qa_pairs.append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'domain': item.get('domain', 'technical'),
                        'active_fragment': item.get('active_fragment', 'theta')
                    })
            elif source_type == "multi_turn":
                # Convert multi-turn to individual QA pairs with context
                if 'turns' in item:
                    context = ""
                    for i, turn in enumerate(item['turns']):
                        if turn['role'] == 'user':
                            # Find the next theta response
                            if i + 1 < len(item['turns']) and item['turns'][i + 1]['role'] == 'theta':
                                question = context + turn['content'] if context else turn['content']
                                answer = item['turns'][i + 1]['content']
                                qa_pairs.append({
                                    'question': question,
                                    'answer': answer,
                                    'domain': item.get('domain', 'conversation'),
                                    'conversation_id': item.get('conversation_id', 'unknown'),
                                    'type': 'multi_turn'
                                })
                                # Build context for next turn
                                context = f"[Previous: {turn['content'][:100]}...] "
            elif source_type == "trust_progression":
                if 'question' in item and 'answer' in item:
                    qa_pairs.append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'domain': 'trust_progression',
                        'trust_level': item.get('trust_level', 'familiar')
                    })
            elif source_type == "mood_variations":
                # Expand mood variations into multiple QA pairs
                if 'question' in item and 'variations' in item:
                    for var in item['variations']:
                        qa_pairs.append({
                            'question': item['question'],
                            'answer': var['answer'],
                            'domain': 'mood_variations',
                            'mood': var.get('mood', 'calm')
                        })
            else:
                # Standard QA format
                if 'question' in item and 'answer' in item:
                    qa_pairs.append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'domain': item.get('domain', 'enhanced_training')
                    })
        return qa_pairs
    
    # Load and convert each enhanced dataset
    if enhanced_training_dir.exists():
        print("Loading enhanced training datasets...")
        
        contrastive_data = load_json_dataset(contrastive_personality_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(contrastive_data, "contrastive"))
        
        fragment_data = load_json_dataset(fragment_specific_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(fragment_data, "fragment_specific"))
        
        multi_turn_data = load_json_dataset(multi_turn_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(multi_turn_data, "multi_turn"))
        
        trust_data = load_json_dataset(trust_progression_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(trust_data, "trust_progression"))
        
        uncertainty_data = load_json_dataset(uncertainty_handling_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(uncertainty_data, "standard"))
        
        refusal_data = load_json_dataset(graceful_refusals_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(refusal_data, "standard"))
        
        proactive_data = load_json_dataset(proactive_suggestions_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(proactive_data, "standard"))
        
        mood_data = load_json_dataset(mood_variations_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(mood_data, "mood_variations"))
        
        long_form_data = load_json_dataset(long_form_technical_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(long_form_data, "standard"))
        
        code_review_data = load_json_dataset(real_code_reviews_path)
        enhanced_training_qa.extend(convert_enhanced_to_qa(code_review_data, "standard"))
        
        print(f"Loaded {len(enhanced_training_qa)} enhanced training examples")
    else:
        print(f"Enhanced training directory not found: {enhanced_training_dir}")
    
    # Handle potentially missing files
    try:
        if not Path(ethical_scenarios_path).exists():
            print(f"Creating placeholder for {ethical_scenarios_path}")
            ethical_scenarios_data = {"description": "Placeholder for ethical scenarios"}
            with open(ethical_scenarios_path, 'w') as f:
                json.dump(ethical_scenarios_data, f, indent=2)
    except Exception as e:
        print(f"Error handling ethical scenarios: {e}")

    # Combine all datasets
    all_qa_pairs = frostline_qa + cybersecurity_qa + software_dev_qa + it_support_qa + hardware_qa + \
                  network_security_qa + programming_concepts_qa + cloud_computing_qa + \
                  advanced_cybersecurity_qa + advanced_programming_qa + advanced_cloud_qa + \
                  shy_dakota_qa + \
                  javascript_dom_qa + typescript_migration_qa + web_framework_qa + \
                  reading_comprehension_qa + grammar_rules_qa + contextual_conversation_qa + \
                  advanced_english_qa + combat_scenarios_qa + \
                  epsilon_personality_qa + ai_fragments_qa + virtual_environment_qa + \
                  theta_identity_qa + enhanced_identity_qa + fragments_under_theta_qa + delta_fragment_qa + \
                  sigma_fragment_qa + omega_fragment_qa + gamma_fragment_qa + \
                  eta_fragment_qa + iota_fragment_qa + beta_fragment_qa + \
                  lambda_fragment_qa + kappa_fragment_qa + \
                  cybersecurity_advanced_qa + quantum_computing_basics_qa + \
                  machine_learning_frameworks_qa + cloud_computing_architecture_qa + \
                  devops_practices_qa + \
                  basic_conversation_qa + common_questions_qa + small_talk_qa + \
                  emotional_responses_qa + motivational_dialogues_qa + conflict_resolution_qa + \
                  theta_personal_stories_qa + fragment_backstories_qa + \
                  combat_theta_qa + technical_theta_qa + personal_theta_qa + \
                  identity_consistency_qa + conversational_coherence_qa + code_block_usage_qa + \
                  factual_accuracy_qa + context_retention_qa + conversation_multiturn_qa + \
                  improved_formatting_qa + technical_accuracy_qa + \
                  weather_api_queries_qa + time_zone_queries_qa + current_events_handling_qa + \
                  web_search_integration_qa + realtime_data_integration_qa + \
                  technical_documentation_search_qa + language_models_rag_qa + \
                  enhanced_training_qa + openassistant_oasst1_qa
    
    # Save processed Q&A data
    with open(output_path, 'w') as file:
        json.dump(all_qa_pairs, file, indent=2)
        
    # Save non-Q&A datasets separately in their original format
    diverse_datasets_dir = datasets_dir / "diverse_curriculum"
    os.makedirs(diverse_datasets_dir, exist_ok=True)
    
    # Function to save a dataset and return its item count
    def save_diverse_dataset(data, filename, dir_path=diverse_datasets_dir):
        file_path = dir_path / filename
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        
        # Count items based on dataset structure
        if isinstance(data, dict):
            # Count experiences in Emotional_learning.json
            if 'experiences' in data and isinstance(data['experiences'], list):
                return len(data['experiences'])
            # Count categories in Personal_preferences.json
            elif 'categories' in data and isinstance(data['categories'], list):
                return len(data['categories'])
            # Count scenarios in Ethical_scenarios.json
            elif 'scenarios' in data and isinstance(data['scenarios'], list):
                return len(data['scenarios']) if 'scenarios' in data else 1
            # Count domains in Technical_concepts.json
            elif 'domains' in data and isinstance(data['domains'], list):
                return len(data['domains'])
            # Count narratives in Narrative_experiences.json
            elif 'narratives' in data and isinstance(data['narratives'], list):
                return len(data['narratives'])
            # Count components in Emotional_intelligence.json
            elif 'emotional_intelligence_framework' in data and \
                 'core_components' in data['emotional_intelligence_framework']:
                return len(data['emotional_intelligence_framework']['core_components'])
            # Count reasoning patterns in Cognitive_reasoning.json
            elif 'reasoning_patterns' in data and isinstance(data['reasoning_patterns'], list):
                return len(data['reasoning_patterns'])
            # Count frameworks in Psychological_frameworks.json
            elif 'frameworks' in data and isinstance(data['frameworks'], list):
                return len(data['frameworks'])
            # Count dynamics in Conversational_dynamics.json
            elif 'dynamics' in data and isinstance(data['dynamics'], list):
                return len(data['dynamics'])
            # Count memory systems in Memory_simulation.json
            elif 'memory_systems' in data and isinstance(data['memory_systems'], list):
                return len(data['memory_systems'])
            # Count humor types in Humor_comprehension.json
            elif 'humor_types' in data and isinstance(data['humor_types'], list):
                return len(data['humor_types'])
            # Count cultural frameworks in Cultural_contexts.json
            elif 'cultural_frameworks' in data and isinstance(data['cultural_frameworks'], list):
                return len(data['cultural_frameworks'])
            # Count ethical systems in Ethical_reasoning.json
            elif 'ethical_systems' in data and isinstance(data['ethical_systems'], list):
                return len(data['ethical_systems'])
            # Count tactical domains in Tactical_knowledge.json
            elif 'domains' in data and isinstance(data['domains'], list):
                # This structure is shared with Technical_concepts.json but counting here for clarity
                return len(data['domains'])
            else:
                return 1
        return 0
    
    # Save each non-Q&A dataset
    emotional_learning_count = save_diverse_dataset(emotional_learning_data, "emotional_learning.json")
    personal_preferences_count = save_diverse_dataset(personal_preferences_data, "personal_preferences.json")
    ethical_scenarios_count = save_diverse_dataset(ethical_scenarios_data, "ethical_scenarios.json")
    technical_concepts_count = save_diverse_dataset(technical_concepts_data, "technical_concepts.json")
    narrative_experiences_count = save_diverse_dataset(narrative_experiences_data, "narrative_experiences.json")
    emotional_intelligence_count = save_diverse_dataset(emotional_intelligence_data, "emotional_intelligence.json")
    
    # Save additional diverse curriculum datasets
    cognitive_reasoning_count = save_diverse_dataset(cognitive_reasoning_data, "cognitive_reasoning.json")
    psychological_frameworks_count = save_diverse_dataset(psychological_frameworks_data, "psychological_frameworks.json")
    conversational_dynamics_count = save_diverse_dataset(conversational_dynamics_data, "conversational_dynamics.json")
    human_experience_simulation_count = save_diverse_dataset(human_experience_simulation_data, "human_experience_simulation.json")
    humor_comprehension_count = save_diverse_dataset(humor_comprehension_data, "humor_comprehension.json")
    cultural_contexts_count = save_diverse_dataset(cultural_contexts_data, "cultural_contexts.json")
    ethical_reasoning_count = save_diverse_dataset(ethical_reasoning_data, "ethical_reasoning.json")
    tactical_knowledge_count = save_diverse_dataset(tactical_knowledge_data, "tactical_knowledge.json")
    interpersonal_intelligence_count = save_diverse_dataset(interpersonal_intelligence_data, "interpersonal_intelligence.json")
    memory_simulation_count = save_diverse_dataset(memory_simulation_data, "memory_simulation.json")
    
    # Print summary
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
    print(f"- {len(shy_dakota_qa)} personal data examples (Dakota & Shyanne)")
    print(f"- {len(javascript_dom_qa)} JavaScript DOM interaction examples")
    print(f"- {len(typescript_migration_qa)} TypeScript migration examples")
    print(f"- {len(web_framework_qa)} web framework integration examples")
    print(f"- {len(reading_comprehension_qa)} reading comprehension examples")
    print(f"- {len(grammar_rules_qa)} grammar and language rules examples")
    print(f"- {len(contextual_conversation_qa)} contextual conversation examples")
    print(f"- {len(advanced_english_qa)} advanced English usage examples")
    print(f"- {len(combat_scenarios_qa)} combat scenarios examples")
    print(f"- {len(epsilon_personality_qa)} Epsilon personality traits examples")
    print(f"- {len(ai_fragments_qa)} AI fragments capabilities examples")
    print(f"- {len(virtual_environment_qa)} virtual environment operations examples")
    print(f"- {len(theta_identity_qa)} Theta as Alpha AI identity examples")
    print(f"- {len(enhanced_identity_qa)} enhanced Theta identity examples")
    print(f"- {len(fragments_under_theta_qa)} fragments under Theta governance examples")
    print(f"- {len(delta_fragment_qa)} Delta fragment examples")
    print(f"- {len(sigma_fragment_qa)} Sigma fragment examples")
    print(f"- {len(omega_fragment_qa)} Omega fragment examples")
    print(f"- {len(gamma_fragment_qa)} Gamma fragment examples")
    print(f"- {len(eta_fragment_qa)} Eta fragment examples")
    print(f"- {len(iota_fragment_qa)} Iota fragment examples")
    print(f"- {len(beta_fragment_qa)} Beta fragment examples")
    print(f"- {len(lambda_fragment_qa)} Lambda fragment examples")
    print(f"- {len(kappa_fragment_qa)} Kappa fragment examples")
    print(f"- {len(cybersecurity_advanced_qa)} advanced cybersecurity practices examples")
    print(f"- {len(quantum_computing_basics_qa)} quantum computing basics examples")
    print(f"- {len(machine_learning_frameworks_qa)} machine learning frameworks examples")
    print(f"- {len(cloud_computing_architecture_qa)} cloud computing architecture examples")
    print(f"- {len(devops_practices_qa)} DevOps practices examples")
    print(f"- {len(basic_conversation_qa)} basic conversation examples")
    print(f"- {len(common_questions_qa)} common questions examples")
    print(f"- {len(small_talk_qa)} small talk examples")
    print(f"- {len(emotional_responses_qa)} emotional intelligence responses")
    print(f"- {len(motivational_dialogues_qa)} motivational dialogue examples")
    print(f"- {len(conflict_resolution_qa)} conflict resolution responses")
    print(f"- {len(theta_personal_stories_qa)} Theta personal stories")
    print(f"- {len(fragment_backstories_qa)} fragment backstory examples")
    print(f"- {len(combat_theta_qa)} combat-domain Theta integration examples")
    print(f"- {len(technical_theta_qa)} technical-domain Theta integration examples")
    print(f"- {len(personal_theta_qa)} personal-domain Theta interaction examples")
    
    # Print counts for new conversational improvement datasets
    print(f"- {len(identity_consistency_qa)} identity consistency examples")
    print(f"- {len(conversational_coherence_qa)} conversational coherence examples")
    print(f"- {len(code_block_usage_qa)} code block usage examples")
    print(f"- {len(factual_accuracy_qa)} factual accuracy examples")
    print(f"- {len(context_retention_qa)} context retention examples")
    print(f"- {len(conversation_multiturn_qa)} multi-turn conversation examples")
    print(f"- {len(improved_formatting_qa)} improved formatting examples")
    print(f"- {len(technical_accuracy_qa)} technical accuracy examples")
    print(f"\nWeb Search Integration Datasets:")
    print(f"- {len(weather_api_queries_qa)} weather API query examples")
    print(f"- {len(time_zone_queries_qa)} time zone query examples")
    print(f"- {len(current_events_handling_qa)} current events handling examples")
    print(f"- {len(web_search_integration_qa)} web search integration examples")
    print(f"- {len(realtime_data_integration_qa)} real-time data integration examples")
    print(f"- {len(technical_documentation_search_qa)} technical documentation search examples")
    print(f"- {len(language_models_rag_qa)} language models with RAG examples")
    # Human-like DPO dataset is handled separately as part of mixed curriculum
    
    # Count total diverse curriculum items
    # Save Human-Like DPO Dataset in its original format
    human_like_dpo_count = 0
    try:
        human_like_dpo_file = diverse_datasets_dir / "human_like_dpo.json"
        with open(human_like_dpo_file, 'w') as file:
            json.dump(human_like_dpo_data, file, indent=2)
            
        # Count entries if they exist
        if isinstance(human_like_dpo_data, dict) and "entries" in human_like_dpo_data:
            human_like_dpo_count = len(human_like_dpo_data["entries"])
    except Exception as e:
        print(f"Error saving human_like_dpo_data: {e}")
        human_like_dpo_count = 0
    
    # Save OpenMathInstruct-1 Dataset in its original problem-solution format
    openmath_instruct_count = 0
    try:
        openmath_instruct_file = diverse_datasets_dir / "openmath_instruct_1.json"
        with open(openmath_instruct_file, 'w') as file:
            json.dump(openmath_instruct_data, file, indent=2)
            
        # Count entries if they exist
        if isinstance(openmath_instruct_data, dict) and "entries" in openmath_instruct_data:
            openmath_instruct_count = len(openmath_instruct_data["entries"])
        elif isinstance(openmath_instruct_data, list):
            openmath_instruct_count = len(openmath_instruct_data)
    except Exception as e:
        print(f"Error saving openmath_instruct_data: {e}")
        openmath_instruct_count = 0
        
    print(f"- {human_like_dpo_count} from Human-Like DPO Dataset (original format)")
    print(f"- {openmath_instruct_count} from OpenMathInstruct-1 Dataset (problem-solution format)")
    
    # Save Theta Opinions dataset (specialized opinion format - not Q&A)
    theta_opinions_count = 0
    try:
        theta_opinions_file = diverse_datasets_dir / "theta_opinions.json"
        with open(theta_opinions_file, 'w', encoding='utf-8') as file:
            json.dump(theta_opinions_data, file, indent=2)
        
        if isinstance(theta_opinions_data, list):
            theta_opinions_count = len(theta_opinions_data)
        print(f"Saved theta_opinions.json with {theta_opinions_count} opinion entries")
    except Exception as e:
        print(f"Error saving theta_opinions_data: {e}")
    
    # Save Natural Conversational datasets (specialized personality format)
    natural_conversation_count = 0
    try:
        # Save basic_conversation_natural
        if basic_conversation_natural_data:
            basic_natural_file = diverse_datasets_dir / "basic_conversation_natural.json"
            with open(basic_natural_file, 'w', encoding='utf-8') as file:
                json.dump(basic_conversation_natural_data, file, indent=2)
            if isinstance(basic_conversation_natural_data, list):
                natural_conversation_count += len(basic_conversation_natural_data)
        
        # Save small_talk_natural
        if small_talk_natural_data:
            small_talk_natural_file = diverse_datasets_dir / "small_talk_natural.json"
            with open(small_talk_natural_file, 'w', encoding='utf-8') as file:
                json.dump(small_talk_natural_data, file, indent=2)
            if isinstance(small_talk_natural_data, list):
                natural_conversation_count += len(small_talk_natural_data)
        
        print(f"Saved natural conversation datasets with {natural_conversation_count} entries")
    except Exception as e:
        print(f"Error saving natural conversation data: {e}")
    
    # Save Technical Patterns datasets (specialized non-Q&A format)
    technical_patterns_count = 0
    try:
        patterns_datasets = [
            ("code_review_patterns.json", code_review_patterns_data),
            ("security_vulnerabilities.json", security_vulnerabilities_data),
            ("architecture_patterns.json", architecture_patterns_data),
            ("debugging_scenarios.json", debugging_scenarios_data),
            ("api_design_patterns.json", api_design_patterns_data),
            ("error_handling_patterns.json", error_handling_patterns_data),
        ]
        
        for filename, data in patterns_datasets:
            if data:
                pattern_file = diverse_datasets_dir / filename
                with open(pattern_file, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2)
                if isinstance(data, list):
                    technical_patterns_count += len(data)
                elif isinstance(data, dict):
                    technical_patterns_count += 1
        
        print(f"Saved technical patterns datasets with {technical_patterns_count} entries")
    except Exception as e:
        print(f"Error saving technical patterns data: {e}")
    
    print(f"- {theta_opinions_count} from Theta Opinions (specialized opinion format)")
    print(f"- {natural_conversation_count} from Natural Conversation datasets")
    print(f"- {technical_patterns_count} from Technical Patterns datasets")
    
    total_diverse_items = human_like_dpo_count + openmath_instruct_count + theta_opinions_count + natural_conversation_count + technical_patterns_count + \
                       emotional_learning_count + personal_preferences_count + ethical_scenarios_count + \
                       technical_concepts_count + narrative_experiences_count + emotional_intelligence_count + \
                       cognitive_reasoning_count + psychological_frameworks_count + conversational_dynamics_count + \
                       human_experience_simulation_count + humor_comprehension_count + cultural_contexts_count + \
                       ethical_reasoning_count + tactical_knowledge_count + interpersonal_intelligence_count + \
                       memory_simulation_count
    
    # Print information about the non-Q&A datasets
    print(f"\nDiverse Curriculum Datasets (Non-Q&A Format):")
    print(f"Saved to {diverse_datasets_dir}")
    print(f"Processed {total_diverse_items} diverse curriculum items total:")
    print(f"- {emotional_learning_count} emotional learning scenarios")
    print(f"- {personal_preferences_count} personal preferences categories")
    print(f"- {ethical_scenarios_count} ethical scenarios")
    print(f"- {technical_concepts_count} technical concept domains")
    print(f"- {narrative_experiences_count} narrative experiences")
    print(f"- {emotional_intelligence_count} emotional intelligence components")
    print(f"- {cognitive_reasoning_count} cognitive reasoning patterns")
    print(f"- {psychological_frameworks_count} psychological frameworks")
    print(f"- {conversational_dynamics_count} conversational dynamics")
    print(f"- {human_experience_simulation_count} human experience simulations")
    print(f"- {humor_comprehension_count} humor comprehension patterns")
    print(f"- {cultural_contexts_count} cultural contexts")
    print(f"- {ethical_reasoning_count} ethical reasoning frameworks")
    print(f"- {tactical_knowledge_count} tactical knowledge domains")
    print(f"- {interpersonal_intelligence_count} interpersonal intelligence patterns")
    print(f"- {memory_simulation_count} memory simulation patterns")
    
    return output_path

if __name__ == "__main__":
    process_data()
