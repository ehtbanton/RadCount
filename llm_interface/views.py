from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pathlib import Path
import json
import subprocess
import sys
import platform
import csv
from datetime import datetime
import requests
import re

from .llm_service import LlamaService


def try_fix_truncated_json_array(text):
    """
    Attempt to fix a truncated JSON array by removing the incomplete last element
    and closing the array properly.
    """
    # Find the last complete object (ends with })
    last_complete = text.rfind('},')
    if last_complete == -1:
        last_complete = text.rfind('}')

    if last_complete == -1:
        return None

    # Take everything up to and including the last complete object
    fixed = text[:last_complete + 1]

    # Close the array
    if not fixed.rstrip().endswith(']'):
        fixed = fixed.rstrip().rstrip(',') + '\n]'

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def extract_json_array(text):
    """
    Robustly extract a JSON array from LLM response text.
    Handles cases where the LLM includes extra text around the JSON,
    markdown code blocks, and brackets within string values.
    Also handles indexed text like [0]word [1]word by finding real JSON arrays.
    Can salvage truncated responses by extracting complete entities.
    """
    original_text = text  # Keep for error reporting

    # Strip markdown code blocks first (common LLM pattern)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Strategy 1: Try direct parse if it looks like clean JSON
    if text.startswith('['):
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass  # Try other strategies

    # Strategy 2: Find JSON array start - look for '[' followed by '{' or '[' or ']'
    # This avoids matching indexed text like [0]word
    start_idx = -1
    for i, char in enumerate(text):
        if char == '[':
            # Check what follows - should be whitespace then {/[/]/newline for a real JSON array
            rest = text[i+1:].lstrip()
            if not rest:  # End of string
                continue
            # Real JSON array starts with [ followed by { or [ or ] or whitespace+{
            if rest[0] in '{[]':
                start_idx = i
                break
            # Also check for newline followed by {
            if rest[0] == '\n' or rest[0] == '\r':
                rest2 = rest.lstrip()
                if rest2 and rest2[0] == '{':
                    start_idx = i
                    break

    if start_idx == -1:
        # Strategy 3: Look for '[\n  {' or '[\r\n  {' pattern (formatted JSON)
        match = re.search(r'\[\s*\{', text)
        if match:
            start_idx = match.start()

    if start_idx == -1:
        # Strategy 4: Fallback - try each '[' and see if it produces valid JSON
        for i, char in enumerate(text):
            if char == '[':
                # Try to find matching ] and parse
                test_text = text[i:]
                bracket_count = 0
                in_string = False
                escape_next = False
                for j, c in enumerate(test_text):
                    if escape_next:
                        escape_next = False
                        continue
                    if c == '\\' and in_string:
                        escape_next = True
                        continue
                    if c == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if c == '[':
                            bracket_count += 1
                        elif c == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                candidate = test_text[:j+1]
                                try:
                                    result = json.loads(candidate)
                                    if isinstance(result, list):
                                        return result, None
                                except json.JSONDecodeError:
                                    pass
                                break

    if start_idx == -1:
        return None, f"No JSON array found in response. Response preview: {text[:300]}"

    # Find matching closing bracket, being aware of string contexts
    bracket_count = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i
                    break

    if end_idx == -1:
        # Response might be truncated - try to salvage complete entities
        partial_json = text[start_idx:] if start_idx >= 0 else text
        fixed_result = try_fix_truncated_json_array(partial_json)
        if fixed_result and len(fixed_result) > 0:
            # Return what we could salvage with a warning
            return fixed_result, None  # Successfully salvaged
        return None, f"Unbalanced brackets in JSON array (response may be truncated). Text from start: {text[start_idx:start_idx+300]}"

    json_str = text[start_idx:end_idx + 1]

    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        # Try to fix truncated JSON
        fixed_result = try_fix_truncated_json_array(json_str)
        if fixed_result and len(fixed_result) > 0:
            return fixed_result, None  # Successfully salvaged
        # Try to provide more context in error
        preview = json_str[:300] + "..." if len(json_str) > 300 else json_str
        return None, f"JSON parse error: {str(e)}. Preview: {preview}"


def extract_json_object(text):
    """
    Robustly extract a JSON object from LLM response text.
    Handles cases where the LLM includes extra text around the JSON,
    markdown code blocks, and braces within string values.
    """
    # Strip markdown code blocks first (common LLM pattern)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = text.strip()

    start_idx = text.find('{')
    if start_idx == -1:
        return None, "No JSON object found in response"

    # Find matching closing brace, being aware of string contexts
    brace_count = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

    if end_idx == -1:
        return None, "Unbalanced braces in JSON object"

    json_str = text[start_idx:end_idx + 1]

    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"


def create_indexed_report_text(report_text):
    """
    Create an indexed version of the report text where each word is preceded
    by its 0-based index in brackets.

    Example: "The patient has a tumour" -> "[0]The [1]patient [2]has [3]a [4]tumour"

    Returns:
        tuple: (indexed_text, word_list) where word_list is the list of original words
    """
    # Split by whitespace while preserving newlines for readability
    lines = report_text.split('\n')
    indexed_lines = []
    word_index = 0
    all_words = []

    for line in lines:
        words = line.split()
        indexed_words = []
        for word in words:
            indexed_words.append(f"[{word_index}]{word}")
            all_words.append(word)
            word_index += 1
        indexed_lines.append(' '.join(indexed_words))

    return '\n'.join(indexed_lines), all_words


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama_cpp"


# Helper function to build system prompt from schema
def build_system_prompt_from_schema(schema):
    """Generate extraction prompt dynamically from schema configuration."""
    prompt = f"""You are a clinical information extraction system specialized in radiology reports. Extract entities and relations from the provided radiology report using the {schema['name']} schema.

SCHEMA: {schema['name']}
{schema['description']}

ENTITY TYPES ({len(schema['entity_types'])} types):

"""

    for i, entity_type in enumerate(schema['entity_types'], 1):
        prompt += f"{i}. **{entity_type['name']}**: {entity_type['description']}\n"

    prompt += f"\nRELATION TYPES ({len(schema['relation_types'])} types):\n\n"

    for i, relation_type in enumerate(schema['relation_types'], 1):
        prompt += f"{i}. **{relation_type['name']}**: {relation_type['description']}\n"
        if relation_type.get('valid_pairs'):
            pairs_str = ", ".join([f"({pair[0]} → {pair[1]})" for pair in relation_type['valid_pairs'][:3]])
            if len(relation_type['valid_pairs']) > 3:
                pairs_str += f" (and {len(relation_type['valid_pairs']) - 3} more)"
            prompt += f"   Valid pairs: {pairs_str}\n"

    prompt += """

EXTRACTION INSTRUCTIONS:

1. Read the radiology report carefully
2. Identify ALL entities that fit the entity types defined above
3. For each entity, determine its type based on the definitions
4. Identify relationships between entities using the relation types defined above
5. Extract as many valid entity-relation triplets as possible
6. Be thorough - extract ALL clinically relevant information
7. For measurements (sizes, SUVmax values), extract them as separate entities
8. Preserve anatomical precision (e.g., "segment II", "T8 vertebra", "right upper lobe")

OUTPUT FORMAT:

Return a JSON array of triplets. Each triplet must have this exact structure:
{
  "entity1_text": "the first entity text",
  "entity1_type": "one of the entity types listed above",
  "relation_type": "one of the relation types listed above",
  "entity2_text": "the second entity text",
  "entity2_type": "one of the entity types listed above"
}

IMPORTANT:
- Return ONLY the JSON array, no additional text
- Ensure all JSON is valid and properly formatted
- Extract ALL relevant entities and relations
- Be consistent with entity type assignments
- Every relation must connect two valid entities
- For measurements, extract both the value and link it to the lesion
"""

    return prompt


# Legacy: RadGraph System Prompt for Entity-Relation Extraction (kept for reference)
RADGRAPH_SYSTEM_PROMPT_LEGACY = """You are a clinical information extraction system specialized in radiology reports. Extract entities and relations from the provided radiology report using the RadGraph schema.

ENTITY TYPES (4 types):

1. **Anatomy**: Anatomical body parts that appear in the radiology report.
   - Examples: "lung", "heart", "right lower lobe", "pleura", "aorta", "mediastinum"
   - Include specific anatomical locations and modifiers

2. **Observation:Definitely Present**: Visual features, pathophysiologic processes, or diagnostic disease classifications that are CONFIRMED to be present.
   - Examples: "opacity", "effusion", "consolidation", "enlarged", "fracture", "pneumonia"
   - Use this for findings that are stated as definite or present without uncertainty

3. **Observation:Uncertain**: Visual features or findings that are POSSIBLY present or SUSPECTED.
   - Examples: "possible infiltrate", "suspected pneumonia", "questionable nodule"
   - Use this for findings with uncertainty markers like "possible", "suspected", "questionable", "concerning for", "suggestive"

4. **Observation:Definitely Absent**: Findings that are EXPLICITLY RULED OUT or stated as NOT present.
   - Examples: "no effusion", "no acute findings", "pneumothorax is absent"
   - Use this for explicit negations, not just absence of mention

RELATION TYPES (3 types):

1. **suggestive_of**: One observation implies or suggests another observation.
   - Format: (Observation → Observation)
   - Example: "infiltrate suggestive_of pneumonia"
   - Use when one finding raises suspicion for a diagnosis

2. **located_at**: An observation is related to or located at an anatomical structure.
   - Format: (Observation → Anatomy)
   - Example: "opacity located_at right lower lobe"
   - Use to link findings to anatomical locations

3. **modify**: One entity modifies, quantifies, or describes another entity.
   - Format: (Observation → Observation) OR (Anatomy → Anatomy)
   - Examples:
     - "increased modify opacity" (one observation modifying another)
     - "right modify lobe" (one anatomy modifying another)
   - Use for descriptors, qualifiers, size terms, temporal changes

EXTRACTION INSTRUCTIONS:

1. Read the radiology report carefully
2. Identify ALL entities that fit the four entity types above
3. For each entity, determine its type based on the definitions
4. Identify relationships between entities using the three relation types
5. Extract as many valid entity-relation triplets as possible
6. Be thorough - extract ALL clinically relevant information

OUTPUT FORMAT:

Return a JSON array of triplets. Each triplet must have this exact structure:
{
  "entity1_text": "the first entity text",
  "entity1_type": "one of: Anatomy, Observation:Definitely Present, Observation:Uncertain, Observation:Definitely Absent",
  "relation_type": "one of: suggestive_of, located_at, modify",
  "entity2_text": "the second entity text",
  "entity2_type": "one of: Anatomy, Observation:Definitely Present, Observation:Uncertain, Observation:Definitely Absent"
}

EXAMPLE:

Report: "Increased right lower lobe opacity, concerning for infection. No pleural effusion."

Output:
[
  {
    "entity1_text": "increased",
    "entity1_type": "Observation:Definitely Present",
    "relation_type": "modify",
    "entity2_text": "opacity",
    "entity2_type": "Observation:Definitely Present"
  },
  {
    "entity1_text": "right",
    "entity1_type": "Anatomy",
    "relation_type": "modify",
    "entity2_text": "lobe",
    "entity2_type": "Anatomy"
  },
  {
    "entity1_text": "lower",
    "entity1_type": "Anatomy",
    "relation_type": "modify",
    "entity2_text": "lobe",
    "entity2_type": "Anatomy"
  },
  {
    "entity1_text": "opacity",
    "entity1_type": "Observation:Definitely Present",
    "relation_type": "located_at",
    "entity2_text": "lobe",
    "entity2_type": "Anatomy"
  },
  {
    "entity1_text": "opacity",
    "entity1_type": "Observation:Definitely Present",
    "relation_type": "suggestive_of",
    "entity2_text": "infection",
    "entity2_type": "Observation:Uncertain"
  },
  {
    "entity1_text": "effusion",
    "entity1_type": "Observation:Definitely Absent",
    "relation_type": "located_at",
    "entity2_text": "pleura",
    "entity2_type": "Anatomy"
  }
]

IMPORTANT:
- Return ONLY the JSON array, no additional text
- Ensure all JSON is valid and properly formatted
- Extract ALL relevant entities and relations
- Be consistent with entity type assignments
- Every relation must connect two valid entities
"""


def detect_gpu():
    """Detect available GPU hardware."""
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"

    return "cpu"


def get_available_models():
    """Get list of available models from both the models directory and F:/_llm_models/."""
    models = []

    # Define directories to search
    search_dirs = [MODELS_DIR]
    external_models_dir = Path("F:/_llm_models")
    if external_models_dir.exists():
        search_dirs.append(external_models_dir)

    # Search all directories
    for models_dir in search_dirs:
        if not models_dir.exists():
            continue

        for model_file in models_dir.glob("*.gguf"):
            # Skip mmproj files
            if not model_file.name.startswith("mmproj"):
                models.append({
                    'filename': model_file.name,
                    'path': str(model_file),
                    'size_mb': model_file.stat().st_size / 1024 / 1024
                })

    return models


def get_current_model():
    """Get the currently loaded model filename."""
    model_file = PROJECT_ROOT / "llama_server.model"
    if model_file.exists():
        try:
            with open(model_file, 'r') as f:
                return f.read().strip()
        except:
            pass
    return None


def set_current_model(model_filename):
    """Store the currently loaded model filename."""
    model_file = PROJECT_ROOT / "llama_server.model"
    with open(model_file, 'w') as f:
        f.write(model_filename)


def clear_current_model():
    """Clear the currently loaded model."""
    model_file = PROJECT_ROOT / "llama_server.model"
    if model_file.exists():
        model_file.unlink()


def get_current_context_size():
    """Get the currently configured context size."""
    context_size_file = PROJECT_ROOT / "llama_server.context_size"
    if context_size_file.exists():
        try:
            with open(context_size_file, 'r') as f:
                return int(f.read().strip())
        except:
            pass
    return 4096  # Default context size


def set_current_context_size(context_size):
    """Store the currently configured context size."""
    context_size_file = PROJECT_ROOT / "llama_server.context_size"
    with open(context_size_file, 'w') as f:
        f.write(str(context_size))


def clear_current_context_size():
    """Clear the currently configured context size."""
    context_size_file = PROJECT_ROOT / "llama_server.context_size"
    if context_size_file.exists():
        context_size_file.unlink()


def get_large_files():
    """Get list of large data files from large_data directory."""
    large_files = []
    large_data_dir = PROJECT_ROOT / "large_data"

    if not large_data_dir.exists():
        return large_files

    for file_path in large_data_dir.iterdir():
        if file_path.is_file():
            large_files.append({
                'filename': file_path.name,
                'size_mb': file_path.stat().st_size / 1024 / 1024
            })

    return large_files


def get_csv_metadata():
    """Get metadata about the CSV file (if it exists)."""
    large_data_dir = PROJECT_ROOT / "large_data"

    if not large_data_dir.exists():
        return None

    # Look for a CSV file
    csv_files = list(large_data_dir.glob("*.csv"))

    if not csv_files:
        return None

    csv_file = csv_files[0]

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read first row as headers

            # Count rows (excluding header)
            row_count = sum(1 for _ in reader)

        return {
            'filename': csv_file.name,
            'headers': headers,
            'row_count': row_count,
            'size_mb': csv_file.stat().st_size / 1024 / 1024
        }
    except Exception as e:
        return None


def home(request):
    """Render the main LLM interface page."""
    llm_service = LlamaService()
    context_files = llm_service.get_context_files_info()

    # Calculate total file count
    total_files = (len(context_files['system_files']) +
                   len(context_files['user_files']) +
                   len(context_files['image_files']))

    # Check if we're editing a file
    edit_filename = request.GET.get('edit')
    edit_content = None
    if edit_filename:
        # Security check
        if '..' not in edit_filename and '/' not in edit_filename and '\\' not in edit_filename:
            if edit_filename.endswith('.txt') or edit_filename.endswith('.json'):
                context_dir = PROJECT_ROOT / "Context"
                file_path = context_dir / edit_filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            edit_content = f.read()
                    except:
                        pass

    # Get large files information
    large_files = get_large_files()
    csv_metadata = get_csv_metadata()

    context = {
        'server_running': llm_service.is_server_running(),
        'model_info': llm_service.get_model_info(),
        'available_models': get_available_models(),
        'current_model': get_current_model(),
        'current_context_size': get_current_context_size(),
        'context_files': context_files,
        'total_context_files': total_files,
        'large_files': large_files,
        'total_large_files': len(large_files),
        'csv_metadata': csv_metadata,
        'edit_filename': edit_filename,
        'edit_content': edit_content,
    }

    return render(request, 'llm_interface/home.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def generate(request):
    """Generate a response from the LLM based on context files."""
    try:
        # Parse request parameters
        data = json.loads(request.body) if request.body else {}
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 512))

        # Generate response
        llm_service = LlamaService()

        if not llm_service.is_server_running():
            return JsonResponse({
                'success': False,
                'error': 'LLM server is not running. Please check startup logs.'
            }, status=503)

        result = llm_service.generate_response(
            temperature=temperature,
            max_tokens=max_tokens
        )

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def status(request):
    """Get the status of the LLM server and model info."""
    llm_service = LlamaService()

    return JsonResponse({
        'server_running': llm_service.is_server_running(),
        'model_info': llm_service.get_model_info(),
        'available_models': get_available_models(),
        'current_model': get_current_model(),
        'current_context_size': get_current_context_size(),
    })


@require_http_methods(["GET"])
def token_count(request):
    """Get the total token count for all context files."""
    llm_service = LlamaService()

    if not llm_service.is_server_running():
        return JsonResponse({
            'success': False,
            'error': 'LLM server is not running. Start the server to calculate tokens.'
        }, status=503)

    try:
        counts = llm_service.get_total_context_token_count()
        return JsonResponse({
            'success': True,
            'counts': counts
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to calculate token count: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def start_server(request):
    """Start the llama.cpp server with specified model."""
    try:
        data = json.loads(request.body) if request.body else {}
        model_filename = data.get('model')
        context_size = data.get('context_size', 4096)

        if not model_filename:
            return JsonResponse({
                'success': False,
                'error': 'No model specified'
            }, status=400)

        # Find the model file in either directory
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            # Try external models directory
            external_model_path = Path("F:/_llm_models") / model_filename
            if external_model_path.exists():
                model_path = external_model_path
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Model file not found: {model_filename}'
                }, status=404)

        # Check if server is already running
        llm_service = LlamaService()
        if llm_service.is_server_running():
            return JsonResponse({
                'success': False,
                'error': 'Server is already running. Stop it first before starting with a different model.'
            }, status=400)

        # Find llama-server executable
        if sys.platform == "win32":
            server_exe = None
            for potential_exe in LLAMA_CPP_DIR.rglob("llama-server.exe"):
                server_exe = potential_exe
                break
            if not server_exe:
                server_exe = LLAMA_CPP_DIR / "llama-server.exe"
        else:
            server_exe = None
            for potential_exe in LLAMA_CPP_DIR.rglob("llama-server"):
                server_exe = potential_exe
                break
            if not server_exe:
                server_exe = LLAMA_CPP_DIR / "llama-server"

        if not server_exe or not server_exe.exists():
            return JsonResponse({
                'success': False,
                'error': 'llama-server executable not found. Please run startup.py to install llama.cpp.'
            }, status=500)

        # Check for mmproj file
        mmproj_path = None
        for mmproj_file in MODELS_DIR.glob("mmproj*.gguf"):
            mmproj_path = mmproj_file
            break

        # Detect GPU for optimal settings
        gpu_type = detect_gpu()
        gpu_layers = "33" if gpu_type in ["cuda", "metal"] else "0"

        # Build command
        cmd = [
            str(server_exe),
            "-m", str(model_path),
            "--host", "127.0.0.1",
            "--port", "8080",
            "-c", str(context_size),
            "--n-gpu-layers", gpu_layers,
        ]

        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])

        # Start server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(LLAMA_CPP_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )

        # Store process ID for later stopping
        import os
        pid_file = PROJECT_ROOT / "llama_server.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        # Store the current model and context size
        set_current_model(model_filename)
        set_current_context_size(context_size)

        return JsonResponse({
            'success': True,
            'message': f'Server starting with model: {model_filename}',
            'pid': process.pid
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to start server: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def stop_server(request):
    """Stop the llama.cpp server."""
    try:
        pid_file = PROJECT_ROOT / "llama_server.pid"

        if not pid_file.exists():
            return JsonResponse({
                'success': False,
                'error': 'No server PID file found. Server may not be running.'
            }, status=404)

        # Read PID
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Kill process
        if sys.platform == "win32":
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=False)
        else:
            import signal
            import os
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Remove PID file
        pid_file.unlink()

        # Clear current model
        clear_current_model()

        return JsonResponse({
            'success': True,
            'message': 'Server stopped successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to stop server: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def upload_context(request):
    """Upload a context file to the Context directory."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file provided'
            }, status=400)

        uploaded_file = request.FILES['file']
        filename = uploaded_file.name

        # Validate file extension
        allowed_extensions = ['.txt', '.json', '.jpg', '.jpeg', '.png']
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return JsonResponse({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            }, status=400)

        # Ensure Context directory exists
        context_dir = PROJECT_ROOT / "Context"
        context_dir.mkdir(exist_ok=True)

        # Save the file
        file_path = context_dir / filename
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Validate JSON files
        if file_ext == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Try to parse JSON
            except json.JSONDecodeError:
                file_path.unlink()  # Delete invalid JSON
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid JSON format'
                }, status=400)

        return JsonResponse({
            'success': True,
            'message': f'File {filename} uploaded successfully',
            'filename': filename
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to upload file: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_context(request):
    """Get the content of a context file."""
    try:
        filename = request.GET.get('filename')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        # Only allow .txt and .json files for editing
        if not (filename.endswith('.txt') or filename.endswith('.json')):
            return JsonResponse({
                'success': False,
                'error': 'Only text and JSON files can be edited'
            }, status=400)

        context_dir = PROJECT_ROOT / "Context"
        file_path = context_dir / filename

        if not file_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'File {filename} not found'
            }, status=404)

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return JsonResponse({
            'success': True,
            'filename': filename,
            'content': content
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to read file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_context(request):
    """Save or update a context file (always overwrites if exists)."""
    try:
        data = json.loads(request.body) if request.body else {}
        filename = data.get('filename', '').strip()
        content = data.get('content', '')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Validate filename
        if not (filename.endswith('.txt') or filename.endswith('.json')):
            return JsonResponse({
                'success': False,
                'error': 'Filename must end with .txt or .json'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        # Validate JSON content if it's a JSON file
        if filename.endswith('.json'):
            try:
                json.loads(content)  # Validate JSON syntax
            except json.JSONDecodeError:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid JSON content'
                }, status=400)

        # Ensure Context directory exists
        context_dir = PROJECT_ROOT / "Context"
        context_dir.mkdir(exist_ok=True)

        file_path = context_dir / filename

        # Save the file (overwrite if exists)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return JsonResponse({
            'success': True,
            'message': f'File {filename} saved successfully',
            'filename': filename
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_context(request):
    """Delete a context file from the Context directory."""
    try:
        data = json.loads(request.body) if request.body else {}
        filename = data.get('filename')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        context_dir = PROJECT_ROOT / "Context"
        file_path = context_dir / filename

        if not file_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'File {filename} not found'
            }, status=404)

        # Delete the file
        file_path.unlink()

        return JsonResponse({
            'success': True,
            'message': f'File {filename} deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def upload_large(request):
    """Upload a CSV file to the large_data directory. Only one CSV file is allowed at a time."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file provided'
            }, status=400)

        uploaded_file = request.FILES['file']
        filename = uploaded_file.name

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        # Validate that it's a CSV file
        if not filename.lower().endswith('.csv'):
            return JsonResponse({
                'success': False,
                'error': 'Only CSV files are allowed'
            }, status=400)

        # Ensure large_data directory exists
        large_data_dir = PROJECT_ROOT / "large_data"
        large_data_dir.mkdir(exist_ok=True)

        # Delete any existing CSV files (only one CSV file allowed)
        for existing_csv in large_data_dir.glob("*.csv"):
            existing_csv.unlink()

        # Save the file
        file_path = large_data_dir / filename
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Validate CSV format
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Try to read headers
                if not headers:
                    file_path.unlink()  # Delete invalid file
                    return JsonResponse({
                        'success': False,
                        'error': 'CSV file is empty or invalid'
                    }, status=400)
        except Exception as e:
            file_path.unlink()  # Delete invalid file
            return JsonResponse({
                'success': False,
                'error': f'Invalid CSV format: {str(e)}'
            }, status=400)

        return JsonResponse({
            'success': True,
            'message': f'CSV file {filename} uploaded successfully',
            'filename': filename
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to upload file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_large(request):
    """Delete a large file from the large_data directory."""
    try:
        data = json.loads(request.body) if request.body else {}
        filename = data.get('filename')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        large_data_dir = PROJECT_ROOT / "large_data"
        file_path = large_data_dir / filename

        if not file_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'File {filename} not found'
            }, status=404)

        # Delete the file
        file_path.unlink()

        return JsonResponse({
            'success': True,
            'message': f'File {filename} deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete file: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_csv_entry(request):
    """Get a specific entry from the CSV file by row number (1-indexed, excluding header)."""
    try:
        entry_num = request.GET.get('entry')

        if not entry_num:
            return JsonResponse({
                'success': False,
                'error': 'No entry number provided'
            }, status=400)

        try:
            entry_num = int(entry_num)
        except ValueError:
            return JsonResponse({
                'success': False,
                'error': 'Entry number must be an integer'
            }, status=400)

        if entry_num < 1:
            return JsonResponse({
                'success': False,
                'error': 'Entry number must be at least 1'
            }, status=400)

        # Find the CSV file
        large_data_dir = PROJECT_ROOT / "large_data"
        if not large_data_dir.exists():
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_files = list(large_data_dir.glob("*.csv"))
        if not csv_files:
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_file = csv_files[0]

        # Read the CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read headers

            # Skip to the desired row
            for i, row in enumerate(reader, start=1):
                if i == entry_num:
                    # Create a dictionary mapping headers to values
                    entry_data = {}
                    for j, header in enumerate(headers):
                        if j < len(row):
                            entry_data[header] = row[j]
                        else:
                            entry_data[header] = ""

                    return JsonResponse({
                        'success': True,
                        'entry_number': entry_num,
                        'data': entry_data
                    })

            # If we got here, the entry number is out of range
            return JsonResponse({
                'success': False,
                'error': f'Entry {entry_num} not found. File has {i} entries.'
            }, status=404)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to read CSV entry: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_csv_entry_to_context(request):
    """Save a CSV entry as a JSON file in the Context directory."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_num = data.get('entry_number')

        if not entry_num:
            return JsonResponse({
                'success': False,
                'error': 'No entry number provided'
            }, status=400)

        try:
            entry_num = int(entry_num)
        except ValueError:
            return JsonResponse({
                'success': False,
                'error': 'Entry number must be an integer'
            }, status=400)

        # Find the CSV file
        large_data_dir = PROJECT_ROOT / "large_data"
        if not large_data_dir.exists():
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_files = list(large_data_dir.glob("*.csv"))
        if not csv_files:
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_file = csv_files[0]
        csv_filename = csv_file.stem  # Get filename without extension

        # Read the CSV entry
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read headers

            # Find the desired row
            entry_data = None
            for i, row in enumerate(reader, start=1):
                if i == entry_num:
                    # Create a dictionary mapping headers to values
                    entry_data = {}
                    for j, header in enumerate(headers):
                        if j < len(row):
                            entry_data[header] = row[j]
                        else:
                            entry_data[header] = ""
                    break

            if entry_data is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Entry {entry_num} not found in CSV'
                }, status=404)

        # Create filename in format: db_<entry_num>_<csv_filename>.json
        json_filename = f"db_{entry_num}_{csv_filename}.json"

        # Ensure Context directory exists
        context_dir = PROJECT_ROOT / "Context"
        context_dir.mkdir(exist_ok=True)

        # Save the JSON file
        json_file_path = context_dir / json_filename
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(entry_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': f'Entry {entry_num} saved as {json_filename}',
            'filename': json_filename
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save entry: {str(e)}'
        }, status=500)


def get_extraction_functions_file():
    """Get the path to the extraction functions JSON file."""
    return PROJECT_ROOT / "extraction_functions.json"


def load_extraction_functions():
    """Load extraction functions from JSON file."""
    functions_file = get_extraction_functions_file()

    if not functions_file.exists():
        # Create default function 1 - filter entries containing "lung"
        default_functions = {
            "1": {
                "name": "Lung Entries Filter",
                "description": "Extracts all entries containing the word 'lung'",
                "code": """# Filter entries containing 'lung'
result = []
for entry in all_entries:
    entry_str = json.dumps(entry).lower()
    if 'lung' in entry_str:
        result.append(entry)"""
            }
        }

        with open(functions_file, 'w', encoding='utf-8') as f:
            json.dump(default_functions, f, indent=2, ensure_ascii=False)

        return default_functions

    try:
        with open(functions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}


def save_extraction_functions(functions):
    """Save extraction functions to JSON file."""
    functions_file = get_extraction_functions_file()
    with open(functions_file, 'w', encoding='utf-8') as f:
        json.dump(functions, f, indent=2, ensure_ascii=False)


@require_http_methods(["GET"])
def get_extraction_functions(request):
    """Get all extraction functions."""
    try:
        functions = load_extraction_functions()
        return JsonResponse({
            'success': True,
            'functions': functions
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load extraction functions: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_extraction_function(request):
    """Save or update an extraction function."""
    try:
        data = json.loads(request.body) if request.body else {}
        function_id = data.get('function_id')
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        code = data.get('code', '').strip()

        if not function_id:
            return JsonResponse({
                'success': False,
                'error': 'No function ID provided'
            }, status=400)

        if not name or not code:
            return JsonResponse({
                'success': False,
                'error': 'Name and code are required'
            }, status=400)

        functions = load_extraction_functions()
        functions[str(function_id)] = {
            'name': name,
            'description': description,
            'code': code
        }

        save_extraction_functions(functions)

        return JsonResponse({
            'success': True,
            'message': f'Function {function_id} saved successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save function: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_extraction_function(request):
    """Delete an extraction function."""
    try:
        data = json.loads(request.body) if request.body else {}
        function_id = data.get('function_id')

        if not function_id:
            return JsonResponse({
                'success': False,
                'error': 'No function ID provided'
            }, status=400)

        functions = load_extraction_functions()

        if str(function_id) not in functions:
            return JsonResponse({
                'success': False,
                'error': f'Function {function_id} not found'
            }, status=404)

        del functions[str(function_id)]
        save_extraction_functions(functions)

        return JsonResponse({
            'success': True,
            'message': f'Function {function_id} deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete function: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def execute_extraction_function(request):
    """Execute an extraction function on the CSV data."""
    try:
        data = json.loads(request.body) if request.body else {}
        function_id = data.get('function_id')

        if not function_id:
            return JsonResponse({
                'success': False,
                'error': 'No function ID provided'
            }, status=400)

        # Load the extraction function
        functions = load_extraction_functions()

        if str(function_id) not in functions:
            return JsonResponse({
                'success': False,
                'error': f'Function {function_id} not found'
            }, status=404)

        function_data = functions[str(function_id)]

        # Find the CSV file
        large_data_dir = PROJECT_ROOT / "large_data"
        if not large_data_dir.exists():
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_files = list(large_data_dir.glob("*.csv"))
        if not csv_files:
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_file = csv_files[0]
        csv_filename = csv_file.stem

        # Read all CSV entries
        all_entries = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_entries.append(dict(row))

        # Execute the extraction function
        try:
            # Capture print output
            import io
            import sys

            print_output = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = print_output

            try:
                # Create a safe execution environment
                exec_globals = {
                    'all_entries': all_entries,
                    'json': json,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'filter': filter,
                    'map': map,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'any': any,
                    'all': all,
                    'sorted': sorted,
                }
                exec_locals = {}

                exec(function_data['code'], exec_globals, exec_locals)

                if 'result' not in exec_locals:
                    return JsonResponse({
                        'success': False,
                        'error': 'Extraction function must set a "result" variable',
                        'console_output': print_output.getvalue()
                    }, status=400)

                result = exec_locals['result']

            finally:
                # Restore stdout
                sys.stdout = old_stdout

            # Get the captured output
            console_output = print_output.getvalue()

        except Exception as e:
            # Restore stdout in case of exception
            sys.stdout = old_stdout
            return JsonResponse({
                'success': False,
                'error': f'Error executing function: {str(e)}',
                'console_output': print_output.getvalue() if 'print_output' in locals() else ''
            }, status=500)

        # Create filename: fn<number>_<date>_<csv_name>.json
        current_date = datetime.now().strftime('%Y%m%d')
        json_filename = f"fn{function_id}_{current_date}_{csv_filename}.json"

        # Save to Context folder
        context_dir = PROJECT_ROOT / "Context"
        context_dir.mkdir(exist_ok=True)

        json_file_path = context_dir / json_filename
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': f'Extraction complete: {json_filename}',
            'filename': json_filename,
            'result_count': len(result) if isinstance(result, list) else 1,
            'console_output': console_output
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to execute extraction: {str(e)}'
        }, status=500)




@csrf_exempt
@require_http_methods(["POST"])
def generate_with_prompt(request):
    """Generate a response with a custom user prompt and report text."""
    try:
        data = json.loads(request.body) if request.body else {}
        user_prompt = data.get('user_prompt', '')
        report_text = data.get('report_text', '')
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 50))

        if not user_prompt:
            return JsonResponse({'success': False, 'error': 'user_prompt is required'}, status=400)

        if not report_text:
            return JsonResponse({'success': False, 'error': 'report_text is required'}, status=400)

        llm_service = LlamaService()

        if not llm_service.is_server_running():
            return JsonResponse({'success': False, 'error': 'LLM server is not running'}, status=503)

        # Build messages with report text and prompt
        messages = []

        user_text = "Report: " + report_text + "\n\n" + user_prompt

        messages.append({
            'role': 'user',
            'content': user_text
        })

        # Call LLM
        response = requests.post(
            f"{llm_service.base_url}/v1/chat/completions",
            headers={'Content-Type': 'application/json'},
            json={
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            },
            timeout=60
        )

        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': f'LLM request failed: {response.text}'}, status=500)

        result = response.json()
        generated_text = result['choices'][0]['message']['content']

        return JsonResponse({'success': True, 'response': generated_text})

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to generate response: {str(e)}'}, status=500)


# Entity-Relation Management Endpoints

@require_http_methods(["GET"])
def get_entities(request):
    """Get all entity-relation data as array of entry objects with triplets."""
    try:
        entities_file = PROJECT_ROOT / "entities.json"

        # Get CSV metadata to know how many entries we should have
        csv_metadata = get_csv_metadata()
        row_count = csv_metadata['row_count'] if csv_metadata else 0

        if not entities_file.exists():
            # Initialize entities file with entries for all CSV rows
            data = [{
                "entry": i,
                "active_schema": "radgraph",
                "ground_truths": {
                    "radgraph": {"triplets": []},
                    "pet_ct_oncology": {"triplets": []}
                },
                "extraction_methods": {
                    "radgraph": [],
                    "pet_ct_oncology": []
                }
            } for i in range(1, row_count + 1)]

            # Save the initialized data
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return JsonResponse({
                'success': True,
                'data': data
            })

        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Sync entries with CSV row count
        existing_entries = {entry.get('entry') for entry in data if 'entry' in entry}

        # Add missing entries
        for i in range(1, row_count + 1):
            if i not in existing_entries:
                data.append({
                    "entry": i,
                    "active_schema": "radgraph",
                    "ground_truths": {
                        "radgraph": {"triplets": []},
                        "pet_ct_oncology": {"triplets": []}
                    },
                    "extraction_methods": {
                        "radgraph": [],
                        "pet_ct_oncology": []
                    }
                })

        # Remove entries beyond CSV row count
        data = [entry for entry in data if entry.get('entry', 0) <= row_count]

        # Sort by entry number
        data.sort(key=lambda x: x.get('entry', 0))

        # Ensure all entries have required structure
        for entry in data:
            if 'active_schema' not in entry:
                entry['active_schema'] = 'radgraph'
            if 'ground_truths' not in entry:
                entry['ground_truths'] = {}
            if 'extraction_methods' not in entry:
                entry['extraction_methods'] = {}
            # Ensure common schemas exist
            for schema in ['radgraph', 'pet_ct_oncology']:
                if schema not in entry['ground_truths']:
                    entry['ground_truths'][schema] = {"triplets": []}
                if schema not in entry['extraction_methods']:
                    entry['extraction_methods'][schema] = []

        # Save synced data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'data': data
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load entities: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_entities(request):
    """Save all entity-relation data as array of entry objects with triplets."""
    try:
        data = json.loads(request.body) if request.body else []

        # Validate that data is an array
        if not isinstance(data, list):
            return JsonResponse({
                'success': False,
                'error': 'Entities data must be an array'
            }, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Entities saved successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save entities: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def extract_entities_relations(request):
    """Extract entities and relations from a report using the active schema and prompt."""
    try:
        data = json.loads(request.body) if request.body else {}
        report_text = data.get('report_text', '')
        entry_number = data.get('entry_number', None)

        if not report_text:
            return JsonResponse({'success': False, 'error': 'report_text is required'}, status=400)

        llm_service = LlamaService()

        if not llm_service.is_server_running():
            return JsonResponse({'success': False, 'error': 'LLM server is not running'}, status=503)

        # Load active schema from schemas.json
        schemas_file = PROJECT_ROOT / "schemas.json"
        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        active_schema_name = schemas_data.get('active_schema', 'radgraph')
        active_schema = schemas_data['schemas'][active_schema_name]

        # Load active extraction prompt
        prompts_data = load_extraction_prompts()
        active_prompt_name = prompts_data.get('active_prompt', 'default_schema_based')
        active_prompt = prompts_data['prompts'][active_prompt_name]

        # Build system prompt from template and schema
        system_prompt = build_prompt_from_template(active_prompt['template'], active_schema)

        # Build messages with dynamic system prompt
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Extract all entities and relations from this radiology report:\n\n{report_text}"
            }
        ]

        # Call LLM with settings optimized for deterministic extraction
        response = requests.post(
            f"{llm_service.base_url}/v1/chat/completions",
            headers={'Content-Type': 'application/json'},
            json={
                'messages': messages,
                'temperature': 0.1,  # Low temperature for consistency
                'max_tokens': 8000   # Allow longer responses for complex reports
            },
            timeout=120  # Longer timeout for complex extraction
        )

        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': f'LLM request failed: {response.text}'}, status=500)

        result = response.json()
        generated_text = result['choices'][0]['message']['content']

        # Parse JSON response
        try:
            triplets, parse_error = extract_json_array(generated_text)
            if triplets is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Failed to parse LLM response: {parse_error}',
                    'raw_response': generated_text
                }, status=500)

            # Validate triplet structure
            if not isinstance(triplets, list):
                return JsonResponse({
                    'success': False,
                    'error': 'LLM response is not a valid array of triplets',
                    'raw_response': generated_text
                }, status=500)

            # Validate each triplet has required fields
            required_fields = ['entity1_text', 'entity1_type', 'relation_type', 'entity2_text', 'entity2_type']
            for i, triplet in enumerate(triplets):
                if not isinstance(triplet, dict):
                    return JsonResponse({
                        'success': False,
                        'error': f'Triplet {i} is not a valid object',
                        'raw_response': generated_text
                    }, status=500)
                for field in required_fields:
                    if field not in triplet:
                        return JsonResponse({
                            'success': False,
                            'error': f'Triplet {i} missing required field: {field}',
                            'raw_response': generated_text
                        }, status=500)

            # Note: We no longer auto-save here. The frontend will handle
            # saving via the create_extraction_method or run_extraction_method endpoints

            return JsonResponse({
                'success': True,
                'triplets': triplets,
                'count': len(triplets)
            })

        except json.JSONDecodeError as e:
            return JsonResponse({
                'success': False,
                'error': f'Failed to parse LLM response as JSON: {str(e)}',
                'raw_response': generated_text
            }, status=500)

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to extract entities: {str(e)}'}, status=500)


# Ground Truth and Extraction Methods Endpoints

@csrf_exempt
@require_http_methods(["POST"])
def add_ground_truth_triplet(request):
    """Add a ground truth triplet to a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        triplet = data.get('triplet')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not triplet:
            return JsonResponse({'success': False, 'error': 'triplet is required'}, status=400)

        # Validate triplet structure
        required_fields = ['entity1_text', 'entity1_type', 'relation_type', 'entity2_text', 'entity2_type']
        for field in required_fields:
            if field not in triplet:
                return JsonResponse({'success': False, 'error': f'Triplet missing required field: {field}'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                active_schema = entry.get('active_schema', 'radgraph')
                if 'ground_truths' not in entry:
                    entry['ground_truths'] = {}
                if active_schema not in entry['ground_truths']:
                    entry['ground_truths'][active_schema] = {"triplets": []}
                entry['ground_truths'][active_schema]['triplets'].append(triplet)
                entry_found = True
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Ground truth triplet added successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to add ground truth triplet: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_ground_truth_triplet(request):
    """Delete a ground truth triplet from a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        triplet_index = data.get('triplet_index')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if triplet_index is None:
            return JsonResponse({'success': False, 'error': 'triplet_index is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                active_schema = entry.get('active_schema', 'radgraph')
                if 'ground_truths' in entry and active_schema in entry['ground_truths'] and 'triplets' in entry['ground_truths'][active_schema]:
                    if 0 <= triplet_index < len(entry['ground_truths'][active_schema]['triplets']):
                        del entry['ground_truths'][active_schema]['triplets'][triplet_index]
                        entry_found = True
                    else:
                        return JsonResponse({'success': False, 'error': 'Invalid triplet_index'}, status=400)
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found or has no ground truth'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Ground truth triplet deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to delete ground truth triplet: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def add_ground_truth_entity(request):
    """Add a ground truth entity to a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        entity = data.get('entity')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not entity:
            return JsonResponse({'success': False, 'error': 'entity is required'}, status=400)

        # Validate entity structure
        required_fields = ['text', 'type']
        for field in required_fields:
            if field not in entity:
                return JsonResponse({'success': False, 'error': f'Entity missing required field: {field}'}, status=400)

        # Handle optional word indices
        if 'start_word' in entity:
            try:
                entity['start_word'] = int(entity['start_word'])
                if entity['start_word'] < 0:
                    entity['start_word'] = None
            except (ValueError, TypeError):
                entity['start_word'] = None
        else:
            entity['start_word'] = None

        if 'end_word' in entity:
            try:
                entity['end_word'] = int(entity['end_word'])
                if entity['end_word'] < 0:
                    entity['end_word'] = None
            except (ValueError, TypeError):
                entity['end_word'] = None
        else:
            entity['end_word'] = None

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        new_entity_id = None
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                active_schema = entry.get('active_schema', 'radgraph')
                if 'ground_truths' not in entry:
                    entry['ground_truths'] = {}
                if active_schema not in entry['ground_truths']:
                    entry['ground_truths'][active_schema] = {"entities": [], "triplets": []}
                if 'entities' not in entry['ground_truths'][active_schema]:
                    entry['ground_truths'][active_schema]['entities'] = []

                # Generate entity ID with consistent scheme: gt_e{n}
                existing_ids = [e.get('id', '') for e in entry['ground_truths'][active_schema]['entities']]
                entity_num = 1
                while f"gt_e{entity_num}" in existing_ids:
                    entity_num += 1
                new_entity_id = f"gt_e{entity_num}"

                entity['id'] = new_entity_id
                entry['ground_truths'][active_schema]['entities'].append(entity)
                entry_found = True
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Ground truth entity added successfully',
            'entity_id': new_entity_id
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to add ground truth entity: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_ground_truth_entity(request):
    """Delete a ground truth entity from a specific entry (also removes related triplets)."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        entity_id = data.get('entity_id')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not entity_id:
            return JsonResponse({'success': False, 'error': 'entity_id is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                active_schema = entry.get('active_schema', 'radgraph')
                if 'ground_truths' in entry and active_schema in entry['ground_truths']:
                    gt = entry['ground_truths'][active_schema]
                    if 'entities' in gt:
                        # Find and remove the entity
                        original_len = len(gt['entities'])
                        gt['entities'] = [e for e in gt['entities'] if e.get('id') != entity_id]
                        if len(gt['entities']) < original_len:
                            entry_found = True
                            # Also remove any triplets referencing this entity
                            if 'triplets' in gt:
                                gt['triplets'] = [
                                    t for t in gt['triplets']
                                    if t.get('entity1_id') != entity_id and t.get('entity2_id') != entity_id
                                ]
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entity {entity_id} not found'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Ground truth entity deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to delete ground truth entity: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def add_ground_truth_relation(request):
    """Add a ground truth relation (triplet using entity IDs) to a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        relation = data.get('relation')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not relation:
            return JsonResponse({'success': False, 'error': 'relation is required'}, status=400)

        # Validate relation structure
        required_fields = ['entity1_id', 'relation_type', 'entity2_id']
        for field in required_fields:
            if field not in relation:
                return JsonResponse({'success': False, 'error': f'Relation missing required field: {field}'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                active_schema = entry.get('active_schema', 'radgraph')
                if 'ground_truths' not in entry:
                    entry['ground_truths'] = {}
                if active_schema not in entry['ground_truths']:
                    entry['ground_truths'][active_schema] = {"entities": [], "triplets": []}
                if 'triplets' not in entry['ground_truths'][active_schema]:
                    entry['ground_truths'][active_schema]['triplets'] = []

                # Verify entity IDs exist
                entity_ids = {e.get('id') for e in entry['ground_truths'][active_schema].get('entities', [])}
                if relation['entity1_id'] not in entity_ids:
                    return JsonResponse({'success': False, 'error': f"Entity {relation['entity1_id']} not found"}, status=400)
                if relation['entity2_id'] not in entity_ids:
                    return JsonResponse({'success': False, 'error': f"Entity {relation['entity2_id']} not found"}, status=400)

                # Generate triplet ID with consistent scheme: gt_r{n}
                existing_triplet_ids = [t.get('id', '') for t in entry['ground_truths'][active_schema]['triplets']]
                triplet_num = 1
                while f"gt_r{triplet_num}" in existing_triplet_ids:
                    triplet_num += 1
                relation['id'] = f"gt_r{triplet_num}"

                entry['ground_truths'][active_schema]['triplets'].append(relation)
                entry_found = True
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Ground truth relation added successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to add ground truth relation: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_ground_truth_relation(request):
    """Delete a ground truth relation from a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        relation_index = data.get('relation_index')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if relation_index is None:
            return JsonResponse({'success': False, 'error': 'relation_index is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                active_schema = entry.get('active_schema', 'radgraph')
                if 'ground_truths' in entry and active_schema in entry['ground_truths'] and 'triplets' in entry['ground_truths'][active_schema]:
                    if 0 <= relation_index < len(entry['ground_truths'][active_schema]['triplets']):
                        del entry['ground_truths'][active_schema]['triplets'][relation_index]
                        entry_found = True
                    else:
                        return JsonResponse({'success': False, 'error': 'Invalid relation_index'}, status=400)
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found or has no ground truth'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Ground truth relation deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to delete ground truth relation: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def extract_entities_llm(request):
    """Extract entities only using LLM (first step of two-step extraction)."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_id = data.get('method_id')
        report_text = data.get('report_text', '')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_id:
            return JsonResponse({'success': False, 'error': 'method_id is required'}, status=400)

        if not report_text:
            return JsonResponse({'success': False, 'error': 'report_text is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        # Load entities
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find the entry and method
        target_entry = None
        target_method = None
        method_index = 0

        for entry in entities_data:
            if entry.get('entry') == entry_number:
                target_entry = entry
                if 'extraction_methods' in entry:
                    for schema_name, methods in entry['extraction_methods'].items():
                        for idx, method in enumerate(methods):
                            if method.get('id') == method_id:
                                target_method = method
                                method_index = idx + 1  # 1-based index
                                break
                        if target_method:
                            break
                break

        if not target_entry:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        if not target_method:
            return JsonResponse({'success': False, 'error': f'Method {method_id} not found'}, status=404)

        # Get schema
        schema_name = target_method.get('schema', 'radgraph')

        # Load schema
        schemas_file = PROJECT_ROOT / "schemas.json"
        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        if schema_name not in schemas_data['schemas']:
            return JsonResponse({'success': False, 'error': f'Schema {schema_name} not found'}, status=404)

        active_schema = schemas_data['schemas'][schema_name]

        # Load entity extraction prompt from extraction_prompts.json
        prompts_file = PROJECT_ROOT / "extraction_prompts.json"
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

        # Get active entity extraction prompt
        entity_prompts = prompts_data.get('entity_extraction', {})
        active_prompt_key = entity_prompts.get('active_prompt', 'default_entity')
        prompt_config = entity_prompts.get('prompts', {}).get(active_prompt_key, {})
        prompt_template = prompt_config.get('template', '')

        # Build entity types text - include ALL entity types
        entity_types_text = ""
        for i, entity_type in enumerate(active_schema['entity_types'], 1):
            entity_types_text += f"{i}. **{entity_type['name']}**: {entity_type['description']}\n"

        # Create indexed report text for word-based indexing
        indexed_report, word_list = create_indexed_report_text(report_text)

        # Check if prompt uses indexed_report (new style) or not (legacy)
        uses_word_indexing = '{indexed_report}' in prompt_template

        # Format the prompt template with schema information
        format_args = {
            'schema_name': active_schema['name'],
            'entity_types': entity_types_text,
            'entity_count': len(active_schema['entity_types'])
        }
        if uses_word_indexing:
            format_args['indexed_report'] = indexed_report

        system_prompt = prompt_template.format(**format_args)

        # Check if LLM server is running
        llm_service = LlamaService()
        if not llm_service.is_server_running():
            return JsonResponse({'success': False, 'error': 'LLM server is not running'}, status=503)

        # Build messages - use indexed report if word indexing is enabled
        user_content = f"Extract all entities from this radiology report:\n\n{indexed_report if uses_word_indexing else report_text}"
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_content
            }
        ]

        # Call LLM
        response = requests.post(
            f"{llm_service.base_url}/v1/chat/completions",
            headers={'Content-Type': 'application/json'},
            json={
                'messages': messages,
                'temperature': 0.1,
                'max_tokens': 8000
            },
            timeout=120
        )

        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': f'LLM request failed: {response.text}'}, status=500)

        result = response.json()
        generated_text = result['choices'][0]['message']['content']

        # Parse JSON response
        try:
            entities, parse_error = extract_json_array(generated_text)
            if entities is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Failed to parse LLM response: {parse_error}',
                    'raw_response': generated_text
                }, status=500)

            # Validate entity structure
            if not isinstance(entities, list):
                return JsonResponse({
                    'success': False,
                    'error': 'LLM response is not a valid array of entities',
                    'raw_response': generated_text
                }, status=500)

            # Validate each entity and ensure IDs are unique
            required_fields = ['text', 'type']
            seen_ids = set()
            max_word_index = len(word_list) - 1 if word_list else -1

            for i, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    return JsonResponse({
                        'success': False,
                        'error': f'Entity {i} is not a valid object',
                        'raw_response': generated_text
                    }, status=500)
                for field in required_fields:
                    if field not in entity:
                        return JsonResponse({
                            'success': False,
                            'error': f'Entity {i} missing required field: {field}',
                            'raw_response': generated_text
                        }, status=500)
                # Ensure ID exists and is unique with consistent scheme: auto{method_index}_e{n}
                if 'id' not in entity:
                    entity['id'] = f"auto{method_index}_e{i+1}"
                if entity['id'] in seen_ids:
                    entity['id'] = f"auto{method_index}_e{len(seen_ids)+1}"
                seen_ids.add(entity['id'])

                # Handle word indices (optional, defaults to None if not present)
                if 'start_word' in entity:
                    try:
                        entity['start_word'] = int(entity['start_word'])
                        # Validate range
                        if entity['start_word'] < 0 or (max_word_index >= 0 and entity['start_word'] > max_word_index):
                            entity['start_word'] = None
                    except (ValueError, TypeError):
                        entity['start_word'] = None
                else:
                    entity['start_word'] = None

                if 'end_word' in entity:
                    try:
                        entity['end_word'] = int(entity['end_word'])
                        # Validate range
                        if entity['end_word'] < 0 or (max_word_index >= 0 and entity['end_word'] > max_word_index):
                            entity['end_word'] = None
                    except (ValueError, TypeError):
                        entity['end_word'] = None
                else:
                    entity['end_word'] = None

            # Save entities to method
            target_method['entities'] = entities
            target_method['triplets'] = []  # Clear triplets since we're re-extracting
            target_method['timestamp'] = datetime.now().isoformat()

            # Save updated data
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)

            return JsonResponse({
                'success': True,
                'entities': entities,
                'count': len(entities)
            })

        except json.JSONDecodeError as e:
            return JsonResponse({
                'success': False,
                'error': f'Failed to parse LLM response as JSON: {str(e)}',
                'raw_response': generated_text
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        import traceback
        return JsonResponse({
            'success': False,
            'error': f'Failed to extract entities: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def extract_relations_llm(request):
    """Extract relations using LLM given existing entities (second step of two-step extraction)."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_id = data.get('method_id')
        report_text = data.get('report_text', '')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_id:
            return JsonResponse({'success': False, 'error': 'method_id is required'}, status=400)

        if not report_text:
            return JsonResponse({'success': False, 'error': 'report_text is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        # Load entities
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find the entry and method
        target_entry = None
        target_method = None
        method_index = 0

        for entry in entities_data:
            if entry.get('entry') == entry_number:
                target_entry = entry
                if 'extraction_methods' in entry:
                    for schema_name, methods in entry['extraction_methods'].items():
                        for idx, method in enumerate(methods):
                            if method.get('id') == method_id:
                                target_method = method
                                method_index = idx + 1  # 1-based index
                                break
                        if target_method:
                            break
                break

        if not target_entry:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        if not target_method:
            return JsonResponse({'success': False, 'error': f'Method {method_id} not found'}, status=404)

        # Check if entities exist
        existing_entities = target_method.get('entities', [])
        if not existing_entities:
            return JsonResponse({'success': False, 'error': 'No entities found. Please extract entities first.'}, status=400)

        # Get schema
        schema_name = target_method.get('schema', 'radgraph')

        # Load schema
        schemas_file = PROJECT_ROOT / "schemas.json"
        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        if schema_name not in schemas_data['schemas']:
            return JsonResponse({'success': False, 'error': f'Schema {schema_name} not found'}, status=404)

        active_schema = schemas_data['schemas'][schema_name]

        # Load relation extraction prompt from extraction_prompts.json
        prompts_file = PROJECT_ROOT / "extraction_prompts.json"
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

        # Get active relation extraction prompt
        relation_prompts = prompts_data.get('relation_extraction', {})
        active_prompt_key = relation_prompts.get('active_prompt', 'default_relation')
        prompt_config = relation_prompts.get('prompts', {}).get(active_prompt_key, {})
        prompt_template = prompt_config.get('template', '')

        # Build relation types text
        relation_types_text = ""
        for i, relation_type in enumerate(active_schema['relation_types'], 1):
            relation_types_text += f"{i}. **{relation_type['name']}**: {relation_type['description']}\n"
            if relation_type.get('valid_pairs'):
                pairs_str = ", ".join([f"({pair[0]} → {pair[1]})" for pair in relation_type['valid_pairs'][:3]])
                if len(relation_type['valid_pairs']) > 3:
                    pairs_str += f" (and {len(relation_type['valid_pairs']) - 3} more)"
                relation_types_text += f"   Valid pairs: {pairs_str}\n"

        # Format entities for the prompt
        entities_list = ""
        for entity in existing_entities:
            entities_list += f"- {entity['id']}: \"{entity['text']}\" (Type: {entity['type']})\n"

        # Format the prompt template with schema and entities information
        system_prompt = prompt_template.format(
            schema_name=active_schema['name'],
            entities_list=entities_list,
            relation_types=relation_types_text,
            relation_count=len(active_schema['relation_types']),
            report_text=report_text
        )

        # Check if LLM server is running
        llm_service = LlamaService()
        if not llm_service.is_server_running():
            return JsonResponse({'success': False, 'error': 'LLM server is not running'}, status=503)

        # Build messages
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Identify all relations between the entities in this radiology report:\n\n{report_text}"
            }
        ]

        # Call LLM
        response = requests.post(
            f"{llm_service.base_url}/v1/chat/completions",
            headers={'Content-Type': 'application/json'},
            json={
                'messages': messages,
                'temperature': 0.1,
                'max_tokens': 4000
            },
            timeout=180
        )

        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': f'LLM request failed: {response.text}'}, status=500)

        result = response.json()
        generated_text = result['choices'][0]['message']['content']

        # Check for empty response
        if not generated_text or not generated_text.strip():
            return JsonResponse({
                'success': False,
                'error': 'LLM returned empty response. The model may have timed out or the prompt may be too long.',
                'raw_response': generated_text
            }, status=500)

        # Parse JSON response
        try:
            # Handle empty array case
            if generated_text.strip() == '[]':
                relations = []
            else:
                relations, parse_error = extract_json_array(generated_text)
                if relations is None:
                    return JsonResponse({
                        'success': False,
                        'error': f'Failed to parse LLM response: {parse_error}',
                        'raw_response': generated_text[:1000]
                    }, status=500)

            # Validate relation structure
            if not isinstance(relations, list):
                return JsonResponse({
                    'success': False,
                    'error': 'LLM response is not a valid array of relations',
                    'raw_response': generated_text[:1000]
                }, status=500)

            # Create entity ID lookup
            valid_entity_ids = {e['id'] for e in existing_entities}

            # Validate each relation
            required_fields = ['entity1_id', 'relation_type', 'entity2_id']
            valid_relations = []
            for i, relation in enumerate(relations):
                if not isinstance(relation, dict):
                    continue
                # Check required fields
                if not all(field in relation for field in required_fields):
                    continue
                # Check entity IDs are valid
                if relation['entity1_id'] not in valid_entity_ids:
                    continue
                if relation['entity2_id'] not in valid_entity_ids:
                    continue
                valid_relations.append(relation)

            # Assign triplet IDs with consistent scheme: auto{method_index}_r{n}
            for i, triplet in enumerate(valid_relations):
                if 'id' not in triplet:
                    triplet['id'] = f"auto{method_index}_r{i+1}"

            # Save relations to method
            target_method['triplets'] = valid_relations
            target_method['timestamp'] = datetime.now().isoformat()

            # Save updated data
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)

            return JsonResponse({
                'success': True,
                'relations': valid_relations,
                'count': len(valid_relations)
            })

        except json.JSONDecodeError as e:
            # Show context around error
            error_pos = e.pos if hasattr(e, 'pos') and e.pos else 0
            error_context = generated_text[max(0, error_pos-100):error_pos+100]
            return JsonResponse({
                'success': False,
                'error': f'Failed to parse LLM response as JSON: {str(e)}',
                'error_context': f'...{error_context}...',
                'raw_response_length': len(generated_text),
                'suggestion': 'The response may be truncated or malformed. Try with fewer entities.'
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to extract relations: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def extract_oneshot(request):
    """Extract both entities and relations in a single LLM call (one-shot extraction)."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_id = data.get('method_id')
        report_text = data.get('report_text', '')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_id:
            return JsonResponse({'success': False, 'error': 'method_id is required'}, status=400)

        if not report_text:
            return JsonResponse({'success': False, 'error': 'report_text is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        # Load entities
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find the entry and method
        target_entry = None
        target_method = None
        method_index = 0

        for entry in entities_data:
            if entry.get('entry') == entry_number:
                target_entry = entry
                if 'extraction_methods' in entry:
                    for schema_name, methods in entry['extraction_methods'].items():
                        for idx, method in enumerate(methods):
                            if method.get('id') == method_id:
                                target_method = method
                                method_index = idx + 1  # 1-based index
                                break
                        if target_method:
                            break
                break

        if not target_entry:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        if not target_method:
            return JsonResponse({'success': False, 'error': f'Method {method_id} not found'}, status=404)

        # Get schema
        schema_name = target_method.get('schema', 'radgraph')

        # Load schema
        schemas_file = PROJECT_ROOT / "schemas.json"
        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        if schema_name not in schemas_data['schemas']:
            return JsonResponse({'success': False, 'error': f'Schema {schema_name} not found'}, status=404)

        active_schema = schemas_data['schemas'][schema_name]

        # Build entity types text
        entity_types_text = ""
        for i, entity_type in enumerate(active_schema['entity_types'], 1):
            entity_types_text += f"{i}. **{entity_type['name']}**: {entity_type['description']}\n"

        # Build relation types text
        relation_types_text = ""
        for i, relation_type in enumerate(active_schema['relation_types'], 1):
            relation_types_text += f"{i}. **{relation_type['name']}**: {relation_type['description']}\n"
            if relation_type.get('valid_pairs'):
                pairs_str = ", ".join([f"({pair[0]} → {pair[1]})" for pair in relation_type['valid_pairs'][:3]])
                if len(relation_type['valid_pairs']) > 3:
                    pairs_str += f" (and {len(relation_type['valid_pairs']) - 3} more)"
                relation_types_text += f"   Valid pairs: {pairs_str}\n"

        # Create one-shot prompt
        system_prompt = f"""You are an expert medical entity and relation extraction system using the {active_schema['name']} schema.

Your task is to extract BOTH entities and relations from radiology reports in a single response.

## Entity Types
{entity_types_text}

## Relation Types
{relation_types_text}

## Output Format
Respond with a JSON object containing two arrays:
{{
  "entities": [
    {{"text": "entity text", "type": "entity_type"}},
    ...
  ],
  "relations": [
    {{"entity1_id": "a{method_index}e1", "relation_type": "relation_name", "entity2_id": "a{method_index}e2"}},
    ...
  ]
}}

Important:
- Use entity IDs like "a{method_index}e1", "a{method_index}e2", "a{method_index}e3" etc. for entities
- In relations, reference entities by their IDs (e.g., "a{method_index}e1", "a{method_index}e2")
- Only extract entities and relations that are clearly stated in the report
- Ensure all entity IDs in relations exist in the entities array
"""

        # Check if LLM server is running
        llm_service = LlamaService()
        if not llm_service.is_server_running():
            return JsonResponse({'success': False, 'error': 'LLM server is not running'}, status=503)

        # Build messages
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Extract all entities and relations from this radiology report:\n\n{report_text}"
            }
        ]

        # Call LLM
        response = requests.post(
            f"{llm_service.base_url}/v1/chat/completions",
            headers={'Content-Type': 'application/json'},
            json={
                'messages': messages,
                'temperature': 0.1,
                'max_tokens': 4000
            },
            timeout=180
        )

        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': f'LLM request failed: {response.text}'}, status=500)

        result = response.json()
        generated_text = result['choices'][0]['message']['content']

        # Parse JSON response
        try:
            extraction_result, parse_error = extract_json_object(generated_text)
            if extraction_result is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Failed to parse LLM response: {parse_error}',
                    'raw_response': generated_text[:1000]
                }, status=500)

            # Validate structure
            if not isinstance(extraction_result, dict):
                return JsonResponse({
                    'success': False,
                    'error': 'LLM response is not a valid JSON object',
                    'raw_response': generated_text[:1000]
                }, status=500)

            if 'entities' not in extraction_result or 'relations' not in extraction_result:
                return JsonResponse({
                    'success': False,
                    'error': 'LLM response missing entities or relations',
                    'raw_response': generated_text[:1000]
                }, status=500)

            entities = extraction_result['entities']
            relations = extraction_result['relations']

            # Validate and assign entity IDs
            required_entity_fields = ['text', 'type']
            seen_ids = set()
            for i, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    continue
                if not all(field in entity for field in required_entity_fields):
                    continue
                # Ensure ID exists and is unique with consistent scheme: auto{method_index}_e{n}
                if 'id' not in entity:
                    entity['id'] = f"auto{method_index}_e{i+1}"
                if entity['id'] in seen_ids:
                    entity['id'] = f"auto{method_index}_e{len(seen_ids)+1}"
                seen_ids.add(entity['id'])

            # Validate relations
            valid_entity_ids = {e['id'] for e in entities if 'id' in e}
            valid_relations = []
            required_relation_fields = ['entity1_id', 'relation_type', 'entity2_id']

            for i, relation in enumerate(relations):
                if not isinstance(relation, dict):
                    continue
                if not all(field in relation for field in required_relation_fields):
                    continue
                # Check entity IDs are valid
                if relation['entity1_id'] not in valid_entity_ids:
                    continue
                if relation['entity2_id'] not in valid_entity_ids:
                    continue

                # Assign triplet ID with consistent scheme: auto{method_index}_r{n}
                if 'id' not in relation:
                    relation['id'] = f"auto{method_index}_r{i+1}"

                valid_relations.append(relation)

            # Save to method
            target_method['entities'] = entities
            target_method['triplets'] = valid_relations
            target_method['timestamp'] = datetime.now().isoformat()

            # Save updated data
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)

            return JsonResponse({
                'success': True,
                'entities': entities,
                'relations': valid_relations,
                'entity_count': len(entities),
                'relation_count': len(valid_relations)
            })

        except json.JSONDecodeError as e:
            return JsonResponse({
                'success': False,
                'error': f'Failed to parse LLM response as JSON: {str(e)}',
                'raw_response': generated_text[:1000]
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to extract: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def create_extraction_method(request):
    """Create a new extraction method for a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_name = data.get('method_name')
        schema = data.get('schema', 'radgraph')
        prompt = data.get('prompt', 'default_schema_based')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_name:
            return JsonResponse({'success': False, 'error': 'method_name is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                if 'extraction_methods' not in entry:
                    entry['extraction_methods'] = {}
                if schema not in entry['extraction_methods']:
                    entry['extraction_methods'][schema] = []

                # Generate unique method ID
                method_id = f"method_{entry_number}_{schema}_{len(entry['extraction_methods'][schema]) + 1}"

                # Create new method
                new_method = {
                    "id": method_id,
                    "name": method_name,
                    "schema": schema,
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "triplets": []
                }

                entry['extraction_methods'][schema].append(new_method)
                entry_found = True

                # Save updated data
                with open(entities_file, 'w', encoding='utf-8') as f:
                    json.dump(entities_data, f, indent=2, ensure_ascii=False)

                return JsonResponse({
                    'success': True,
                    'message': 'Extraction method created successfully',
                    'method_id': method_id
                })

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to create extraction method: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_extraction_method(request):
    """Run an extraction method (calls LLM and saves results)."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_id = data.get('method_id')
        report_text = data.get('report_text', '')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_id:
            return JsonResponse({'success': False, 'error': 'method_id is required'}, status=400)

        if not report_text:
            return JsonResponse({'success': False, 'error': 'report_text is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        # Load entities
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find the entry and method
        target_entry = None
        target_method = None
        method_index = 0

        for entry in entities_data:
            if entry.get('entry') == entry_number:
                target_entry = entry
                if 'extraction_methods' in entry:
                    # Search across all schemas
                    for schema_name, methods in entry['extraction_methods'].items():
                        for idx, method in enumerate(methods):
                            if method.get('id') == method_id:
                                target_method = method
                                method_index = idx + 1  # 1-based index
                                break
                        if target_method:
                            break
                break

        if not target_entry:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        if not target_method:
            return JsonResponse({'success': False, 'error': f'Method {method_id} not found'}, status=404)

        # Get schema and prompt from method
        schema_name = target_method.get('schema', 'radgraph')
        prompt_name = target_method.get('prompt', 'default_schema_based')

        # Load schema
        schemas_file = PROJECT_ROOT / "schemas.json"
        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        if schema_name not in schemas_data['schemas']:
            return JsonResponse({'success': False, 'error': f'Schema {schema_name} not found'}, status=404)

        active_schema = schemas_data['schemas'][schema_name]

        # Load prompt
        prompts_data = load_extraction_prompts()
        if prompt_name not in prompts_data['prompts']:
            return JsonResponse({'success': False, 'error': f'Prompt {prompt_name} not found'}, status=404)

        active_prompt = prompts_data['prompts'][prompt_name]

        # Build system prompt from template and schema
        system_prompt = build_prompt_from_template(active_prompt['template'], active_schema)

        # Check if LLM server is running
        llm_service = LlamaService()
        if not llm_service.is_server_running():
            return JsonResponse({'success': False, 'error': 'LLM server is not running'}, status=503)

        # Build messages
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Extract all entities and relations from this radiology report:\n\n{report_text}"
            }
        ]

        # Call LLM
        response = requests.post(
            f"{llm_service.base_url}/v1/chat/completions",
            headers={'Content-Type': 'application/json'},
            json={
                'messages': messages,
                'temperature': 0.1,
                'max_tokens': 8000
            },
            timeout=120
        )

        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': f'LLM request failed: {response.text}'}, status=500)

        result = response.json()
        generated_text = result['choices'][0]['message']['content']

        # Parse JSON response
        try:
            triplets, parse_error = extract_json_array(generated_text)
            if triplets is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Failed to parse LLM response: {parse_error}',
                    'raw_response': generated_text
                }, status=500)

            # Validate triplet structure
            if not isinstance(triplets, list):
                return JsonResponse({
                    'success': False,
                    'error': 'LLM response is not a valid array of triplets',
                    'raw_response': generated_text
                }, status=500)

            # Validate each triplet
            required_fields = ['entity1_text', 'entity1_type', 'relation_type', 'entity2_text', 'entity2_type']
            for i, triplet in enumerate(triplets):
                if not isinstance(triplet, dict):
                    return JsonResponse({
                        'success': False,
                        'error': f'Triplet {i} is not a valid object',
                        'raw_response': generated_text
                    }, status=500)
                for field in required_fields:
                    if field not in triplet:
                        return JsonResponse({
                            'success': False,
                            'error': f'Triplet {i} missing required field: {field}',
                            'raw_response': generated_text
                        }, status=500)

            # Assign triplet IDs with consistent scheme: auto{method_index}_r{n}
            for i, triplet in enumerate(triplets):
                if 'id' not in triplet:
                    triplet['id'] = f"auto{method_index}_r{i+1}"

            # Save triplets to method
            target_method['triplets'] = triplets
            target_method['timestamp'] = datetime.now().isoformat()

            # Save updated data
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)

            return JsonResponse({
                'success': True,
                'triplets': triplets,
                'count': len(triplets)
            })

        except json.JSONDecodeError as e:
            return JsonResponse({
                'success': False,
                'error': f'Failed to parse LLM response as JSON: {str(e)}',
                'raw_response': generated_text
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to run extraction method: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_extraction_method(request):
    """Delete an extraction method from a specific entry."""
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_id = data.get('method_id')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_id:
            return JsonResponse({'success': False, 'error': 'method_id is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find and update the entry
        entry_found = False
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                if 'extraction_methods' in entry:
                    # Find method across all schemas
                    method_found = False
                    for schema_name, methods in entry['extraction_methods'].items():
                        for i, method in enumerate(methods):
                            if method.get('id') == method_id:
                                del entry['extraction_methods'][schema_name][i]
                                entry_found = True
                                method_found = True
                                break
                        if method_found:
                            break

                    if not method_found:
                        return JsonResponse({'success': False, 'error': f'Method {method_id} not found'}, status=404)
                break

        if not entry_found:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        # Save updated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': 'Extraction method deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to delete extraction method: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def calculate_metrics(request):
    """Calculate evaluation metrics comparing ground truth to an extraction method.

    Returns both entity-level and triplet-level metrics:
    - Entity metrics: precision, recall, F1 based on (text, type) matching
    - Triplet metrics: precision, recall, F1 based on full triplet signature
    """
    try:
        data = json.loads(request.body) if request.body else {}
        entry_number = data.get('entry_number')
        method_id = data.get('method_id')

        if entry_number is None:
            return JsonResponse({'success': False, 'error': 'entry_number is required'}, status=400)

        if not method_id:
            return JsonResponse({'success': False, 'error': 'method_id is required'}, status=400)

        entities_file = PROJECT_ROOT / "entities.json"

        if not entities_file.exists():
            return JsonResponse({'success': False, 'error': 'entities.json not found'}, status=404)

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)

        # Find the entry
        target_entry = None
        for entry in entities_data:
            if entry.get('entry') == entry_number:
                target_entry = entry
                break

        if not target_entry:
            return JsonResponse({'success': False, 'error': f'Entry {entry_number} not found'}, status=404)

        # Find the method to get its schema
        target_method = None
        method_schema = None
        if 'extraction_methods' in target_entry:
            for schema_name, methods in target_entry['extraction_methods'].items():
                for method in methods:
                    if method.get('id') == method_id:
                        target_method = method
                        method_schema = schema_name
                        break
                if target_method:
                    break

        if not target_method:
            return JsonResponse({'success': False, 'error': f'Method {method_id} not found'}, status=404)

        # Get ground truth data for this schema
        gt_data = {}
        if 'ground_truths' in target_entry and method_schema in target_entry['ground_truths']:
            gt_data = target_entry['ground_truths'][method_schema]

        ground_truth_entities = gt_data.get('entities', [])
        ground_truth_triplets = gt_data.get('triplets', [])

        # Get predicted data
        predicted_entities = target_method.get('entities', [])
        predicted_triplets = target_method.get('triplets', [])

        # ============ ENTITY METRICS ============
        # Entity signature: (text, type) - case-insensitive, whitespace-normalized
        def entity_signature(entity):
            return (
                entity.get('text', '').lower().strip(),
                entity.get('type', '').lower().strip()
            )

        # Build signature-to-ID mappings
        gt_entity_map = {entity_signature(e): e.get('id') for e in ground_truth_entities}
        pred_entity_map = {entity_signature(e): e.get('id') for e in predicted_entities}

        gt_entity_set = set(gt_entity_map.keys())
        pred_entity_set = set(pred_entity_map.keys())

        # Track matched/unmatched IDs
        matched_gt_entity_ids = [gt_entity_map[sig] for sig in (gt_entity_set & pred_entity_set)]
        matched_pred_entity_ids = [pred_entity_map[sig] for sig in (gt_entity_set & pred_entity_set)]
        unmatched_gt_entity_ids = [gt_entity_map[sig] for sig in (gt_entity_set - pred_entity_set)]
        unmatched_pred_entity_ids = [pred_entity_map[sig] for sig in (pred_entity_set - gt_entity_set)]

        entity_tp = len(gt_entity_set & pred_entity_set)
        entity_fp = len(pred_entity_set - gt_entity_set)
        entity_fn = len(gt_entity_set - pred_entity_set)

        entity_precision = entity_tp / len(pred_entity_set) if len(pred_entity_set) > 0 else 0.0
        entity_recall = entity_tp / len(gt_entity_set) if len(gt_entity_set) > 0 else 0.0
        entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0

        # ============ TRIPLET METRICS ============
        # Build entity lookup maps for resolving IDs to (text, type)
        def build_entity_lookup(entities):
            return {e.get('id'): (e.get('text', '').lower().strip(), e.get('type', '').lower().strip()) for e in entities}

        gt_entity_lookup = build_entity_lookup(ground_truth_entities)
        pred_entity_lookup = build_entity_lookup(predicted_entities)

        # Triplet signature using new format (entity IDs -> resolved to text+type)
        def triplet_signature_new(triplet, entity_lookup):
            e1_id = triplet.get('entity1_id', '')
            e2_id = triplet.get('entity2_id', '')
            e1 = entity_lookup.get(e1_id, ('', ''))
            e2 = entity_lookup.get(e2_id, ('', ''))
            return (
                e1[0], e1[1],  # entity1 text, type
                triplet.get('relation_type', '').lower().strip(),
                e2[0], e2[1]   # entity2 text, type
            )

        # Legacy triplet signature (old format with inline entity data)
        def triplet_signature_legacy(triplet):
            return (
                triplet.get('entity1_text', '').lower().strip(),
                triplet.get('entity1_type', '').lower().strip(),
                triplet.get('relation_type', '').lower().strip(),
                triplet.get('entity2_text', '').lower().strip(),
                triplet.get('entity2_type', '').lower().strip()
            )

        # Build signature-to-ID mappings for triplets
        gt_triplet_map = {}
        pred_triplet_map = {}

        for t in ground_truth_triplets:
            if 'entity1_id' in t and 'entity2_id' in t:
                sig = triplet_signature_new(t, gt_entity_lookup)
            else:
                sig = triplet_signature_legacy(t)
            gt_triplet_map[sig] = t.get('id', '')

        for t in predicted_triplets:
            if 'entity1_id' in t and 'entity2_id' in t:
                sig = triplet_signature_new(t, pred_entity_lookup)
            else:
                sig = triplet_signature_legacy(t)
            pred_triplet_map[sig] = t.get('id', '')

        gt_triplet_set = set(gt_triplet_map.keys())
        pred_triplet_set = set(pred_triplet_map.keys())

        # Track matched/unmatched triplet IDs
        matched_gt_triplet_ids = [gt_triplet_map[sig] for sig in (gt_triplet_set & pred_triplet_set) if gt_triplet_map[sig]]
        matched_pred_triplet_ids = [pred_triplet_map[sig] for sig in (gt_triplet_set & pred_triplet_set) if pred_triplet_map[sig]]
        unmatched_gt_triplet_ids = [gt_triplet_map[sig] for sig in (gt_triplet_set - pred_triplet_set) if gt_triplet_map[sig]]
        unmatched_pred_triplet_ids = [pred_triplet_map[sig] for sig in (pred_triplet_set - gt_triplet_set) if pred_triplet_map[sig]]

        triplet_tp = len(gt_triplet_set & pred_triplet_set)
        triplet_fp = len(pred_triplet_set - gt_triplet_set)
        triplet_fn = len(gt_triplet_set - pred_triplet_set)

        triplet_precision = triplet_tp / len(pred_triplet_set) if len(pred_triplet_set) > 0 else 0.0
        triplet_recall = triplet_tp / len(gt_triplet_set) if len(gt_triplet_set) > 0 else 0.0
        triplet_f1 = (2 * triplet_precision * triplet_recall) / (triplet_precision + triplet_recall) if (triplet_precision + triplet_recall) > 0 else 0.0

        return JsonResponse({
            'success': True,
            'metrics': {
                'entity': {
                    'precision': round(entity_precision, 4),
                    'recall': round(entity_recall, 4),
                    'f1_score': round(entity_f1, 4),
                    'true_positives': entity_tp,
                    'false_positives': entity_fp,
                    'false_negatives': entity_fn,
                    'ground_truth_count': len(ground_truth_entities),
                    'predicted_count': len(predicted_entities),
                    'matched_gt_entities': matched_gt_entity_ids,
                    'matched_pred_entities': matched_pred_entity_ids,
                    'unmatched_gt_entities': unmatched_gt_entity_ids,
                    'unmatched_pred_entities': unmatched_pred_entity_ids
                },
                'triplet': {
                    'precision': round(triplet_precision, 4),
                    'recall': round(triplet_recall, 4),
                    'f1_score': round(triplet_f1, 4),
                    'true_positives': triplet_tp,
                    'false_positives': triplet_fp,
                    'false_negatives': triplet_fn,
                    'ground_truth_count': len(ground_truth_triplets),
                    'predicted_count': len(predicted_triplets),
                    'matched_gt_triplets': matched_gt_triplet_ids,
                    'matched_pred_triplets': matched_pred_triplet_ids,
                    'unmatched_gt_triplets': unmatched_gt_triplet_ids,
                    'unmatched_pred_triplets': unmatched_pred_triplet_ids
                }
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to calculate metrics: {str(e)}'}, status=500)


# Schema Management Endpoints

@require_http_methods(["GET"])
def get_schemas(request):
    """Get all available schemas and the active schema name."""
    try:
        schemas_file = PROJECT_ROOT / "schemas.json"

        if not schemas_file.exists():
            return JsonResponse({
                'success': False,
                'error': 'schemas.json not found'
            }, status=404)

        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        return JsonResponse({
            'success': True,
            'active_schema': schemas_data.get('active_schema', 'radgraph'),
            'schemas': schemas_data.get('schemas', {})
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load schemas: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_active_schema(request):
    """Get the currently active schema configuration."""
    try:
        schemas_file = PROJECT_ROOT / "schemas.json"

        if not schemas_file.exists():
            return JsonResponse({
                'success': False,
                'error': 'schemas.json not found'
            }, status=404)

        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        active_name = schemas_data.get('active_schema', 'radgraph')
        active_schema = schemas_data['schemas'].get(active_name)

        if not active_schema:
            return JsonResponse({
                'success': False,
                'error': f'Active schema "{active_name}" not found'
            }, status=404)

        return JsonResponse({
            'success': True,
            'schema_name': active_name,
            'schema': active_schema
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load active schema: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def set_active_schema(request):
    """Set which schema to use for extraction."""
    try:
        data = json.loads(request.body) if request.body else {}
        schema_name = data.get('schema_name')

        if not schema_name:
            return JsonResponse({
                'success': False,
                'error': 'schema_name is required'
            }, status=400)

        schemas_file = PROJECT_ROOT / "schemas.json"

        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        # Verify schema exists
        if schema_name not in schemas_data['schemas']:
            return JsonResponse({
                'success': False,
                'error': f'Schema "{schema_name}" not found'
            }, status=404)

        # Update active schema
        schemas_data['active_schema'] = schema_name

        # Save back to file
        with open(schemas_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': f'Active schema set to "{schema_name}"'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to set active schema: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_schema(request):
    """Create or update a schema definition."""
    try:
        data = json.loads(request.body) if request.body else {}
        schema_name = data.get('schema_name')
        schema_data = data.get('schema_data')

        if not schema_name or not schema_data:
            return JsonResponse({
                'success': False,
                'error': 'schema_name and schema_data are required'
            }, status=400)

        # Validate schema structure
        required_fields = ['name', 'description', 'entity_types', 'relation_types']
        for field in required_fields:
            if field not in schema_data:
                return JsonResponse({
                    'success': False,
                    'error': f'Schema missing required field: {field}'
                }, status=400)

        schemas_file = PROJECT_ROOT / "schemas.json"

        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        # Add or update schema
        schemas_data['schemas'][schema_name] = schema_data

        # Save back to file
        with open(schemas_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': f'Schema "{schema_name}" saved successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save schema: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_schema(request):
    """Delete a custom schema (cannot delete built-in schemas)."""
    try:
        data = json.loads(request.body) if request.body else {}
        schema_name = data.get('schema_name')

        if not schema_name:
            return JsonResponse({
                'success': False,
                'error': 'schema_name is required'
            }, status=400)

        # Prevent deletion of built-in schemas
        if schema_name in ['radgraph', 'pet_ct_oncology']:
            return JsonResponse({
                'success': False,
                'error': 'Cannot delete built-in schemas'
            }, status=403)

        schemas_file = PROJECT_ROOT / "schemas.json"

        with open(schemas_file, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)

        # Check if schema exists
        if schema_name not in schemas_data['schemas']:
            return JsonResponse({
                'success': False,
                'error': f'Schema "{schema_name}" not found'
            }, status=404)

        # Delete schema
        del schemas_data['schemas'][schema_name]

        # If deleted schema was active, switch to radgraph
        if schemas_data.get('active_schema') == schema_name:
            schemas_data['active_schema'] = 'radgraph'

        # Save back to file
        with open(schemas_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': f'Schema "{schema_name}" deleted successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete schema: {str(e)}'
        }, status=500)


# Extraction Prompt Management Endpoints

def get_extraction_prompts_file():
    """Get the path to the extraction prompts JSON file."""
    return PROJECT_ROOT / "extraction_prompts.json"


def load_extraction_prompts():
    """Load extraction prompts from JSON file."""
    prompts_file = get_extraction_prompts_file()

    if not prompts_file.exists():
        # Create default prompt based on the existing build_system_prompt_from_schema function
        default_prompts = {
            "active_prompt": "default_schema_based",
            "prompts": {
                "default_schema_based": {
                    "name": "Default Schema-Based Prompt",
                    "description": "The original extraction prompt that dynamically builds from the active schema",
                    "template": """You are a clinical information extraction system specialized in radiology reports. Extract entities and relations from the provided radiology report using the {schema_name} schema.

SCHEMA: {schema_name}
{schema_description}

ENTITY TYPES ({entity_count} types):

{entity_types}

RELATION TYPES ({relation_count} types):

{relation_types}

EXTRACTION INSTRUCTIONS:

1. Read the radiology report carefully
2. Identify ALL entities that fit the entity types defined above
3. For each entity, determine its type based on the definitions
4. Identify relationships between entities using the relation types defined above
5. Extract as many valid entity-relation triplets as possible
6. Be thorough - extract ALL clinically relevant information
7. For measurements (sizes, SUVmax values), extract them as separate entities
8. Preserve anatomical precision (e.g., "segment II", "T8 vertebra", "right upper lobe")

OUTPUT FORMAT:

Return a JSON array of triplets. Each triplet must have this exact structure:
{{
  "entity1_text": "the first entity text",
  "entity1_type": "one of the entity types listed above",
  "relation_type": "one of the relation types listed above",
  "entity2_text": "the second entity text",
  "entity2_type": "one of the entity types listed above"
}}

IMPORTANT:
- Return ONLY the JSON array, no additional text
- Ensure all JSON is valid and properly formatted
- Extract ALL relevant entities and relations
- Be consistent with entity type assignments
- Every relation must connect two valid entities
- For measurements, extract both the value and link it to the lesion"""
                }
            }
        }

        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(default_prompts, f, indent=2, ensure_ascii=False)

        return default_prompts

    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"active_prompt": "default_schema_based", "prompts": {}}


def save_extraction_prompts(prompts_data):
    """Save extraction prompts to JSON file."""
    prompts_file = get_extraction_prompts_file()
    with open(prompts_file, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)


def build_prompt_from_template(template, schema):
    """Build a system prompt from a template and schema."""
    # Format entity types
    entity_types_text = ""
    for i, entity_type in enumerate(schema['entity_types'], 1):
        entity_types_text += f"{i}. **{entity_type['name']}**: {entity_type['description']}\n"

    # Format relation types
    relation_types_text = ""
    for i, relation_type in enumerate(schema['relation_types'], 1):
        relation_types_text += f"{i}. **{relation_type['name']}**: {relation_type['description']}\n"
        if relation_type.get('valid_pairs'):
            pairs_str = ", ".join([f"({pair[0]} → {pair[1]})" for pair in relation_type['valid_pairs'][:3]])
            if len(relation_type['valid_pairs']) > 3:
                pairs_str += f" (and {len(relation_type['valid_pairs']) - 3} more)"
            relation_types_text += f"   Valid pairs: {pairs_str}\n"

    # Replace placeholders in template
    prompt = template.replace('{schema_name}', schema['name'])
    prompt = prompt.replace('{schema_description}', schema['description'])
    prompt = prompt.replace('{entity_types}', entity_types_text.strip())
    prompt = prompt.replace('{relation_types}', relation_types_text.strip())
    prompt = prompt.replace('{entity_count}', str(len(schema['entity_types'])))
    prompt = prompt.replace('{relation_count}', str(len(schema['relation_types'])))

    return prompt


@require_http_methods(["GET"])
def get_extraction_prompts_view(request):
    """Get all available extraction prompts and the active prompt name."""
    try:
        prompts_data = load_extraction_prompts()

        return JsonResponse({
            'success': True,
            'active_prompt': prompts_data.get('active_prompt', 'default_schema_based'),
            'prompts': prompts_data.get('prompts', {})
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load extraction prompts: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def set_active_prompt(request):
    """Set which extraction prompt to use."""
    try:
        data = json.loads(request.body) if request.body else {}
        prompt_name = data.get('prompt_name')

        if not prompt_name:
            return JsonResponse({
                'success': False,
                'error': 'prompt_name is required'
            }, status=400)

        prompts_data = load_extraction_prompts()

        # Verify prompt exists
        if prompt_name not in prompts_data['prompts']:
            return JsonResponse({
                'success': False,
                'error': f'Prompt "{prompt_name}" not found'
            }, status=404)

        # Update active prompt
        prompts_data['active_prompt'] = prompt_name

        # Save back to file
        save_extraction_prompts(prompts_data)

        return JsonResponse({
            'success': True,
            'message': f'Active prompt set to "{prompt_name}"'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to set active prompt: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_prompt(request):
    """Create or update an extraction prompt."""
    try:
        data = json.loads(request.body) if request.body else {}
        prompt_name = data.get('prompt_name')
        prompt_data = data.get('prompt_data')

        if not prompt_name or not prompt_data:
            return JsonResponse({
                'success': False,
                'error': 'prompt_name and prompt_data are required'
            }, status=400)

        # Validate prompt structure
        required_fields = ['name', 'template']
        for field in required_fields:
            if field not in prompt_data:
                return JsonResponse({
                    'success': False,
                    'error': f'Prompt missing required field: {field}'
                }, status=400)

        prompts_data = load_extraction_prompts()

        # Add or update prompt
        prompts_data['prompts'][prompt_name] = prompt_data

        # Save back to file
        save_extraction_prompts(prompts_data)

        return JsonResponse({
            'success': True,
            'message': f'Prompt "{prompt_name}" saved successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save prompt: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_prompt(request):
    """Delete a custom extraction prompt (cannot delete built-in prompts)."""
    try:
        data = json.loads(request.body) if request.body else {}
        prompt_name = data.get('prompt_name')

        if not prompt_name:
            return JsonResponse({
                'success': False,
                'error': 'prompt_name is required'
            }, status=400)

        # Prevent deletion of built-in prompts
        if prompt_name in ['default_schema_based']:
            return JsonResponse({
                'success': False,
                'error': 'Cannot delete built-in prompts'
            }, status=403)

        prompts_data = load_extraction_prompts()

        # Check if prompt exists
        if prompt_name not in prompts_data['prompts']:
            return JsonResponse({
                'success': False,
                'error': f'Prompt "{prompt_name}" not found'
            }, status=404)

        # Delete prompt
        del prompts_data['prompts'][prompt_name]

        # If deleted prompt was active, switch to default
        if prompts_data.get('active_prompt') == prompt_name:
            prompts_data['active_prompt'] = 'default_schema_based'

        # Save back to file
        save_extraction_prompts(prompts_data)

        return JsonResponse({
            'success': True,
            'message': f'Prompt "{prompt_name}" deleted successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete prompt: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_entity_extraction_prompts(request):
    """Get entity extraction prompts."""
    try:
        prompts_data = load_extraction_prompts()
        entity_data = prompts_data.get('entity_extraction', {
            'active_prompt': 'default_entity',
            'prompts': {}
        })

        return JsonResponse({
            'success': True,
            'active_prompt': entity_data.get('active_prompt', 'default_entity'),
            'prompts': entity_data.get('prompts', {})
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load entity extraction prompts: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_relation_extraction_prompts(request):
    """Get relation extraction prompts."""
    try:
        prompts_data = load_extraction_prompts()
        relation_data = prompts_data.get('relation_extraction', {
            'active_prompt': 'default_relation',
            'prompts': {}
        })

        return JsonResponse({
            'success': True,
            'active_prompt': relation_data.get('active_prompt', 'default_relation'),
            'prompts': relation_data.get('prompts', {})
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load relation extraction prompts: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_entity_extraction_prompt(request):
    """Save an entity extraction prompt."""
    try:
        data = json.loads(request.body) if request.body else {}

        # Support both simplified (name, description, template) and advanced (prompt_key, prompt_data) formats
        prompt_key = data.get('prompt_key')
        prompt_data = data.get('prompt_data')

        # If simplified format, build prompt_data and use active key
        if not prompt_key and 'name' in data:
            prompts_data = load_extraction_prompts()
            prompt_key = prompts_data.get('entity_extraction', {}).get('active_prompt', 'default_entity')
            prompt_data = {
                'name': data.get('name', ''),
                'description': data.get('description', ''),
                'template': data.get('template', '')
            }

        if not prompt_key or not prompt_data:
            return JsonResponse({
                'success': False,
                'error': 'prompt_key and prompt_data (or name, description, template) are required'
            }, status=400)

        prompts_data = load_extraction_prompts()

        if 'entity_extraction' not in prompts_data:
            prompts_data['entity_extraction'] = {'active_prompt': 'default_entity', 'prompts': {}}

        prompts_data['entity_extraction']['prompts'][prompt_key] = prompt_data

        set_active = data.get('set_active', False)
        if set_active:
            prompts_data['entity_extraction']['active_prompt'] = prompt_key

        save_extraction_prompts(prompts_data)

        return JsonResponse({
            'success': True,
            'message': f'Entity extraction prompt "{prompt_key}" saved successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save entity extraction prompt: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_relation_extraction_prompt(request):
    """Save a relation extraction prompt."""
    try:
        data = json.loads(request.body) if request.body else {}

        # Support both simplified (name, description, template) and advanced (prompt_key, prompt_data) formats
        prompt_key = data.get('prompt_key')
        prompt_data = data.get('prompt_data')

        # If simplified format, build prompt_data and use active key
        if not prompt_key and 'name' in data:
            prompts_data = load_extraction_prompts()
            prompt_key = prompts_data.get('relation_extraction', {}).get('active_prompt', 'default_relation')
            prompt_data = {
                'name': data.get('name', ''),
                'description': data.get('description', ''),
                'template': data.get('template', '')
            }

        if not prompt_key or not prompt_data:
            return JsonResponse({
                'success': False,
                'error': 'prompt_key and prompt_data (or name, description, template) are required'
            }, status=400)

        prompts_data = load_extraction_prompts()

        if 'relation_extraction' not in prompts_data:
            prompts_data['relation_extraction'] = {'active_prompt': 'default_relation', 'prompts': {}}

        prompts_data['relation_extraction']['prompts'][prompt_key] = prompt_data

        set_active = data.get('set_active', False)
        if set_active:
            prompts_data['relation_extraction']['active_prompt'] = prompt_key

        save_extraction_prompts(prompts_data)

        return JsonResponse({
            'success': True,
            'message': f'Relation extraction prompt "{prompt_key}" saved successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save relation extraction prompt: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def migrate_entity_ids(request):
    """One-time migration: Convert entity IDs to new format and add triplet IDs."""
    try:
        entities_file = str(PROJECT_ROOT / "entities.json")

        # Create backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = str(PROJECT_ROOT / f"entities_backup_{timestamp}.json")
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        # Migrate each entry
        for entry in entities_data:
            # Migrate ground truth to new consistent scheme: gt_e{n}, gt_r{n}
            if 'ground_truths' in entry:
                for schema_name, gt_data in entry['ground_truths'].items():
                    entity_id_map = {}

                    # Migrate entity IDs to gt_e{n}
                    if 'entities' in gt_data:
                        for i, entity in enumerate(gt_data['entities']):
                            old_id = entity.get('id', '')
                            new_id = f"gt_e{i+1}"
                            entity['id'] = new_id
                            entity_id_map[old_id] = new_id

                    # Add triplet IDs (gt_r{n}) and update entity references
                    if 'triplets' in gt_data:
                        for i, triplet in enumerate(gt_data['triplets']):
                            triplet['id'] = f"gt_r{i+1}"
                            if triplet.get('entity1_id') in entity_id_map:
                                triplet['entity1_id'] = entity_id_map[triplet['entity1_id']]
                            if triplet.get('entity2_id') in entity_id_map:
                                triplet['entity2_id'] = entity_id_map[triplet['entity2_id']]

            # Migrate auto-extracted to new consistent scheme: auto{method_idx}_e{n}, auto{method_idx}_r{n}
            if 'extraction_methods' in entry:
                for schema_name, methods in entry['extraction_methods'].items():
                    for method_idx, method in enumerate(methods, 1):
                        entity_id_map = {}

                        # Migrate entity IDs to auto{method_idx}_e{n}
                        if 'entities' in method:
                            for i, entity in enumerate(method['entities']):
                                old_id = entity.get('id', '')
                                new_id = f"auto{method_idx}_e{i+1}"
                                entity['id'] = new_id
                                entity_id_map[old_id] = new_id

                        # Add triplet IDs (auto{method_idx}_r{n}) and update entity references
                        if 'triplets' in method:
                            for i, triplet in enumerate(method['triplets']):
                                triplet['id'] = f"auto{method_idx}_r{i+1}"
                                # Only update refs if using new format (has entity IDs)
                                if 'entity1_id' in triplet and triplet.get('entity1_id') in entity_id_map:
                                    triplet['entity1_id'] = entity_id_map[triplet['entity1_id']]
                                if 'entity2_id' in triplet and triplet.get('entity2_id') in entity_id_map:
                                    triplet['entity2_id'] = entity_id_map[triplet['entity2_id']]

        # Save migrated data
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        return JsonResponse({
            'success': True,
            'message': f'Migration completed. Backup saved to: {Path(backup_file).name}'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)