
import json
import os
import sys
from pathlib import Path

def inspect_jsonl():
    """
    Finds a JSONL file in output_jsonl/ and validates its structure.
    """
    # Use path relative to this script file
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "output_jsonl"
    
    if not output_dir.exists():
        print(f"Directory not found: {output_dir.resolve()}")
        return

    # Find first jsonl
    jsonl_files = list(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found in output_jsonl/.")
        return
        
    target_file = jsonl_files[0]
    print(f"Inspecting file: {target_file.name}")
    print(f"File size: {target_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("-" * 50)

    line_count = 0
    errors = 0
    
    with open(target_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_count += 1
            try:
                data = json.loads(line)
                
                # Print the first record details
                if i == 0:
                    print("\nFirst Record Structure & Content:")
                    print(json.dumps(data, indent=2))
                    
                    # Validation: check for critical keys
                    expected_keys = ["id", "combined_content", "consignee_name"]
                    # Note: 'id' might not be explicitly added by my pipeline? 
                    # Azure Search often expects 'id' or uses a key. 
                    # My config has 'carr_eqp_uid' which might be the key?
                    # The pipeline just dumps the dataframe record. 
                    
                    print("\nKey Check of first record:")
                    keys = set(data.keys())
                    print(f"Total keys: {len(keys)}")
                    if "carr_eqp_uid" in keys:
                        print(" [x] carr_eqp_uid found (likely unique key)")
                    else:
                        print(" [!] carr_eqp_uid MISSING")
                        
                    if "combined_content" in keys:
                        print(" [x] combined_content found (RAG text)")
                    else:
                        print(" [!] combined_content MISSING")
                        
            except json.JSONDecodeError as e:
                errors += 1
                print(f"JSON Error at line {i+1}: {e}")
                if errors > 5:
                    print("Too many errors, stopping.")
                    break
    
    print("-" * 50)
    print(f"Scanned {line_count} lines.")
    if errors == 0:
        print("Status: VALID JSONL")
    else:
        print(f"Status: FAILED ({errors} formatting errors)")

if __name__ == "__main__":
    inspect_jsonl()
