import requests
import json_numpy as json
from PIL import Image
import numpy as np
import os

def get_batch_actions(instructions, image_path, temperature=1.0):
    """
    Get batch actions for multiple instructions.
    
    Args:
        instructions: List of instruction strings or a single instruction string
        image_path: Path to the image file
        temperature: Temperature for sampling
    
    Returns:
        Tuple of (output_ids, actions) as numpy arrays
    """
    image_path = os.path.abspath(image_path)
    
    # Handle both single instruction and list of instructions
    if isinstance(instructions, str):
        instructions = [instructions]
    
    payload = {
        "instructions": instructions,
        "image_path": image_path,
        "temperature": temperature
    }

    res = requests.post(
        "http://localhost:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    res.raise_for_status()
    return np.array(json.loads(res.text)["output_ids"]), np.array(json.loads(res.text)["actions"])

# Example with multiple different instructions
instructions = [
    "close the drawer",
    "open the drawer", 
    "pick up the cup"
]
image_path = "robot.jpg"

actions = get_batch_actions(
    instructions=instructions,
    image_path=image_path,
    temperature=1.0
)

print("Discrete Action Tokens: \n", actions[0])
print("Continuous actions: \n", actions[1])