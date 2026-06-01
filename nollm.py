# tools
def calculator(expression):
    try: 
        result = eval(expression)
        return f"Result = {result}"
    except:
        return "Invalid Math expression"
def search_tool(query):
    # Fake search
    knowledge = {
        "capital of nepal": "Kathmandu",
        "capital of france": "Paris",
        "2+2": "4"
    }
    return knowledge.get(query.lower(), "No information found")


# Agent Brain
def agent_reasoning(goal):
    if any(char.isdigit() for char in goal):
        return "calculator"
    
    if "capital" in goal.lower():
        return "search"
    
    return "unknown"

# Agent Loop
def agent(goal):
    