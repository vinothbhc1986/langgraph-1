from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# State is a plain Python TypedDict.
# Every node can READ from it and WRITE to it.
class PatientState(TypedDict):
    patient_name:      str           
    physician_notes:   str           
    cardiologist_notes: str          
    surgeon_notes:     str  

# A node is just a Python function.
def general_physician_node(state: PatientState) -> dict:
    print("General Physician examining:",state['patient_name'])

    notes = "Patient reports chest pain and shortness of breath. Heart rate elevated."
    return {"physician_notes": notes}
 
def cardiologist_node(state: PatientState) -> dict:
    print("Cardiologist Reading Physician notes: ",state['physician_notes'])

    notes ="Angiography completed. Significant arterial blockage detected. Surgery strongly recommended."
    return {"cardiologist_notes": notes}
 
def surgeon_node(state: PatientState) -> dict:
    print("Surgeon Reading Physician notes: ",state['physician_notes'])
    print("Surgeon Reading Cardiologist findings: ",state['cardiologist_notes'])

    notes = "Bypass surgery (CABG) scheduled"
    return {"surgeon_notes": notes}

# BUILD THE GRAPH  
builder = StateGraph(PatientState)
 
# Add nodes — first arg is the name, second is the function
builder.add_node("general_physician", general_physician_node)
builder.add_node("cardiologist",      cardiologist_node)
builder.add_node("surgeon",           surgeon_node)
 
# Add edges — defines execution order
builder.add_edge(START, "general_physician")        
builder.add_edge("general_physician", "cardiologist")
builder.add_edge("cardiologist", "surgeon")
builder.add_edge("surgeon", END) 
 
# Compile the graph into a runnable
graph = builder.compile()

# Pass in the INITIAL state — only the keys you want to seed
initial_state: PatientState = {
    "patient_name": "Rajesh",
    "physician_notes": "",   # will be filled by Node 1
    "cardiologist_notes": "",   # will be filled by Node 2
    "surgeon_notes": "",   # will be filled by Node 3
}
final_state = graph.invoke(initial_state)

print(final_state)
 
 
        
 