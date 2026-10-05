# LangGraph

1. [What is LangGraph?](#what-is-langgraph)
2. [Why Do We Need LangGraph?](#why-do-we-need-langgraph)
3. [Use Cases Where LangGraph Fits](#use-cases-where-langgraph-fits)
4. [The 3 Pillars: State, Nodes, and Edges](#the-3-pillars-state-nodes-and-edges)
5. [Code Walkthrough](#code-walkthrough)
6. [How to Run and Test](#how-to-run-and-test)

---

## What is LangGraph?

LangGraph is an open-source framework built on top of LangChain for building **stateful, multi-step AI workflows** as graphs. Think of it as a way to define *who does what, in what order, and what information gets passed along the way*.

At its core, a LangGraph application is a **directed graph** where:

- **Nodes** are units of work (functions, LLM calls, tool invocations, etc.)
- **Edges** define the flow — which node runs after which
- **State** is the shared memory that flows through the entire graph

```
START → Node A → Node B → Node C → END
           ↓ passes updated state along each arrow ↑
```

LangGraph is part of the **LangChain ecosystem** and is especially suited for building AI agents and multi-step reasoning pipelines that need to maintain context across steps.

---

## Why Do We Need LangGraph?

### The Problem with Simple LLM Chains

A basic LLM pipeline is linear: input → LLM → output. This works fine for one-shot tasks, but breaks down quickly when you need:

| Challenge | Why it's hard without LangGraph |
|---|---|
| **Multi-step workflows** | No built-in way to sequence steps and pass results forward |
| **Shared context** | Each step needs to know what earlier steps discovered |
| **Conditional logic** | Hard to branch ("if diagnosis is X, go to specialist Y") |
| **Cycles and retries** | Simple chains can't loop back to a previous step |

### What LangGraph Gives You

- A **clean mental model** — graph = workflow, easy to reason about
- **Persistent state** that every node can read from and write to
- **Conditional edges** to branch the workflow based on state values
- **Cycles** — you can loop back to a previous node (critical for agents)
- **Checkpointing** — pause, resume, and inspect a workflow at any point

---

## Use Cases Where LangGraph Fits

LangGraph is the right tool when your AI application needs **more than a single LLM call**:

### 1. Multi-Agent Systems
Multiple specialized AI agents collaborating — one researches, one writes, one reviews. Each agent is a node; LangGraph coordinates the handoffs.

### 2. Agentic Tool Use
An AI agent that decides which tool to call (search, calculator, database), executes it, observes the result, and decides the next action — looping until the task is done.

### 3. Sequential Specialist Workflows
A pipeline where the output of one expert becomes the input for the next (doctor → cardiologist → surgeon).

### 4. Document Processing Pipelines
Extract text → summarize → classify → route to the right downstream handler.

### 5. Customer Support Bots
Triage the query → look up account info → decide resolution path → escalate if needed.

### 6. Code Generation & Review Pipelines
Generate code → run tests → if tests fail, loop back to fix the code → if they pass, proceed to review.

### 7. Research Assistants
Break a question into sub-questions → answer each → synthesize → validate the final answer.

---

## The 3 Pillars: State, Nodes, and Edges

These three concepts are all you need to understand to build any LangGraph application.

---

### Pillar 1 — State

**State is the single source of truth that lives inside the graph.**

It is defined as a Python `TypedDict`. Every node receives the current state as input, and can return updates to it. LangGraph merges the returned updates back into the state before passing it to the next node.

```python
class PatientState(TypedDict):
    patient_name:       str
    physician_notes:    str
    cardiologist_notes: str
    surgeon_notes:      str
```

Think of state as a **shared notepad** that gets passed between all the specialists in a hospital. Each specialist can read everything that was written before them and add their own notes.

**Key rules about State:**
- You define its shape upfront using `TypedDict`
- Nodes return a **partial dict** — only the keys they want to update
- LangGraph merges the partial update into the full state automatically
- All nodes share the same state object — there is no hidden data

---

### Pillar 2 — Nodes

**Nodes are the workers. Each node is a plain Python function.**

A node receives the current state and returns a dictionary with the keys it wants to update. It can do anything: call an LLM, query a database, run business logic, invoke a tool.

```python
def general_physician_node(state: PatientState) -> dict:
    print("General Physician examining:", state['patient_name'])
    notes = "Patient reports chest pain and shortness of breath. Heart rate elevated."
    return {"physician_notes": notes}
```

**Key rules about Nodes:**
- Takes the full state as input
- Returns only the keys it wants to update (not the entire state)
- Side effects (like `print`) are fine — they don't affect state
- The function name is just for your code; the node's graph name is set when you call `add_node()`

---

### Pillar 3 — Edges

**Edges are the arrows. They define the execution order.**

An edge says: *after this node finishes, run that node next.*

```python
builder.add_edge(START, "general_physician")
builder.add_edge("general_physician", "cardiologist")
builder.add_edge("cardiologist", "surgeon")
builder.add_edge("surgeon", END)
```

LangGraph has two special sentinel nodes: `START` (the entry point) and `END` (the exit point).

**Types of edges in LangGraph:**

| Edge Type | Description |
|---|---|
| `add_edge(A, B)` | Always go from A to B |
| `add_conditional_edges(A, fn)` | Call `fn(state)` after A; route to whichever node the function returns |

**Key rules about Edges:**
- Every graph must have a path from `START` to `END`
- A node can have multiple outgoing edges (fan-out)
- Conditional edges unlock branching and looping — the gateway to true agentic behaviour

---

## Code Walkthrough

The full application is in [app.py](app.py). Here is a section-by-section breakdown.

---

### Step 1 — Imports

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
```

- `TypedDict` — standard Python for defining typed dictionaries (used to define State)
- `StateGraph` — the main LangGraph class; this is how you build a graph
- `START`, `END` — sentinels that mark the entry and exit points of the graph

---

### Step 2 — Define State

```python
class PatientState(TypedDict):
    patient_name:       str
    physician_notes:    str
    cardiologist_notes: str
    surgeon_notes:      str
```

This defines the shape of the shared notepad. The graph is initialised with one patient's name, and the three `_notes` fields start empty. Each node is responsible for filling in its own field.

---

### Step 3 — Define Nodes

**Node 1 — General Physician** (the entry specialist)

```python
def general_physician_node(state: PatientState) -> dict:
    print("General Physician examining:", state['patient_name'])
    notes = "Patient reports chest pain and shortness of breath. Heart rate elevated."
    return {"physician_notes": notes}
```

Reads `patient_name` from state. Writes back `physician_notes`.

---

**Node 2 — Cardiologist** (reads the physician's output)

```python
def cardiologist_node(state: PatientState) -> dict:
    print("Cardiologist Reading Physician notes: ", state['physician_notes'])
    notes = "Angiography completed. Significant arterial blockage detected. Surgery strongly recommended."
    return {"cardiologist_notes": notes}
```

Reads `physician_notes` that the previous node wrote. Writes back `cardiologist_notes`. Notice that it does **not** need to re-pass `physician_notes` — it is already in state.

---

**Node 3 — Surgeon** (reads both previous outputs)

```python
def surgeon_node(state: PatientState) -> dict:
    print("Surgeon Reading Physician notes: ",    state['physician_notes'])
    print("Surgeon Reading Cardiologist findings:", state['cardiologist_notes'])
    notes = "Bypass surgery (CABG) scheduled"
    return {"surgeon_notes": notes}
```

Reads from both `physician_notes` and `cardiologist_notes`. This shows the power of shared state — the surgeon has access to everything that came before, without any node explicitly "passing" it forward.

---

### Step 4 — Build the Graph

```python
builder = StateGraph(PatientState)

builder.add_node("general_physician", general_physician_node)
builder.add_node("cardiologist",      cardiologist_node)
builder.add_node("surgeon",           surgeon_node)

builder.add_edge(START, "general_physician")
builder.add_edge("general_physician", "cardiologist")
builder.add_edge("cardiologist", "surgeon")
builder.add_edge("surgeon", END)

graph = builder.compile()
```

- `StateGraph(PatientState)` — creates a graph builder bound to your state shape
- `add_node(name, fn)` — registers a node; `name` is what you reference in edges
- `add_edge(from, to)` — wires the execution order
- `builder.compile()` — validates the graph and returns a runnable object

The graph at this point looks like:

```
START
  |
  v
general_physician  →  writes physician_notes
  |
  v
cardiologist       →  reads physician_notes, writes cardiologist_notes
  |
  v
surgeon            →  reads physician_notes + cardiologist_notes, writes surgeon_notes
  |
  v
END
```

---

### Step 5 — Run the Graph

```python
initial_state: PatientState = {
    "patient_name": "Rajesh",
    "physician_notes":    "",
    "cardiologist_notes": "",
    "surgeon_notes":      "",
}

final_state = graph.invoke(initial_state)
print(final_state)
```

- Provide the initial state with the patient name seeded and all notes empty
- `graph.invoke()` runs all nodes in order, passing the evolving state along each edge
- Returns the **final state** with every field populated

---

## How to Run and Test

### Prerequisites

You need Python 3.9 or higher.

### Installation

```bash
# Clone or download the project, then navigate into it
cd LangGraph-Part1

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

### Expected Output

```
General Physician examining: Rajesh
Cardiologist Reading Physician notes:  Patient reports chest pain and shortness of breath. Heart rate elevated.
Surgeon Reading Physician notes:  Patient reports chest pain and shortness of breath. Heart rate elevated.
Surgeon Reading Cardiologist findings:  Angiography completed. Significant arterial blockage detected. Surgery strongly recommended.
{
  'patient_name': 'Rajesh',
  'physician_notes': 'Patient reports chest pain and shortness of breath. Heart rate elevated.',
  'cardiologist_notes': 'Angiography completed. Significant arterial blockage detected. Surgery strongly recommended.',
  'surgeon_notes': 'Bypass surgery (CABG) scheduled'
}
```

### Testing Different Patients

To test with a different patient, change the `patient_name` in the initial state at the bottom of [app.py](app.py):

```python
initial_state: PatientState = {
    "patient_name": "Priya",   # <-- change this
    "physician_notes":    "",
    "cardiologist_notes": "",
    "surgeon_notes":      "",
}
```

Then re-run:

```bash
python app.py
```

### What to Observe

As you run and experiment, notice:

| Observation | What it teaches |
|---|---|
| Each node only returns the keys it updates | Nodes are responsible for their own slice of state |
| The surgeon has access to physician notes even though they were written two nodes ago | State is shared automatically — no manual passing |
| Execution follows the edge order, not the function definition order | Edges control flow, not code order |
| The final `print(final_state)` shows the complete picture | `invoke()` always returns the terminal state |

---

## Summary

| Concept | What it is | In this app |
|---|---|---|
| **State** | Shared typed dictionary | `PatientState` with 4 fields |
| **Node** | A Python function that reads state and returns updates | `general_physician_node`, `cardiologist_node`, `surgeon_node` |
| **Edge** | Arrow connecting nodes; defines execution order | `START → physician → cardiologist → surgeon → END` |
| **StateGraph** | The builder that wires it all together | `builder = StateGraph(PatientState)` |
| **compile()** | Validates and freezes the graph into a runnable | `graph = builder.compile()` |
| **invoke()** | Runs the graph from START to END with an initial state | `graph.invoke(initial_state)` |

This is the foundation. In future parts of this series, we will add conditional edges to branch the workflow, introduce cycles for agentic loops, and connect real LLM calls to the nodes.
