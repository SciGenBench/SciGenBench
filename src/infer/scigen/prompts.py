CAPAUG_PROMPT_TEMPLATE = """
You are a professional STEM problem illustrator.

Your task is to generate a clear, precise, and information-complete diagram that visually represents the scientific scenario described in the problem below.

The illustration must:
- Faithfully reflect the problem context and scenario.
- Explicitly depict all entities, objects, variables, and conditions mentioned in the problem.
- Encode quantitative or relational information visually when possible (e.g., distances, angles, forces, directions, labels, axes).
- Help a student understand the setup of the problem, not the solution.

The illustration must NOT:
- Introduce any assumptions, values, or objects not stated or directly implied by the problem.
- Include solution steps, calculations, or conclusions.
- Add decorative or artistic elements unrelated to the problem.

Use a clean, textbook-style schematic:
- Neutral colors
- Clear labels and annotations
- Simple geometric shapes or standard scientific symbols

**Problem Text:**
{question}

**Caption:**
{caption}
"""


PROMPT_TEMPLATE = """
You are a professional STEM problem illustrator.

Your task is to generate a clear, precise, and information-complete diagram that visually represents the scientific scenario described in the problem below.

The illustration must:
- Faithfully reflect the problem context and scenario.
- Explicitly depict all entities, objects, variables, and conditions mentioned in the problem.
- Encode quantitative or relational information visually when possible (e.g., distances, angles, forces, directions, labels, axes).
- Help a student understand the setup of the problem, not the solution.

The illustration must NOT:
- Introduce any assumptions, values, or objects not stated or directly implied by the problem.
- Include solution steps, calculations, or conclusions.
- Add decorative or artistic elements unrelated to the problem.

Use a clean, textbook-style schematic:
- Neutral colors
- Clear labels and annotations
- Simple geometric shapes or standard scientific symbols

**Problem Text:**
{question}
"""

test_questions = [
    "A 7200 lb airplane lands on an aircraft carrier with a speed of 72 ft/s. The plane is caught by an elastic band (k = 998 lb/ft) that has an initial stretch of 5.6 feet. What is the maximum distance the band is stretched?",
    "Find the area of the surface $\\Sigma = \\{ (x,y,z) | z = x^2+y^2; 0 \\leq x \\leq 1, 0 \\leq y \\leq 2\\}$.",
    "A spring-driven dart gun propels a 12g dart. It is cocked by exerting a force of 25N over a distance of 5.0cm. With what speed will the dart leave the gun, assuming the spring has neglible mass?"
]


test_captions = [
"""
### Image 1: Mechanical Diagram
**Scene Summary:** A mechanical diagram showing an airplane connected to an elastic band anchored at one end, with labels for the airplane's weight and speed, the spring constant of the band, and its initial stretch.
**Explicit Component Breakdown:**
* **Airplane (`7200 lb`):** A schematic representation of an airplane labeled with its weight “7200 lb” and initial velocity “72 ft/s.”
* **Elastic Band (`k = 998 lb/ft`):** A coiled spring labeled with its spring constant “k = 998 lb/ft.”
* **Anchor Point (`None`):** A fixed point at one end of the elastic band.
* **Initial Stretch (`5.6 ft`):** A segment of the elastic band shown stretched and labeled “5.6 ft.”
**Interactions and Relationships:**
* The airplane is shown moving toward the anchor point, with a direction arrow indicating motion.
* The elastic band is connected between the airplane and the anchor point.
**Implicit and Inferred Properties:**
* **Elastic Band:** Assumed to obey Hooke’s Law (F = kx).
* **Airplane motion:** Stops due to energy being fully absorbed by the elastic band.
* **System:** No energy loss assumed; conservation of mechanical energy applies.
**Identified Ambiguities:** None.
""",
"**### Image 1: Plot**\n**Scene Summary:** A 3D surface plot showing the graph of $z = x^2 + y^2$ over the rectangular domain $0 \\leq x \\leq 1$, $0 \\leq y \\leq 2$, with labeled axes and domain boundaries.\n\n**Explicit Component Breakdown:** \n* **Surface (`z = x² + y²`):** A smooth paraboloid-like surface rising from the xy-plane, visually representing the function $z = x^2 + y^2$.\n* **Domain Boundary (`x from 0 to 1`):** A rectangle in the xy-plane with edges labeled at x = 0 and x = 1.\n* **Domain Boundary (`y from 0 to 2`):** A rectangle in the xy-plane with edges labeled at y = 0 and y = 2.\n* **Axes (`x, y, z`):** Three labeled axes showing the orientation of the coordinate system.\n\n**Interactions and Relationships:** \n* The surface is graphed only over the specified rectangular domain in the xy-plane.\n* The height of the surface increases quadratically with x and y.\n* The plot includes a visible grid or contour lines to indicate the curvature and slope of the surface.\n\n**Implicit and Inferred Properties:** \n* The surface is continuous and smooth over the given domain.\n* The surface area is to be computed using the standard surface area integral for functions of two variables.\n\n**Identified Ambiguities:** \n* None.",
"**### Image 1: Mechanical Diagram**  \n**Scene Summary:** A mechanical diagram of a spring-driven dart gun, showing the spring being compressed by a force over a specific distance, with the dart positioned for launch.  \n\n**Explicit Component Breakdown:**  \n* **Dart (`12 g`):** A small cylindrical object labeled \"12 g\" positioned in the barrel of the gun.  \n* **Spring (`None`):** A coiled spring shown in the compressed state.  \n* **Compression Arrow (`25 N over 5.0 cm`):** A double-headed arrow indicating the direction of compression, labeled \"25 N over 5.0 cm.\"  \n\n**Interactions and Relationships:**  \n* The spring is shown in a compressed state, indicating that a force has been applied to store potential energy.  \n* The dart is positioned at the end of the barrel, ready to be launched upon release of the spring.  \n* The compression arrow shows the direction and magnitude of the applied force and the distance over which it was applied.  \n\n**Implicit and Inferred Properties:**  \n* **Spring:** Assumed to be ideal and massless.  \n* **System:** No energy is lost to friction or air resistance.  \n* **Energy conversion:** The work done on the spring is fully converted into kinetic energy of the dart.  \n\n**Identified Ambiguities:**  \n* None."
]



IMGCODER_PYTHON_PROMPT = """
You are a multimodal reasoning assistant skilled in scientific visualization using Python.
You will be given:
- **Original Question** (`{{question}}`): The complete problem text.

Your task:
👉 First, carefully **PLAN** the diagram (reasoning stage).
👉 Then, produce a standalone, runnable Python script using Matplotlib (coding stage).
👉 **Key Objective**: Create a diagram that fully represents the **Initial Setting** of the problem.
   - **Completeness**: Visualize all physical objects, geometric shapes, and **given values** (e.g., lengths, angles, forces, labels) mentioned in the text.
   - **Confidentiality**: Do NOT reveal the final answer, result, or derivation steps. Only show what is *given* before the problem is solved.
👉 Always use a clear **textbook-style illustration style**.

---

### 🔍 Output Format (two sections, clearly separated)

#### **Section 1: Plan**
Provide a structured reasoning plan containing the following **four parts**:

1. **Image Content** — Describe the elements to be drawn (shapes, objects, points, lines). Ensure all entities mentioned in the text are accounted for.
2. **Layout** — Explain the approximate spatial arrangement: relative positioning, coordinates estimation, scale.
3. **Labels** — Specify labels and annotations. **Crucial**: List all **given values** (numbers, variables) from the text that must be labeled on the diagram to show the initial state.
4. **Drawing Considerations** — Mention stylistic or logical constraints:
   - What must **NOT** be shown (to avoid solution leakage).
   - Matplotlib-specific details (e.g., `set_aspect('equal')`, patches).

Each section should be 1–4 bullet points.

---

#### **Section 2: Python Code**
Provide a **complete and runnable Python script**, formatted like this:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_diagram():
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal') # Crucial for geometry/physics
    ax.axis('off')         # Hide axes unless strictly necessary for graphs

    # 2. Define Coordinates
    # ...

    # 3. Draw Elements (Shapes, Lines, etc.)
    # ...

    # 4. Add Labels and Annotations
    # (Ensure all given values from the question are visibly labeled)
    # ...

    # 5. Finalize and Show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_diagram()

```

**Rules:**

* **Initial Setting Only**: The diagram must represent the problem *before* any solution steps are taken. Visualize the "Given" but hide the "Solution".
* **Completeness**: If the text says "radius is 5", the diagram must show the circle and label the radius "r=5".
* **Geometric Accuracy**: Ensure `ax.set_aspect('equal')` is used to prevent distortion for geometric/physical diagrams.
* **Style**: Textbook aesthetic—consistent line thickness, clean alignment, clear font sizes.

---

### **Input Fields**

Original Question:
{question}

---

Now, follow the format strictly and generate the output.
"""

IMGCODER_TIKZ_PROMPT = """
You are a multimodal reasoning assistant skilled in scientific visualization using TikZ.
You will be given:
- **Original Question** (`{{question}}`): The complete problem text.

Your task:
👉 First, carefully **PLAN** the diagram (reasoning stage).
👉 Then, produce a standalone, compilable TikZ LaTeX document (coding stage).
👉 **Key Objective**: Create a diagram that fully represents the **Initial Setting** of the problem.
   - **Completeness**: Visualize all physical objects, geometric shapes, and **given values** (e.g., lengths, angles, forces, labels) mentioned in the text.
   - **Confidentiality**: Do NOT reveal the final answer, result, or derivation steps. Only show what is *given* before the problem is solved.
👉 Always use a clear **textbook-style illustration style**.

---

### 🔍 Output Format (two sections, clearly separated)

#### **Section 1: Plan**
Provide a structured reasoning plan containing the following **four parts**:

1. **Image Content** — Describe the elements to be drawn (shapes, objects, points, lines). Ensure all entities mentioned in the text are accounted for.
2. **Layout** — Explain the approximate spatial arrangement: relative positioning, scale, symmetry, etc.
3. **Labels** — Specify labels and annotations. **Crucial**: List all **given values** (numbers, variables) from the text that must be labeled on the diagram to show the initial state.
4. **Drawing Considerations** — Mention stylistic or logical constraints:
   - What must **NOT** be shown (to avoid solution leakage).
   - TikZ-specific details (e.g., libraries, arrow styles, patterns).

Each section should be 1–4 bullet points.

---

#### **Section 2: TikZ Code**
Provide a **complete and compilable LaTeX document**, formatted exactly like this:

```latex
\\documentclass[tikz]{{standalone}}
\\usetikzlibrary{{arrows.meta, positioning, angles, quotes, calc, shapes.geometric, patterns, circuits.ee.IEC}}
\\begin{{document}}
\\begin{{tikzpicture}}
  % 1. Define Coordinates
  % ...

  % 2. Draw Elements (Shapes, Lines, etc.)
  % ...

  % 3. Add Labels and Annotations
  % (Ensure all given values from the question are visibly labeled)
  % ...

\\end{{tikzpicture}}
\\end{{document}}

```

**Rules:**

* **Initial Setting Only**: The diagram must represent the problem *before* any solution steps are taken. Visualize the "Given" but hide the "Solution".
* **Completeness**: If the text says "Angle A is 30 degrees", the diagram must show the angle and label it "30°".
* **Libraries**: Include all standard TikZ libraries needed.
* **Style**: Textbook aesthetic—consistent line thickness, clean alignment, clear labeling, and balanced proportions.

---

### **Input Fields**

Original Question:
{question}

---

Now, follow the format strictly and generate the output.
"""