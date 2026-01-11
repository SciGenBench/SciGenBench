# prompts.py

# ==============================================================================
# SECTION 1: LLM-as-a-Judge (核心评测)
# ==============================================================================

# 该提示词用于让 LLM (Gemini/GPT-4o) 扮演裁判，对生成的图像进行打分。
# 占位符: {text_description} -> 原始的科学问题文本

QUESTION_EVAL_PROMPT = """
You are an expert evaluator of scientific and technical diagrams (e.g., geometry, physics, chemistry).

Evaluate the image against the caption on these 5 dimensions:

### 1. Correctness & Fidelity (0–2)
Core Question: Does the image completely and accurately represent all elements, labels, and spatial/logical relationships from the caption, with no omissions OR hallucinations?
* **2 (High):** Perfect match. All elements (points, lines, shapes, labels) from the caption are present and correct. All specified spatial (e.g., 'left of', 'inside') and logical (e.g., 'perpendicular', 'tangent', 'connected to') relationships are perfectly accurate. Crucially, there are NO spurious or "hallucinated" elements (e.g., random lines, meaningless intersections) not implied by the caption.
* **1 (Medium):** Mostly correct. Most key elements are present, but with minor omissions, misplacements, or simplifications. Spatial/logical relationships are mostly right but have slight inaccuracies. May have minor spurious elements that don't confuse the main subject.
* **0 (Low):** Major mismatch. Key elements are missing, incorrect, or relationships are wrong. Or, the image contains significant spurious content (visual noise, random intersections) that contradicts or confuses the caption.

---

### 2. Layout & Precision (0–2)
Core Question: Is the layout clear and technically precise? Does the visual arrangement correctly reflect the logical coordinates and relative spatial positions described?
* **2 (High):** Professional and spatially accurate. Layout is clear, balanced, and precise (straight lines, exact connections). Visual positions perfectly match the logical labels/coordinates (e.g., $(10,0)$ is distinctively right of $(2,0)$) and relative positions are strictly maintained.
* **1 (Medium):** Generally readable but with minor distortions. Layout is understandable but may have slight alignment issues or imprecision. Relative positions are mostly correct (topology preserved), but scale or visual distances may be inaccurate (e.g., order is right, but proportions are off).
* **0 (Low):** Sloppy or spatially contradictory. Layout is cluttered, chaotic, or elements are poorly proportioned. Lines are visibly imprecise/disconnected, OR elements are placed in positions that contradict their coordinates (e.g., positive coordinates drawn on the negative axis, or inverted positions).

---

### 3. Readability & Occlusion (0–2)
Core Question: Do visual elements or labels overlap or occlude each other in a way that obscures meaning or reduces readability?
* **2 (High):** No occlusion. Every element (shapes, arrows, text labels) is fully distinct and clearly separated, with no confusing overlap.
* **1 (Medium):** Minor overlap. Some elements or labels slightly touch or overlap, but it only marginally affects readability (e.g., an arrowhead just touches a label). The core content remains understandable.
* **0 (Low):** Significant occlusion. Key elements or labels overlap heavily, making parts of the diagram unreadable, ambiguous, or indistinguishable.

---

### 4. Scientific Plausibility (0–2)
Core Question: Does the image visually conform to the basic principles and conventions of its scientific domain (e.g., physics, geometry), even if not explicitly stated in the caption?
* **2 (High):** Visually plausible. The image "looks right" for its domain. E.g., geometric angles/proportions look reasonable; physics vectors (if representing equilibrium) look balanced; chemical bond angles appear conventional (e.g., VSEPR).
* **1 (Medium):** Minor implausibility. The image is scientifically/logically functional but has minor visual flaws (e.g., a 90° angle looks like 80°; a molecule's bond angle is visibly awkward but still conveys the connection).
* **0 (Low):** Visually implausible. The image clearly violates basic scientific/logical principles in its visual representation (e.g., a force diagram that is obviously unbalanced; a geometric proof figure that is impossibly skewed).

---

### 5. Expressiveness & Richness (0–2)
Core Question: Does the image completely and vividly reproduce the scenario described in the problem?
* **2 (High):** Comprehensive reproduction. The image not only contains the correct elements but also effectively conveys the full *context* or *situation* of the problem. It is visually rich and fully illustrates the prompt's intent.
* **1 (Medium):** Basic representation. The image depicts the necessary elements for the problem but lacks contextual richness or detail. It is functional but minimal.
* **0 (Low):** Incomplete scenario. The image fails to convey the setting or context of the problem, making it difficult to understand the "story" or situation behind the diagram.

---

### **Output Format**
Provide short reasoning for each dimension, then output a JSON object with integer scores.

**Example Output:**

**Reasoning:**
* **Correctness & Fidelity:** The image correctly shows all 5 points and the 3 lines connecting them as described. All labels are present. No extra lines appear.
* **Layout & Precision:** Lines are straight and connect perfectly at the nodes. The layout is balanced.
* **Readability & Occlusion:** Label 'A' and 'B' are slightly too close, but do not overlap. All elements are readable.
* **Scientific Plausibility:** The diagram (a geometric proof) shows angles that appear consistent with the "given" perpendicular lines.
* **Expressiveness & Richness:** The diagram fully captures the geometry problem's scenario, clearly visualizing the intersecting planes described in the text.

JSON
{
  "Correctness_Fidelity": 2,
  "Layout_Precision": 2,
  "Readability_Occlusion": 2,
  "Scientific_Plausibility": 2,
  "Expressiveness_Richness": 2
}


Question: {question}

Reason & JSON output:
"""

# ==============================================================================
# SECTION 3: VQA Answering (用于 VQA 逆向验证)
# ==============================================================================

# 该提示词用于让模型看图答题。
# 占位符: {quiz_content} -> 上一步生成的题目
VQA_ANSWER_PROMPT = """
You are taking a visual exam. Look at the provided scientific image and answer the following question. Analyze the image carefully and output the letter of the correct option (A, B, C, or D).
Question:
{quiz_content}
"""

# VQA_ANSWER_PROMPT = """
# You are taking a visual exam. Look at the provided scientific image and answer the following question. Analyze the image carefully and output the answer.
# Question:
# {quiz_content}
# """

ANSWER_VERIFICATION_PROMPT = """
You are a strict grading assistant. I will provide you with a Question, the Correct Option (Standard Answer), and the Student's Response.

Question: {question}
Correct Option: {answer}
Student Response: "{model_response}"

Your task: Determine if the Student Response matches the Correct Option.

The student might output just the letter (e.g., "A") or a sentence (e.g., "The answer is A"). Both are correct.

If the student selects a different option or says they cannot answer, it is incorrect.

Output strictly valid JSON: {{ "is_correct": 1, "reasoning": "short explanation" }} 
"""

ANSWER_VERIFICATION_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly. 
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT: 
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{answer}
<Standard Answer End>

<Candidate's Answer Begin>
{model_response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""
import re
def process_judgment(judgment_str: str) -> str:
    # First try to find the exact \boxed{letter} pattern
    boxed_matches = re.findall(r'boxed{([A-C])}', judgment_str)
    if boxed_matches:
        return boxed_matches[-1]
    
    # Directly return the judgment if it is A, B, or C
    if judgment_str in ["A", "B", "C"]:
        return judgment_str
    else:
        final_judgment_str = judgment_str.split("Final Judgment:")[-1]
        matches = re.findall(r'\(([A-C])\)*', final_judgment_str)
        if matches:
            return matches[-1]
        matches = re.findall(r'([A-C])', final_judgment_str)
        if matches:
            return matches[-1]
        return ""