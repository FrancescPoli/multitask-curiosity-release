# Task Compositionality Analysis

## Modifiers Breakdown

We standardized the naming to be fully compositional: `modifier` + `base`.

### Global Modifiers (Cross-Family)

* **`dly` (Delay)**: Introduces a memory delay period. The stimulus disappears before the Go signal.
* **`anti` (Anti-Response)**: Requires responding to the location opposite to the target (180° rotation).
* **`ctx` (Context/Gating)**: Selective attention. Stimuli appear on both Modality 1 (Target) and Modality 2 (Distractor); rule is to attend to Modality 1.

### Task-Specific Modifiers

* **`rt` (Reaction Time)**: **Reach Family Only**. The stimulus appears and remains on screen; the subject must respond immediately. (Condition: Stimulus is present during response).
* **`multi` (Multisensory)**: **Decision Family Only**. Congruent stimuli appear simultaneously on both Ring 1 and Ring 2, increasing input energy/reliability.
* **`cat` (Categorization)**: **Match Family Only**. The matching rule follows **Category Membership** (e.g., is the Test in the same Hemisphere as the Sample?) rather than exact spatial identity.
* **`nms` (Non-Match)**: **Match Family Only**. The rule is flipped: Respond if the Test does **NOT** match the Sample.

> [!NOTE]
> **About the `dly` modifier:** While `dly` manifests differently across families (memory delay for Go, sequential presentation for DM, inherent in Match), it represents a **unified compositional feature**: the requirement for **working memory (WM)**. In all cases, `dly` introduces a temporal gap between stimulus encoding and response, forcing the network to actively maintain information. This makes `dly` a legitimate cross-family modifier despite structural differences.

## Primitives Matrix (Single Modifiers)

| Modifier        | Reach (`go`) | Decision (`dm`) | Match (`ms`) |
| :-------------- | :------------: | :---------------: | :------------: |
| **dly**   |   `dlygo`   |     `dlydm`     |   `dlyms`   |
| **anti**  |   `antigo`   |   `antidlyms`   |                |
| **ctx**   |   `ctxgo`   |     `ctxdm`     |  `ctxdlyms`  |
| **rt**    |    `rtgo`    |        N/A        |      N/A      |
| **multi** |      N/A      |    `multidm`    |      N/A      |
| **cat**   |      N/A      |        N/A        |  `catdlyms`  |
| **nms**   |      N/A      |        N/A        |   `dlynms`   |

## Full Compositional Matrix

This matrix tracks all possible combinations of **Task Configurations** (Types).

* **Total Valid Configuration Types**: 40 (IDs 1-41, excluding ID 23)
* **Total Registered Tasks**: 48 (Due to Modality Splits in DM family)
  * *Note*: In the Decision Family (`dm`), tasks often have two variants (`dm1`/`dm2`) corresponding to Modality 1 or Modality 2. These share a single Matrix ID but are distinct registered environments.

IDs are assigned sequentially to Configurations (including invalid ones to maintain logical order).

### 1. The Reach Family (`go`)

Modifiers: `dly`, `anti`, `ctx`, `rt`

#### Base

* **1.** `go`: Standard Visually Guided Saccade [PRESENT]

#### Level 1 Modifiers

* **2.** `+dly` -> `dlygo` [PRESENT]
* **3.** `+rt` -> `rtgo` [PRESENT]
* **4.** `+anti` -> `antigo` [PRESENT]
* **5.** `+ctx` -> `ctxgo` [PRESENT]

#### Level 2 Modifiers

* **6.** `+dly+anti` -> `dlyantigo` [PRESENT]
* **7.** `+rt+anti` -> `rtantigo` [PRESENT]
* **8.** `+dly+ctx` -> `dlyctxgo` [PRESENT]
* **9.** `+anti+ctx` -> `antictxgo` [PRESENT]
* **10.** `+rt+ctx` -> `rtctxgo` [PRESENT]

#### Level 3 Modifiers

* **11.** `+dly+anti+ctx` -> `dlyantictxgo` [PRESENT]
* **12.** `+rt+anti+ctx` -> `rtantictxgo` [PRESENT]

### 2. The Decision Family (`dm`)

Modifiers: `dly` (sequential), `ctx`, `anti`, `multi`

#### Base (Simultaneous)

* **13.** `dm1`/`dm2`: Perceptual Decision Making [PRESENT]

#### Level 1 Modifiers

* **14.** `+dly` (Sequential) -> `dlydm1`/`dlydm2` [PRESENT]
* **15.** `+multi` -> `multidm` [PRESENT]
* **16.** `+ctx` -> `ctxdm1`/`ctxdm2` [PRESENT]
* **17.** `+anti` -> `antidm1`/`antidm2` [PRESENT]

#### Level 2 Modifiers (General)

* **18.** `+dly+ctx` -> `ctxdlydm1`/`ctxdlydm2` [PRESENT]
* **19.** `+dly+anti` -> `antidlydm1`/`antidlydm2` [PRESENT]
* **20.** `+ctx+anti` -> `antictxdm1`/`antictxdm2` [PRESENT]

#### Level 2 Modifiers (Multi Interactions)

* **21.** `+dly+multi` -> `multidlydm` [PRESENT]
* **22.** `+multi+anti` -> `antimultidm` [PRESENT]

#### Level 3 Modifiers

* **23.** `+dly+ctx+anti` -> `antictxdlydm1`/`antictxdlydm2` [PRESENT]
* **24.** `+dly+multi+anti` -> `antimultidlydm` [PRESENT]

### 3. The Match Family (`ms`)

Modifiers: `dly` (base), `ctx`, `anti`, `cat`, `nms`

#### Base (Delayed by default)

* **25.** `dlyms`: Delay Match-to-Sample [PRESENT]

#### Level 1 Modifiers (Applied to dlyms)

* **26.** `+anti` -> `antidlyms` [PRESENT]
* **27.** `+ctx` -> `ctxdlyms` [PRESENT]
* **28.** `+cat` -> `catdlyms` [PRESENT]
* **29.** `+nms` -> `dlynms` [PRESENT] (Non-Match)

#### Level 2 Modifiers

* **30.** `+anti+ctx` -> `antictxdlyms` [PRESENT]
* **31.** `+anti+cat` -> `anticatdlyms` [PRESENT]
* **32.** `+ctx+cat` -> `ctxcatdlyms` [PRESENT]
* **33.** `+cat+nms` -> `catdlynms` [PRESENT]
* **34.** `+anti+nms` -> `antidlynms` [PRESENT]
* **35.** `+ctx+nms` -> `ctxdlynms` [PRESENT]

#### Level 3 Modifiers

* **36.** `+anti+ctx+cat` -> `antictxcatdlyms` [PRESENT]
* **37.** `+anti+ctx+nms` -> `antictxdlynms` [PRESENT]
* **38.** `+anti+cat+nms` -> `anticatdlynms` [PRESENT]
* **39.** `+ctx+cat+nms` -> `ctxcatdlynms` [PRESENT]

#### Level 4 Modifiers

* **40.** `+anti+ctx+cat+nms` -> `antictxcatdlynms` [PRESENT]

## Task Definitions (Poli Suite)

Implementation: [curiosity/poli_tasks.py](file:///c:/Users/fp02/OneDrive/Documenti/multitask-curiosity/curiosity/poli_tasks.py)

### 1. The Reach Family (`go`)

* `poli.go`: Standard Visually Guided Saccade.
* `poli.dlygo`: Memory Guided Saccade.
* `poli.rtgo`: Reaction Time Go (Immediate Response).
* `poli.antigo`: Anti-Saccade.
* `poli.ctxgo`: Context-Selectivity. Target on Ring 1, Distractor on Ring 2. Rule: Attend Ring 1.
* **Compositions**:
  * `poli.dlyantigo`: Memory Guided Anti-Saccade.
  * `poli.rtantigo`: Reaction Time Anti-Saccade.
  * `poli.dlyctxgo`: Memory Guided Context-Selectivity.
  * `poli.antictxgo`: Anti-Context-Selectivity.
  * `poli.rtctxgo`: Reaction Time Context-Selectivity.
  * `poli.dlyantictxgo`: Memory Guided Anti-Context.
  * `poli.rtantictxgo`: Reaction Time Anti-Context.

### 2. The Decision Family (`dm`)

* `poli.dm1` / `poli.dm2`: Perceptual Decision Making. Simultaneous.
* `poli.dlydm1` / `poli.dlydm2`: Delay Decision Making (Sequential: S1 -> Delay -> S2).
* `poli.multidm`: Multisensory Integration DM.
* `poli.ctxdm1` / `poli.ctxdm2`: Context-Dependent DM.
* `poli.antidm1`: Anti-Decision.
* **Compositions**:
  * `poli.ctxdlydm1`: Context Delay DM.
  * `poli.multidlydm`: Multisensory Delay DM.
  * `poli.antidlydm1` [MISSING]: Anti Delay DM.
  * `poli.antictxdm1` [MISSING]: Anti Context DM.
  * `poli.antictxdlydm1` [MISSING]: Anti Context Delay DM.

### 3. The Match Family (`ms`)

* `poli.dlyms`: Delay-Match-to-Sample.
* `poli.dlynms`: Delay-Non-Match-to-Sample.
* `poli.catdlyms`: Category-Match-to-Sample (DMC).
* `poli.catdlynms`: Category-Non-Match-to-Sample (DNMC).
* `poli.antidlyms`: Anti-Match-to-Sample.
* `poli.ctxdlyms`: Context-Match-to-Sample.
* `poli.antidlynms`: Anti-Non-Match.
* `poli.ctxdlynms`: Context-Non-Match.
* **Compositions**:
  * `poli.antictxdlyms`: Anti Context Match.
  * `poli.anticatdlyms`: Anti Category Match.
  * `poli.ctxcatdlyms`: Context Category Match.
  * `poli.catdlynms`: NMS + Category (DNMC).
  * `poli.antidlynms`: NMS + Anti.
  * `poli.ctxdlynms`: NMS + Context.
  * `poli.antictxcatdlyms` (L3): Anti Context Category Match.
  * `poli.antictxdlynms` (L3): Anti Context NMS.
  * `poli.anticatdlynms` (L3): Anti Category NMS.
  * `poli.ctxcatdlynms` (L3): Context Category NMS.
  * `poli.antictxcatdlynms` (L4): Anti Context Category NMS.

### The "Probe" Method: Testing Compositionality

We determine whether a task is "learned compositionally" by performing **Matrix Surgery** at test time.

1. **Freeze the Brain**: We lock all recurrent weights ($W_{rec}$), input weights ($W_{in}$), and output weights ($W_{out}$). The "dynamical landscape" of the RNN is fixed.
2. **Add a New Rule**: We expand the rule input matrix ($W_{rule}$) by adding **one single column**.
3. **Optimize**: We train *only* this new rule vector (using gradient descent) to solve a held-out task (e.g., `dlyantigo`).

**Interpretation**:
This effectively asks the model: *"Is there any input button I can press that triggers the behavior you need?"*

Think of $W_{rec}$ as a **toolbox** of dynamical primitives (or subroutines):

* A tool for "Waiting" (Delay dynamics)
* A tool for "Integrating" (DM dynamics)
* A tool for "Inverting" (Anti dynamics)

The **Rule Vector** is the **hand** that selects which tools to use.

If the probe succeeds, it proves that:

1. The model has learned independent, reusable mechanisms for "Delay" and "Anti" (from previous tasks).
2. These mechanisms are **compatible**—they can be activated simultaneously by a simple linear input, without needing to rewire the brain ($W_{rec}$).

This is the definition of **compositional generalization**: solving a new combination of known primitives without learning new dynamics.

## 2. Systematic Generalization: The Hierarchical Split

To strictly test **Compositional Generalization**, we must ensure the model **never sees** certain combinations of modifiers during training, requiring it to "synthesize" them at test time from independent primitives.

Instead of random omissions, we apply a **Hierarchical Restriction** logic to the primitives. This creates systematic "Blind Spots" where entire classes of compositions are withheld from specific families.

### The Strategy: Primitive Hierarchy

We assign each cross-family primitive to a specific subset of families during training. This ensures that while every primitive is learned *somewhere*, their **combinations** are missing in specific target families.

1.  **Universal (`dly`)**: Trained in **ALL** families (Reach, Decision, Match).
    *   *Role*: The temporal substrate common to all tasks.
2.  **Shared (`anti`)**: Trained in **Reach + Decision**.
    *   *Constraint*: Held out from **Match**. The model must transport "Inversion" logic to the Matching rule.
3.  **Restricted (`ctx`)**: Trained **ONLY in Reach**.
    *   *Constraint*: Held out from **Decision** and **Match**. The model must transport "Gating" logic to these families.

### Summary Table

| Primitive | Reach (`go`) | Decision (`dm`) | Match (`ms`) |
| :--- | :---: | :---: | :---: |
| **Delay** | **Train** | **Train** | **Train** |
| **Anti** | **Train** | **Train** | <span style="color:red">**TEST**</span> |
| **Context** | **Train** | <span style="color:red">**TEST**</span> | <span style="color:red">**TEST**</span> |

---

### The Training Curriculum (20 Tasks)

This curriculum ensures the model learns the "atoms" and family-specific dynamics (`rt`, `multi`, `cat`), plus all compositions within the "Rosetta Stone" family (Reach).

**1. Reach Family (The Rosetta Stone)**
*10 Tasks. Sees all primitives and their combinations.*
*   **Base**: `poli.go`, `poli.rtgo`
*   **Primitives**: `poli.dlygo`, `poli.antigo`, `poli.ctxgo`
*   **Compositions**:
    *   `poli.dlyantigo` (Delay + Anti)
    *   `poli.dlyctxgo` (Delay + Context)
    *   `poli.antictxgo` (Anti + Context)
    *   `poli.dlyantictxgo` (Delay + Anti + Context)
    *   `poli.rtantictxgo` (Full Complexity)

**2. Decision Family**
*6 Tasks. Sees Delay and Anti. Never Context.*
*   **Base**: `poli.dm1`, `poli.dm2` (Simultaneous)
*   **Primitives**: `poli.dlydm1` (Sequential), `poli.multidm` (Multisensory)
*   **Shared**: `poli.antidm1` (Anti)
*   **Composition**: `poli.antidlydm1` (Anti + Delay)

**3. Match Family**
*4 Tasks. Sees Delay. Never Anti or Context.*
*   **Base**: `poli.dlyms` (Delay Match)
*   **Variants**: `poli.dlynms` (Non-Match), `poli.catdlyms` (Category), `poli.catdlynms` (Cat + Non-Match)

*(Total: 10 + 6 + 4 = 20 Tasks)*

---

### Held-Out Set (The Test Targets)

These tasks require **Systematic Generalization**: transporting a known primitive to a new family and composing it with local dynamics.

**1. Transport "Context" (`ctx`)**
*   **To Decision**: 
    *   `poli.ctxdm1`, `poli.ctxdm2`
    *   `poli.ctxdlydm1`, `poli.ctxdlydm2` (Delay + Context)
    *   `poli.antictxdm1` (Context + Anti) [Triple]
    *   `poli.antictxdlydm1` [Quadruple]

**2. Transport "Anti" (`anti`)**
*   **To Match**: 
    *   `poli.antidlyms`, `poli.antidlynms`
    *   `poli.anticatdlyms` (Anti + Category)

**3. Transport "Context" + "Anti" (Double Transport)**
*   **To Match**: 
    *   `poli.antictxdlyms` (The ultimate test)
    *   `poli.antictxcatdlyms`

---

## 4. The Generalization Test Sets

To rigorously evaluate compositionality, we compare performance on **Held-Out Targets** against **Training Controls**.

### The Logic: Input Control Baseline
Why test on tasks the model has already seen?
**Rule:** When testing, we use a **New Rule Index** (fresh input channels) for *all* tasks, including training ones.
**Reasoning:** This establishes a **baseline cost** for "finding" the correct task dynamics by optimizing input weights alone.
*   **Baseline ($N$ steps)**: Time to re-learn the input mapping for a known task (e.g., `dlymns`). The hidden structure already exists.
*   **Generalization ($M$ steps)**: Time to learn the inputs for a held-out task (e.g., `antidlyms`).
*   **Inference:** If $M \approx N$, the model simply "found" the composition in its latent space (True Generalization). If $M \gg N$, the model had to train/modify its hidden dynamics, failing to generalize zero-shot.

### A. Quick Test Set (Rapid Probe)
*Minimal set to check for any signal of compositionality.*
*   **Controls (Training Baseline)**:
    *   `poli.antidm1` (Matched Complexity: 2 Ops)
    *   `poli.dlyantigo` (Matched Complexity: 3 Ops)
    *   `poli.dlyantictxgo` (Matched Complexity: 4 Ops)
*   **Targets (Held-Out)**:
    *   `poli.ctxdm1` (Transport Ctx -> Decision = 2 Ops)
    *   `poli.antidlyms` (Transport Anti -> Match = 3 Ops)
    *   `poli.antictxdlyms` (Double Transport = 4 Ops)

### B. Full Test Set (Comprehensive)
*Complete evaluation of the entire compositional landscape.*
*   **Controls**: All 20 Training Tasks (as baseline benchmarks).
*   **Targets**: All 14 Held-Out Tasks defined above.
*   **Total**: 34 Tasks.
