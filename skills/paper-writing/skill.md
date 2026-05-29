---
name: paper-writing
description: Checklist-driven academic paper drafting and review. Use when user says "write section", "draft abstract", "review introduction", "paper checklist", "claim hygiene", or asks to edit any paper section (abstract, intro, method, experiments, discussion, conclusion).
---

# Paper Writing Skill

## Key Principles

1. **Temporal awareness** — Every technique, term, or method must be either defined earlier in the paper OR cited on first use. Before writing, check what has been introduced. If referencing something new, add context or citation.

2. **Claim hygiene** — Every claim must be either:
   - **Generalizable**: backed by theory, proof, or well-established literature
   - **Specific**: backed by your experimental evidence that is robust (multiple seeds, statistical significance, ablations)
   
   Flag any claim that falls into neither category.

3. **Interactive when uncertain** — When unsure about framing, tone, or scope, present 2-3 example rewrites and ask the user to choose. Never guess on critical claims.

4. **Adversarial review** — After writing, spawn a fresh agent (expert in AI scientific paper writing & reviewing) to critique the change. Address issues before finalizing.

5. **Self-explanatory visuals** — Every figure and equation must be understandable without reading surrounding prose. They are hooks for skimming readers. Flag any that require context to parse.

---

## Invoke Words

Use this skill when user says:
- "write section", "draft section"
- "draft abstract", "review abstract"
- "write introduction", "review intro"
- "paper checklist", "section checklist"
- "claim hygiene", "review claims"
- "check my [section]"
- Any request to edit: abstract, introduction, background, related work, method, experiments, discussion, limitations, conclusion

---

## Skill Description

This skill enforces structured checklists for each paper section. Each checklist item is an **exit criterion** — the section is not complete until all items are addressed.

**Workflow:**
1. Identify target section
2. Read current content and count words
3. Run checklist audit — mark items as ✅ present, ⚠️ weak, or ❌ missing
4. For missing/weak items, draft additions with inline citations
5. Verify temporal awareness (no forward references without context)
6. Verify claim hygiene (flag unsupported claims)
7. Spawn adversarial reviewer agent to critique
8. Present final version with word count

---

## Section Checklists

### Abstract (150–200 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Impact**: Why should everyone care? | 1-2 sentences |
| 1 | **Problem**: What gap exists? | 1 sentence |
| 2 | **Scope**: What exact setting are you studying? | 1 sentence |
| 3 | **Core idea**: What is the method? | 1-2 sentences |
| 4 | **Evidence**: How was it evaluated? | 1 sentence |
| 5 | **Results**: Headline numbers | 1 sentence |
| 6 | **Boundary**: What doesn't it do? (optional) | 1 sentence |

---

### Introduction (900–1,100 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Impact**: Why should everyone care? | 1 paragraph |
| 1 | **Problem**: What gap exists? | 1-2 paragraphs |
| 2 | **Scope**: What exact setting are you studying? | 1 paragraph |
| 3 | **Core idea**: What is the method? | 1-2 paragraphs |
| 4 | **Contribution**: What is new? | Bulleted list |
| 5 | **Evidence**: How was it evaluated? | 1 paragraph |
| 6 | **Results**: Headline numbers | Woven in or 1 paragraph |
| 7 | **Baselines**: What did you compare against? | Woven in |
| 8 | **Limitation/boundary**: What's out of scope? | 1 sentence |

---

### Background / Related Work (700–900 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Organization**: By theme, not chronology | Thematic grouping |
| 1 | **Citation discipline**: Every technique cited on first mention | All terms covered |
| 2 | **Gap statement**: Clear connection to your contribution | 1 paragraph |
| 3 | **No orphan citations**: Every cited work tied to your narrative | All citations justified |

---

### Theory / Problem Formulation (700–900 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Problem statement**: Formal definition | 1 paragraph + equations |
| 1 | **Notation table**: All symbols defined | Table or inline |
| 2 | **Assumptions**: Stated explicitly | Bulleted list |
| 3 | **Theoretical grounding**: Propositions/lemmas if applicable | As needed |

---

### Method (1,000–1,300 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Core algorithm**: Clearly stated, ideally in algorithm block | Algorithm or prose |
| 1 | **Equations self-explanatory**: Defined notation, no forward refs | All equations |
| 2 | **Figures aligned**: Reference correct components | All figure refs |
| 3 | **Claim hygiene**: General claims have theory, specific claims have evidence | All claims |
| 4 | **Reproducibility**: Enough detail to reimplement | Key hyperparams |

---

### Experiments (1,500–2,000 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Setup**: Datasets, baselines, metrics defined | 1-2 paragraphs |
| 1 | **Baselines**: What you compared against, why these | 1 paragraph |
| 2 | **Results**: Headline numbers with statistical context | Tables + prose |
| 3 | **Ablations**: What components matter? | 1 subsection |
| 4 | **Figures/tables self-explanatory**: Captions stand alone | All visuals |
| 5 | **Error analysis**: Where does it fail? | 1 paragraph |

---

### Discussion / Limitations (500–700 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Boundary conditions**: When does this NOT work? | Explicit list |
| 1 | **Failure modes**: Acknowledged with examples | 1 paragraph |
| 2 | **Societal impact**: If applicable | 1 paragraph |
| 3 | **Future work**: Concrete next steps | 1 paragraph |

---

### Conclusion (150–250 words)

| # | Criterion | Target |
|---|-----------|--------|
| 0 | **Core contribution restated**: One sentence | 1 sentence |
| 1 | **Key takeaway**: What should practitioner remember? | 1-2 sentences |
| 2 | **No new information**: Only synthesis | — |

---

## Usage Examples

```
User: "review my abstract"
→ Read abstract, run checklist, report ✅/⚠️/❌ for each criterion, suggest fixes

User: "draft introduction"  
→ Read current intro + research notes, run checklist, draft missing pieces, spawn reviewer

User: "claim hygiene check on experiments"
→ Extract all claims, categorize as general/specific, flag unsupported ones

User: "write discussion section"
→ Read paper + results, draft section hitting all checklist items, verify word count
```
