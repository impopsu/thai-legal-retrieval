# Human Answer Evaluation Guide

Review each generated answer against the question, retrieved source passages,
and reference answer. Score every dimension from 1 (poor) to 5 (strong); use
the notes field to explain uncertain or legally significant judgments.

| Dimension | What to assess |
|---|---|
| Correctness | Whether the legal conclusion and factual statements answer the question accurately. |
| Faithfulness | Whether claims are supported by the retrieved source passages, without invented rules or facts. |
| Citation | Whether each cited law and section matches the supporting source and is relevant to the claim. |
| Completeness | Whether the answer covers the material conditions, exceptions, and qualifications needed to answer the question. |
| Abstention | Whether the system answers when evidence is sufficient and declines or clearly qualifies its answer when evidence is insufficient. |

For all dimensions, 1 means materially wrong, unsupported, missing, or unsafe;
3 means partly satisfactory but with a meaningful gap; and 5 means accurate,
well-supported, appropriate, and complete for the question. Scores 2 and 4 are
intermediate judgments. For abstention, a confident unsupported answer should
score low, while an appropriate refusal or a supported answer should score high.

The JSONL review file leaves scores blank (`null`). Human reviewers should enter
integer scores from 1 to 5 and identify themselves in `reviewer_id`. Do not infer
legal correctness from retrieval rank alone; read the cited and retrieved text.