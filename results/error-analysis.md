# Retrieval Error Analysis

## Sample

Validation sample: 50 questions.

One question (ID 6) had no positive context in the dataset and was therefore treated as a data issue rather than a retrieval error.

## Observed Errors

| ID | Category | Description |
|---|---|---|
| 2 | Wrong section, same law | Retrieved related sections from the correct procurement law but missed section 101. |
| 5 | Wrong section, same law | Retrieved sections from the Civil and Commercial Code but missed section 106. |
| 9 | Wrong section, same law | Retrieved related limitation sections but missed section 186. |
| 11 | Wrong section, same law | Retrieved related Revenue Code sections but missed section 91/21. |
| 28 | Wrong section, same law | Retrieved related Securities and Exchange Act sections but missed section 312. |
| 33 | Wrong section / related law | Retrieved legally related sections but missed section 31. |
| 36 | Wrong law | Retrieved Derivatives Act sections instead of the Securities and Exchange Act section 5. |
| 47 | Wrong section, same law | Retrieved related Civil and Commercial Code sections but missed section 154. |
| 6 | Dataset issue | Positive context is empty although a positive answer exists. |

## Main Finding

Most observed retrieval errors occur at the section-ranking level. The retriever often identifies the correct law or a legally related area but fails to rank the exact gold section within the top five results.

This supports the use of a second-stage Cross-Encoder Reranker after initial candidate retrieval.
