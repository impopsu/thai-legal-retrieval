# Legal Category Review Guide

Review the full provision text in each row and assign its primary legal category
in `human_category`. Use one of these labels: `property`, `criminal`, `murder`,
`financial`, `company`, `labor`, `family`, `tax`, `procedure`, or `other`.

Compare the human label with `auto_category`, then enter `1` in `correct` when
they match or `0` when they do not. Leave `human_category` and `correct` blank
when the text is insufficient to judge, and explain the reason in `notes`.
Record the reviewer in `reviewer_id`.

After review, calculate accuracy as the number of reviewed rows with `correct=1`
divided by the number of rows with a human-assigned category. Do not count blank
or uncertain rows in the denominator. The current CSV is a review template and
does not contain human-verified accuracy yet.