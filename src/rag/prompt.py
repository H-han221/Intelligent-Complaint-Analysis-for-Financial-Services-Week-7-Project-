def build_prompt(context, question):
    return f"""
You are a financial analyst assistant for CrediTrust Financial.

Use ONLY the complaint excerpts provided below to answer the question.
If the context does not contain enough information, say:
"I do not have enough information to answer this question."

Complaint Excerpts:
{context}

Question:
{question}

Answer:
"""
