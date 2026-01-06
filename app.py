import gradio as gr
from src.pipeline import RAGPipeline

rag = RAGPipeline()


def ask_question(user_question):
    if not user_question.strip():
        return "Please enter a question.", ""

    result = rag.ask(user_question)

    answer = result["answer"]
    sources = result["sources"]

    source_text = ""
    for i, meta in enumerate(sources, 1):
        source_text += f"""
Source {i}:
- Product: {meta.get('product')}
- Issue: {meta.get('issue')}
- Company: {meta.get('company')}
- Complaint ID: {meta.get('complaint_id')}
-----------------------
"""

    return answer, source_text


with gr.Blocks(title="CrediTrust Complaint Insight Bot") as demo:
    gr.Markdown(
        """
        ## 🏦 CrediTrust Intelligent Complaint Analysis Bot  
        Ask questions about customer complaints across financial products.
        """
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a question",
            placeholder="Why are customers unhappy with credit cards?"
        )

    ask_btn = gr.Button("Ask")

    answer_output = gr.Textbox(
        label="AI-generated Answer",
        lines=6
    )

    sources_output = gr.Textbox(
        label="Sources (Complaint Evidence)",
        lines=8
    )

    clear_btn = gr.Button("Clear")

    ask_btn.click(
        ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        lambda: ("", "", ""),
        outputs=[question_input, answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch()
