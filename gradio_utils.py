import gradio as gr


def gradio_rag_blocks(title, description, submit_fun, theme):
    with gr.Blocks(theme=theme) as demo:
        # Title
        gr.Markdown(f"# <center> {title} </center>")
        # description
        gr.Markdown(f"#### {description}")

        # Input section
        with gr.Row():
            user_input = gr.Textbox(label="Ask and get answer from your data:")

        # Output section
        with gr.Row():
            with gr.Accordion('Generated answer:', open=True):
                generated_answer_output = gr.Markdown()

            retrieved_resources_output = gr.Textbox(label="Retrieved resources:")

        # Button section
        with gr.Row():
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear")

        # Set the function to be called when the submit button is clicked
        submit_button.click(
            fn=submit_fun,
            inputs=user_input,
            outputs=[generated_answer_output, retrieved_resources_output]
        )

        # Set the function to be called when the clear button is clicked
        clear_button.click(
            fn=lambda: ("", ""),
            inputs=[],
            outputs=[generated_answer_output, retrieved_resources_output]
        )
        clear_button.click(
            fn=lambda: "",
            inputs=[],
            outputs=user_input  # Clears the input box
        )
        # Make Enter key hit the submit button
        user_input.submit(
            fn=submit_fun,
            inputs=user_input,
            outputs=[generated_answer_output, retrieved_resources_output]
        )
    return demo
