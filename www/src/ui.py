from shiny import ui
import markdown

def render_readme():
    with open("README.md", "r") as file:
        content = file.read()

    html_content = markdown.markdown(content)
    
    return html_content

# CSS style to make the page visually appealing
css_style = """
.sidebar-panel {
    background-color: #f9f9f9;
    padding: 15px;
    border-right: 1px solid #ddd;
}

.sidebar-panel h3 {
    color: #2c3e50;
    font-weight: bold;
}

.sidebar-panel .form-group {
    margin-bottom: 10px;
}

.sidebar-panel .btn-primary {
    margin-bottom: 10px;
}

.sidebar-panel .checkbox {
    margin-bottom: 10px;
}

.main-panel {
    padding: 15px;
}
"""

app_ui = ui.page_fluid(
    # Include custom CSS styles defined above
    ui.tags.style(css_style),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.h3("Automated Other responses recoding based on historical recoded responses"),
            # File upload input for "Other" responses
            ui.input_file("other_file", label="Upload xlsx file with Other responses", accept=".xlsx"),

            # File upload input for ignored labels (admin or excluded questions), such questions which others will be matched with direct matching(by characters)
            ui.input_file("ignore_file", label="Upload xlsx file with labels of ignored(names of admin, etc.) questions(1 sheet with first column)", accept=".xlsx"),

            # File upload input for historical recoded responses archive
            ui.input_file("historcy_archive", label="Upload file with historical recoded responses", accept=[".zip", ".7z", ".rar"]),

             # Dropdown to select threshold for responses similarity
            ui.input_selectize(
                "response_similarity_treshold", 
                "Select threshold for response similarity",
                choices=[0.6, 0.65, 0.7, 0.75, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.875, 0.90, 0.925, 0.95],
                selected=0.8, 
                multiple=False
            ),

            # Dropdown to select threshold for questions similarity
            ui.input_selectize(
                "question_similarity_treshold", 
                "Select threshold for question similarity",
                choices=[0.85, 0.875, 0.9, 0.925, 0.95],
                selected=0.9, 
                multiple=False
            ),

             # Action button to process data and generate embeddings for responces and questions
            ui.input_action_button("process", "Process data to embeddings representation", color="primary"),

            # Button to download the recoded responses
            ui.download_button("recode_button", "Compare and recode responses"),

            # Checkbox to enable longitudinal format processing(their other file contains response.en.from.uk column instead of response.en)
            ui.input_checkbox("longitudinal", label="Longitudinal format", value=False),
            width=5
        ),
        ui.panel_main(
        )
    )
)
