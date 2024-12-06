import pandas as pd
import numpy as np
import os
from www.src.utils import (
    extract_archive, preprocess_data, preprocess_data_longit, 
    get_openai_embedding, get_max_similarity, get_max_similarity_index, 
    get_same_format, get_embedding, check_basic_columns, 
    check_file_correct
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from shiny import App, render, ui, reactive

# Check that model folder exists, if not - load it using load_model.py
if not os.path.isdir("bge-small"):
    print("Model folder doesn't exist, please, run load_model.py to load model and redeploy code.")
    exit()

# Load text2vec model for embeddings
model = SentenceTransformer("bge-small")

def server(input, output, session):
    # Ensure required directories exist; create them if not
    # Directory for output xlsx files
    if not os.path.exists("output_dir"):
        os.mkdir("output_dir")
    # Directory for files from history data input
    if not os.path.exists("archive_dir"):
        os.mkdir("archive_dir")

    # Reactive global variables to hold shared data across server functions
    responses_dictionary_global = reactive.Value(None)
    ignored_questions_dictionary_global = reactive.Value(None)
    ignoring_questions_list = reactive.Value(None)
    others = reactive.Value(None)
    combined = reactive.Value(None)
    test_responses_vectors = reactive.Value(None)
    test_questions_vectors = reactive.Value(None)
    train_responses_vectors = reactive.Value(None)
    train_questions_vectors = reactive.Value(None)
    historical_files = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.process)
    def process_files():
        # Check that required files are uploaded
        if input.historcy_archive() is None or input.other_file() is None:
            ui.notification_show("Please upload both other file and historical recoded responses archive", type="error", duration=15)
            return None
        
        # Clear any existing files in the archive directory
        if len(os.listdir("archive_dir")) > 0:
            for file in os.listdir("archive_dir"):
                os.remove("archive_dir/" + file)
        
        # Check that provided other file has edit permissions
        if not check_file_correct(input.other_file()[0]["datapath"]):
            ui.notification_show("Please fix your excel other file. Perhaps, you should to resave it as a new file using 'save as'", type="error", duration=15)
            return
        
        # Load ignored questions list, if provided
        ignoring_questions_list.set([])
        if input.ignore_file() is not None:
            ignore_df = pd.read_excel(input.ignore_file()[0]["datapath"], keep_default_na=False)
            # check that file has only one column with ignored questions names
            if ignore_df.shape[1] == 1:
                ignoring_questions_list.set(ignore_df.iloc[:, 0].tolist())
            else:
                ui.notification_show("Ignoring questions list should have only one column without title, it will be skipped", type="error", duration=15)
        
        # Load and validate "other" responses
        others.set(pd.read_excel(input.other_file()[0]["datapath"], keep_default_na=False))
        # If file is empty - return warning and stop work
        if len(others.get()) == 0:
            m = ui.modal(
                f"Other file is empty",
                title="Other file is empty",
                easy_close=True,
                footer=None,
            )

            ui.modal_show(m)
            return None

        # Load and extract the historical responses archive
        archive_path = input.historcy_archive()[0]["datapath"]
        extract_archive(archive_path, "archive_dir")

        # Check if archive extraction succeeded
        if len(os.listdir("archive_dir")) == 0:
            m = ui.modal(
                f"Archive is empty",
                title="Archive is empty",
                easy_close=True,
                footer=None,
            )

            ui.modal_show(m)
            return None
        
        # Combine all extracted historical data into a single DataFrame
        historical_files.set(os.listdir("archive_dir"))
        historical_data_list = []
        for file in historical_files.get():
            if file.endswith('.xlsx'):
                df = pd.read_excel("archive_dir/" + file, keep_default_na=False)
                historical_data_list.append(df)        
        combined.set(pd.concat(historical_data_list, ignore_index=True))
        
        # Preprocess data based on the selected format (longitudinal or not)
        if input.longitudinal():
            others.set(preprocess_data_longit(others.get(), isOther=True))
            combined.set(preprocess_data_longit(combined.get()))
        else:
            others.set(preprocess_data(others.get(), isOther=True))
            combined.set(preprocess_data(combined.get()))

        # Validate column structures for "other" and combined data
        others_columns_check = check_basic_columns(others.get())
        combined_columns_check = check_basic_columns(combined.get())
        if others_columns_check is not None:
            m = ui.modal(
                f"Other file: {others_columns_check}",
                title="Other file error",
                easy_close=True,
                footer=None,
            )

            ui.modal_show(m)
            return None
        
        if combined_columns_check is not None:
            m = ui.modal(
                f"Archive files: {combined_columns_check}",
                title="Archive files error",
                easy_close=True,
                footer=None,
            )

            ui.modal_show(m)
            return None
        
        # create dictionary for responces, for each vector we will store all responces with their embeddings, and with history recorings
        responses_dictionary = {}
        # the same for ignored question
        ignored_questions_dictionary = {}

        ### Generate embeddings for historical responses and questions

        # if you want to use OpenAI embeddings - uncoment it
        # train_responses_vectors.set(get_openai_embedding(combined.get(), "response.en")["embedding"].tolist())
        # train_questions_vectors = get_openai_embedding(combined, "full.label")["embedding"].tolist()

        # if you want to use OpenAI embeddings - coment it
        train_responses_vectors.set(get_embedding(combined.get(), "response.en", model=model)["embedding"].tolist())
        train_questions_vectors.set(get_embedding(combined.get(), "full.label", model=model)["embedding"].tolist())
        
        # Fill responses dictionary
        for index, row in combined.get().iterrows():
            response = row["response.en"]

            # Get embedding for response
            response_vector = tuple(train_responses_vectors.get()[index])
            
            question = responses_dictionary.get(response_vector)

            # Get embedding for question
            question_vector = tuple(train_questions_vectors.get()[index])

            # check that question not in the ignor list
            if question in ignoring_questions_list.get():
                # append response, this responces will be matched only if they are equal
                ignored_questions_dictionary[response] = {
                    "existing": row["existing"],
                    "translation": row["translation"],
                    "invalid": row["invalid"],
                    "question": row["full.label"],
                    "question_vector": question_vector,
                }
                continue
            
            # check that question already exists in responces_dictionary
            if question is None:
                # append response, this responces will be matched using cosine similarity on their vectors
                responses_dictionary[response_vector] = [{
                    "existing": row["existing"],
                    "translation": row["translation"],
                    "invalid": row["invalid"],
                    "question": row["full.label"],
                    "question_vector": question_vector,
                    "response": response
                }]
            else:
                question.append({
                    "existing": row["existing"],
                    "translation": row["translation"],
                    "invalid": row["invalid"],
                    "question": row["full.label"],
                    "question_vector": question_vector,
                    "response": response
                    })
                responses_dictionary[response_vector] = question

        responses_dictionary_global.set(responses_dictionary)
        ignored_questions_dictionary_global.set(ignored_questions_dictionary)

        keys = list(responses_dictionary.keys())


        # make embeddings for responses and questions to recode
        # test_responses_vectors.set(get_openai_embedding(others.get(), "response.en")["embedding"].tolist())
        test_responses_vectors.set(get_embedding(others.get(), "response.en", model=model)["embedding"].tolist())
        # test_questions_vectors = get_openai_embedding(others, "full.label")["embedding"].tolist()
        test_questions_vectors.set(get_embedding(others.get(), "full.label", model=model)["embedding"].tolist())

        m = ui.modal(
            f"Data processed successfully, now you can recode responses",
            title="Data processed",
            easy_close=True,
            footer=None,
        )

        ui.modal_show(m)

    @render.download()
    def recode_button():
        # validate that data has been procesed, and embeddings dictionaries exist
        if responses_dictionary_global.get() is None or ignored_questions_dictionary_global.get() is None or ignoring_questions_list.get() is None or others.get() is None:
            ui.notification_show("Please process data first", type="error", duration=15)
            return None
        
        # remove previous outputs from the "output_dir" folder
        if len(os.listdir("output_dir")) > 0:
            for file in os.listdir("output_dir"):
                os.remove("output_dir/" + file)

        # construct similarity matrix for target and historical responses embeddings
        similarities_matrix = cosine_similarity(test_responses_vectors.get(), list(responses_dictionary_global.get().keys()))
        # counter for matched responses
        count = 0
        # counter for total number responses
        total = 0

        # copy input data frame for recoding
        others_df = others.get().copy()
        others_df["invalid"] = None
        others_df["existing"] = None
        others_df["translation"] = None
        for index, row in others_df.iterrows():
            response = row["response.en"]
            question = row["full.label"]
            total += 1

            # Handle ignored questions
            if question in ignoring_questions_list.get():
                # match 1 to 1
                matched = ignored_questions_dictionary_global.get().get(response)
                if matched is not None:
                    if matched["question"] == question:
                        count += 1
                        others_df.loc[index, "invalid"] = matched["invalid"]
                        others_df.loc[index, "translation"] = matched["translation"]
                        others_df.loc[index, "existing"] = matched["existing"]
                continue
            
            # extract similarities for target response
            similarities = similarities_matrix[index]
            # get most similar historical response
            max_similarity_response, max_similarity_index_response = get_max_similarity_index(similarities)

            # check that their similarity is above responses similarity treshold
            if max_similarity_response < float(input.response_similarity_treshold()):
                continue
            
            # get responces by matched embeddings
            historical_responses = responses_dictionary_global.get().get(list(responses_dictionary_global.get().keys())[max_similarity_index_response])
            # from received responses defined one with more imilar question
            historical_questions_vectors = [historical_response["question_vector"] for historical_response in historical_responses]
            question_vector = test_questions_vectors.get()[index]
            max_similarity_question, max_similarity_index_question = get_max_similarity(question_vector, historical_questions_vectors)

            # check that question similarity is above question similarity treshold, if yes - get historical recoding
            if max_similarity_question > float(input.question_similarity_treshold()):
                historical_response = historical_responses[max_similarity_index_question]
                count += 1
                if historical_response["existing"] is not np.nan and historical_response["existing"] != "":
                    if not historical_response["existing"] in row["choices.label"]:
                        others_df.at[index, "translation"] = historical_response["existing"]
                    else:
                        others_df.at[index, "existing"] = historical_response["existing"]
                else:
                    others_df.at[index, "invalid"] = historical_response["invalid"]
                    others_df.at[index, "translation"] = historical_response["translation"]
        
        # if input was in longit format - convert it back
        if input.longitudinal():
            others_df.rename(columns={"invalid": "INVALID other (insert yes or leave blank)"}, inplace=True)
            others_df.rename(columns={"existing": "EXISTING other (copy the exact wording from the options in column choices.label)"}, inplace=True)
            others_df.rename(columns={"translation": "TRUE other (provide a better translation if necessary)"}, inplace=True)
            others_df.rename(columns={"response.en": "response.en.from.uk"}, inplace=True)
        else:
            others_df.rename(columns={"invalid": "INVALID other (insert yes or leave blank)"}, inplace=True)
            others_df.rename(columns={"existing": "EXISTING other (copy the exact wording from the options in column choices.label)"}, inplace=True)
            others_df.rename(columns={"translation": "TRUE other (provide a better translation if necessary)"}, inplace=True)
        
        # load data back in the copy of input file
        get_same_format(input.other_file()[0]["datapath"], others_df)
        print(f"Count: {count}, Total: {total}")
        
        #said that everything is good, show number of recoded responses
        m = ui.modal(
            f"Recoded {count} out of {total} responses",
            title="Recoding results",
            easy_close=True,
            footer=None,
        )

        ui.modal_show(m)

        return "output_dir/output.xlsx"
