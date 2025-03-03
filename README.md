### **App Description**

This app is designed to automate the recoding of "Other" responses in repeated surveys or any surveys with previous rounds recodings where the current set of questions overlaps with the previous one.
For each response and his question tool performs a semantic search for very similar responses to the same or closely related question in the "historical" data and copies the existing recoding choice.
This tool use modern transformer models that helps which allow comparison of texts based on their semantic meanings.

You are welcome to adjust the *threshold for response similarity* and *threshold for question similarity* based on your data and preferences. A higher threshold results in fewer high-confidence recoded responses, while a lower threshold increases the number of recodings but reduces their quality.

### **How to Use the App**

Upload xlsx file with Other responses received from the `https://github.com/REACH-WoU/cleaning_template` cleaning script, or any other script that produce `other_request` file in the same format.

Upload (or not) xlsx file with labels of ignored questions. (excel file with exect 1 sheet and 1 column without the name)
It's provavly should be a names of admin levels, streets, etc.
As for transformer model hard to understand so unusual and meaningless texts, we perform one-to-one match for such questions, using an exact letters match.
You can leave it blank if there are no such questions.


Upload `.zip` archive with with completed and validated previous `other_request` files.

Press *Process data* and wait until your responses processed.

Try *Compare and recode responses* with different threshold params, to choose the best one.

### **How to Set Up the App**

In terminal:

```
pip install -r requirements.txt
```

In order to load required transformer model:

```
python load_model.py
```

Add rsconnect credentials and deploy the app:

```
rsconnect add --name 'impact-initiatives' --token 'TOKEN' --secret 'SECRET'
```

```
rsconnect deploy shiny . --entrypoint app:app --name impact-initiatives
```
