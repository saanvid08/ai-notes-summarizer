import streamlit as st
import openai

st.set_page_config(page_title="AI Notes Summarizer", layout="centered")

st.title("🧠 AI Notes Summarizer")
st.write("Paste your notes below and get a clean summary!")

user_input = st.text_area("Your Notes", height=300)
openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")

if st.button("Summarize"):
    if user_input and openai.api_key:
        with st.spinner("Summarizing..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the following notes in clear bullet points."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.5,
                max_tokens=300,
            )
            summary = response["choices"][0]["message"]["content"]
            st.markdown("### 📝 Summary")
            st.write(summary)
    else:
        st.error("Please enter both notes and your OpenAI API key.")
