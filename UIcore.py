import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# ---- Pydantic Model ----
class Movie(BaseModel):
    title: str
    release_year: int
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: Optional[str]

parser = PydanticOutputParser(pydantic_object=Movie)

# ---- Model ----
model = ChatGroq(model="openai/gpt-oss-safeguard-20b")

# ---- Prompt ----
prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph
{format_instructions}
"""),
    ("human", "{paragraph}")
])

# ---- Streamlit UI ----
st.title("🎬 Movie Information Extractor")

para = st.text_area("Enter your paragraph:")

if st.button("Extract"):
    if para.strip() == "":
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Processing..."):
            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions": parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

        st.subheader("Extracted Output:")
        st.write(response.content)