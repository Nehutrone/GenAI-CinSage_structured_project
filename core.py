from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):
    title:str
    release_year:int
    genre:List[str]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary:Optional[str]
parser=PydanticOutputParser(pydantic_object=Movie)
model = ChatGroq(model="openai/gpt-oss-safeguard-20b")
prompt= ChatPromptTemplate.from_messages([
    ('system',"""
Extract movie information from the paragraph
     {format_instructions}
"""),
("human","{paragraph}")
]
)


para=input("'give your paragraph:-")
final_prompt=prompt.invoke(
    {"paragraph":para,
     "format_instructions":parser.get_format_instructions()
     }
)
response=model.invoke(final_prompt)
print(response.content)
