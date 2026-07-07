from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return llm
