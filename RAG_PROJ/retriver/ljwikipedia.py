from langchain_community.retrievers import WikipediaRetriever

retriver = WikipediaRetriever(load_max_docs = 2, load_all_available_meta = True,lang = 'en')

docs=retriver.invoke("abdul kalam")

for i,doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content[:500])  # print first 500 characters