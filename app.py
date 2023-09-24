import gradio as gr
from chromadb import Client, Settings
from clip_embeddings import ClipEmbeddingsfunction

client = Client(Settings(is_persistent=True, persist_directory="./clip_chroma"))

ef = ClipEmbeddingsfunction()
def retrieve_image_from_query(query: str):
    coll = client.get_collection(name = "clip", embedding_function = ef)
    emb = ef.get_text_embeddings(text = query)
    emb = [float(i) for i in emb]
    result = coll.query(
        query_embeddings = emb,
        include = ["documents", "metadatas"],
        n_results=4
            )
    docs = result['documents'][0]  
    descs = result["metadatas"][0]
    list_of_docs = []
    for doc, desc in zip(docs, descs):
        list_of_docs.append((doc, list(desc.values())[0]))
    return list_of_docs

def retrieve_image_from_image(image):
    coll = client.get_collection(name = "clip", embedding_function = ef)
    image = image.name
    result = coll.query(
        query_texts = image,
        include = ["documents", "metadatas"],
        n_results = 4
        )
    docs = result['documents'][0]  
    descs = result["metadatas"][0]
    list_of_docs = []
    for doc, desc in zip(docs, descs):
        list_of_docs.append((doc, list(desc.values())[0]))
    return list_of_docs

def show_img(image):
    return image.name
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(placeholder = "Enter query")
            gr.HTML("OR")
            photo = gr.Image()
            button = gr.UploadButton(label = "Upload file", file_types=["image"])
        with gr.Column():
            gallery = gr.Gallery().style(
                                     object_fit='contain', 
                                     height='auto', 
                                     preview=True
                                  )

    query.submit(fn = retrieve_image_from_query, inputs=[query], outputs=[gallery])
    button.upload(fn = show_img, inputs=[button], outputs = [photo]).then(fn = retrieve_image_from_image, inputs=[button], outputs=[gallery])

if __name__ == "__main__":
    demo.launch()