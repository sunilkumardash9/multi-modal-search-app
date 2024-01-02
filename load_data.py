import os
from chromadb import Client, Settings
from clip_embeddings import ClipEmbeddingsfunction
from typing import List

ef = ClipEmbeddingsfunction()
client = Client(settings = Settings(is_persistent=True, persist_directory="./clip_chroma"))
coll = client.get_or_create_collection(name = "clip", embedding_function = ef)

def get_docs(dir_path: str)-> List[str]:
    docs = []

    for file in os.listdir(dir_path):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            docs.append(dir_path + "/" +file)

    return docs


def add_embeddings_to_chroma():
    img_list = get_docs("/home/sunil/Documents/gradio-gpt-bot/menu-images")
    menu_dict = {}

    menus = [
    {
        "text": "wild mushroom creame chicken - A variety of hand-picked mushrooms, cooked to perfection, mixed with velvety cream and served with freshly chopped scallions"
    },
    {
        "text": "creamy chocolate cheese cake- nested in a dark, moist brownie, sprinkled with choco chips"
    },
    {
        "text": "French onion soup - slow simmered sweet onions, topped with savory cheese and garnished with croutons"
    },
    {
        "text": "zesty salmon fillet - oven baked salmon fillet served with garlic-infused citrusy product kosher product"
    },
    {
        "text": "Nonna edetta's pizza - Fresh mozzarella, special homemade tomato sauce, veg mayonnaise and greens"
    },
    {
        "text": "tiramisu - layered Italian dessert made with refined marsala wine, rum and cocoa powder"
    },
    {
        "text": "Bruschetta - a classic Italian pasta dish with a creamy sauce and a crispy crust"
    },
    {
        "text": "burrata salad - a savory salad made with fresh basil, tomatoes, basil, oregano and more"
    },
    {
        "text": "carbonara pizza - a pizza filled with pasta, bread, Parmesan cheese, oregano and more"
    },
    {
        "text": "chicken parmesan - a savory pasta dish with chicken, Parmesan cheese, oregano and more"
    },
    {
        "text": "sheet pan panzanella - a savory pasta dish with a crispy crust and a creamy sauce"
    }
]

    menu_list = []
    for dish_info in menus:
        menu_dict = {}
        text = dish_info["text"]
        # Extract the name of the dish from the text (assuming it's the first word)
        dish_name = text.split(sep = "-")[0].strip(" ")
        menu_dict[dish_name] = text
        menu_list.append(menu_dict)

    coll.add(ids=[str(i) for i in range(len(img_list))],
         documents = img_list,
         metadatas = menu_list,
         )

add_embeddings_to_chroma()
