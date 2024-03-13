# RAG bot: Porozprávaj sa so svojimi dátami

http://tinyurl.com/noc-24-rag

Vývojové prostredie by ste mali mať už všetci pripravené, potrebné knižnice nainštalované, preto sa môžme pustiť rovno do implementácie. Pre účely tohto workshopu nám bude úplne postačovať jeden zdrojový súbor. V hlavnom priečinku si vytvoríme `ragbot.py` súbor, kde si celého RAGbota naprogramujeme.

Keďže sme si povedali, že trénovanie vlastného jazykového modelu nie je pre nás cesta, náš RAGbot bude používať ChatGPT od OpenAI. Aby sme mohli používať jeho API, musíme si najprv vygenerovať a niekde uložiť API kľúč.

Od tohto momentu, ak nebude povedané inak, všetok kód budeme písať do nášho novo-vytvoreného súboru `ragbot.py`.
Začnime s definovaním API kľuča:

```python
API_KEY = "odniekial-ho-skopirujeme"
```

a teraz si poďme overiť, či nám API kľuč funguje a či ChatGPT dokáže na naše otázky odpovedať:

```python
from llama_index.llms.openai import OpenAI


API_KEY = "odniekial-ho-skopirujeme"


open_ai_llm = OpenAI(api_key=API_KEY)

answer = open_ai_llm.complete("NEJAKA VHODNA OTAZKA")   # TODO!!!
print(answer.text)
```

vykonajme skript z terminálu: `python ragbot.py`. Zistili sme, že hoci nám komunikácia funguje, ChatGPT nám nepomôže nájsť odpoveď na to, čo nás aktuálne trápi. Je to spôsobené tým, že ChatGPT nemá o týchto veciach informácie. A to z dôvodu, že ChatGPT (3.5) pozná svet taký, aký bol do januára 2022 a zároveň pozná len to, čo bolo publikované niekde na internet. Odpoveď na našu otázku preto poznať nemôže.

Fajn, tak ako dosiahnúť to, aby ChatGPT vedel odpovedať aj na takéto otázky? Poskytneme mu kontext! S tým nám pomôže práve LlamaIndex framework. V našom prípade sa odpovede na naše otázky budú nachádzať v dokumentoch uložených v priečinku `my_docs`.
Poďme si ich teda načítať.

```python
from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(input_dir="my_docs").load_data()

print("number of docs loaded: ", len(docs))
```

Dokumenty máme načítané, ale stále nám to nijak nevyriešilo náš problém. V úvode sme si povedali niečo o spracovaní dokumentov a vytváraní embeddingov. Našťastie, LlamaIndex vie toto urobiť takmer úplne sám a k tomu ich aj naindexovať. Potrebujeme mu dať do rúk len tie správne nástroje - v našom prípade OpenAI embedovací model:

```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

open_ai_embedding = OpenAIEmbedding(api_key=API_KEY)
index = VectorStoreIndex.from_documents(documents=docs, embed_model=open_ai_embedding)
```

Dokumenty máme načítané a spracované, index vytvorený. A teraz sa úkaže sila tohto úžasného frameworku, poďme sa pýtať našich dát:

```python
query_engine = index.as_query_engine(llm=open_ai_llm)
answer = query_engine.query("NEJAKA VHODNA OTAZKA")   # TODO!!!
print(answer.response)
```

super! Dostali sme správnu odpoveď. Stačilo na to len niečo viac ako 10 riadkov kódu! Pred tým, ako sa zahĺbime do konverzácie, ešte náš kód trochu upracme.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

API_KEY = "API-KEY"

class RagBot:
    def __init__(self) -> None:
        self.llm = OpenAI(api_key=API_KEY)
        self.embed_model = OpenAIEmbedding(api_key=API_KEY)

    def ingest(self, input_dir: str) -> None:
        docs = SimpleDirectoryReader(input_dir=input_dir).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents=docs, embed_model=self.embed_model
        )
        self.query_engine = self.index.as_query_engine(llm=self.llm)

    def query(self, question: str) -> str:
        response = self.query_engine.query(question)
        return response.response
```

Cool! Máme funkčného RAGbota. Ale nie sme v 90. rokoch minulého storočia a chceli by sme k tomu aj nejaké pekné UI. Na jeho implementáciu, žiaľ, dnes už nemáme čas. Avšak, nenecháme vás odísť sklamaných. Máme tu pripravené jednoduché chatovacie web UI vytvorené v ďalšom skvelom frameworku - `streamlit`.

Z pripraveného repozitára si stiahneme súbor `webui.py`. Nebudeme ho nejak analyzovať, len na správne miesta doplníme správny kód.
Najprv si overme, či nám streamlit a naše web UI funguje. Spustime si túto mini web-appku:

```sh
streamlit run webui.py
```

Dobre, toto funguje, ale zatiaľ to nič nerobí. V kóde sú "placeholdre", kde si spoločne doplníme náš kód. Pred tým, ako budeme klásť otázky, musíme nášho RAGbota inicializovať. Do `init()` funkcie nášho web UI:

```python
@st.cache_resource(show_spinner=False)
def init():
    with st.spinner(text="Načítavam dokumenty, počkajte prosím."):
        pass

        # -------------------------
        # Init RAGbot here
        #
        #
        # -------------------------

bot = init()
```

pridáme našu inicializáciu RAGbota:

```python
@st.cache_resource(show_spinner=False)
def init():
    with st.spinner(text="Načítavam dokumenty, počkajte prosím."):
        from ragbot import RagBot

        bot = RagBot()
        bot.ingest("my_docs")
        return bot

bot = init()
```

a na miesto, kde sa získava otázka a poskytuje odpoveď:

```python
    ...
    with st.spinner("Pracujem na odpovedi..."):
        placeholder = st.empty()

        # -------------------------------------
        # Ask question and retrieve asnwer here
        #
        #
        # -------------------------------------
        response = "Zatiaľ nič nie je implementované :/"

        placeholder.markdown(response)
        add_msg_to_ui_conversation(role="assistant", content=response)
```

zavoláme našu `query()` metódu:

```python
    ...
    with st.spinner("Pracujem na odpovedi..."):
        placeholder = st.empty()

        response = bot.query(question)

        placeholder.markdown(response)
        add_msg_to_ui_conversation(role="assistant", content=response)
```

Hotovo, poďme to vyskúšať. Zabijeme aktuálnu inštanciu (Ctrl+C) a spustíme znova.

```sh
streamlit run webui.py
```

Tak a teraz môžme chatovať s našimi dátami. Oh, no. Je tu problém. Ono si to nepamätá, o čom sa bavíme. Riešenie je jednoduchšie, ako sa môže zdať. Problém je to, že práve používame dopytovací engine bez pamäte, nie ten chatovací. Poďme to teda zmeniť - po naindexovaní dát okrem "query enginu" vytvoríme aj "chat engine" a vytvoríme novú metódu `chat()`:

```python
class RagBot:
    def __init__(self) -> None:
        self.llm = OpenAI(api_key=API_KEY)
        self.embed_model = OpenAIEmbedding(api_key=API_KEY)

    def ingest(self, input_dir: str) -> None:
        docs = SimpleDirectoryReader(input_dir=input_dir).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents=docs, embed_model=self.embed_model
        )
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        # we will use chat engine for chat
        self.chat_engine = self.index.as_chat_engine(chat_mode="condense_question", llm=self.llm)          # <---------

    def query(self, question: str) -> str:
        response = self.query_engine.query(question)
        return response.response

    # --- new chatting method ---
    def chat(self, question: str, history: list) -> str:
        response = self.chat_engine.chat(question, chat_history=history)
        return response.response
    # ---------
```

Keď sa pozrieme na LlamaIndex `chat()` metódu nášho chat enginu, môžme vidieť, že očakáva list predchádzajúcich správ. Čo to ale znamená? Znamená to to, že chatovací engine je sám o sebe bezstavový a nedrží si žiaden kontext. Má to isté výhody, ktoré momentálne nebudeme rozoberať. Jediné, čo potrebujeme spraviť je to, že LlamaIndexu tento list minulých správ poskytneme spolu s každou našou otázkou.


Ďalšim dôvodom na radosť je to, že LlamaIndex si obsah tohto listu manažuje sám, my sme zodpovední len za jeho "state management" - teda inicializáciu a držanie niekde v pamäti. V našom prípade to znamená inicializovať a držať ho niekde vo web UI:

```python
if "messages" not in st.session_state.keys():
    initial_msg = ChatMessage(
        role="assistant",
        content="Ahoj, spýtaj sa niečo svojich dát!",
    )
    st.session_state.messages = [initial_msg]
    st.session_state.history = []                   # <------------
```

a zakaždým ho poskytnúť nášmu RAGbotovi:

```python
    ...
    with st.spinner("Pracujem na odpovedi..."):
        placeholder = st.empty()

        # response = bot.query(question)
        response = bot.chat(question, st.session_state.history)    # <------------

        placeholder.markdown(response)
        add_msg_to_ui_conversation(role="assistant", content=response)
```

reštartujme web-appku a overme, či to funguje. Áno! Tak a teraz si už všetci vieme vytvoriť svojho RAGbota, ktorý bude vedieť odpovedať aj na to, na čo žiaden iný chatbot nie!
