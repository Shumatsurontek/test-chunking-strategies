TEXT = """
🧠 GUIDE COMPLET : Tool Calling + Streaming avec LangChain et Bedrock

## 🔍 Contexte

Dans LangChain, l'utilisation combinée de **streaming token par token** et de **tool calling structuré** fonctionne parfaitement avec OpenAI (`ChatOpenAI`), mais présente de nombreuses limitations avec les modèles hébergés sur Amazon Bedrock (`ChatBedrock`, `ChatBedrockConverse`), comme Claude, Mistral ou Llama.

Tu utilises `ChatBedrock` avec `streaming=True` et constates que :
- Le tool calling ne fonctionne pas en streaming.
- `with_structured_output()` n’est pas pris en charge.
- Tu dois parser manuellement du JSON dans un flux de texte brut.

---

## 🚧 Limitations de ChatBedrock

1. ❌ **Pas de tool calling structuré en streaming**
   - Le modèle stream du texte brut, sans métadonnées structurées comme `.tool_call_chunks`.

2. ❌ **`with_structured_output()` incompatible avec streaming**
   - Nécessite `streaming=False`, ce qui désactive le stream token par token.

3. ❌ **Interception manuelle nécessaire**
   - Tu dois bufferiser le texte et détecter manuellement les tool calls (en JSON).

---

## ✅ Solutions disponibles

### ✅ 1. Streaming manuel + JSON parsing
Tu peux :
- Streamer token par token.
- Assembler les tokens dans un buffer.
- Parser en JSON dès que possible (`json.loads()`).
- Exécuter le tool Python associé.
- Optionnellement interrompre le flux.

### ✅ 2. Non-streaming + `with_structured_output()`
Tu désactives le stream (`streaming=False`) et obtiens un `TypedDict` ou Pydantic en retour :

```python
from langchain_aws import ChatBedrock
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class ToolCall(BaseModel):
    name: str
    arguments: dict

llm = ChatBedrock(model_id="...", streaming=False)
structured_llm = llm.with_structured_output(ToolCall)
result = structured_llm.invoke("Appelle la météo pour Paris")
"""

