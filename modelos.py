# Description: Listar los modelos de OpenAI
import openai

client = openai.OpenAI(api_key=OPENAI_API_KEY)

try:
    models = client.models.list()
    for model in models:
        print(model.id)
except Exception as e:
    print("Error:", e)