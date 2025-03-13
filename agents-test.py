from dotenv import load_dotenv
import os
import logging

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configurar el logging
logging.basicConfig(level=logging.ERROR, filename="errors.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Verificar que la variable OPENAI_API_KEY esté cargada
assert os.getenv("OPENAI_API_KEY"), "La variable de entorno OPENAI_API_KEY no está configurada"

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from pydantic import BaseModel
import asyncio

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

async def homework_guardrail(ctx, agent, input_data):
    print(f"Running guardrail for input: {input_data}")
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    print(f"Guardrail result: {final_output}")
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    print("📚 Bienvenido al tutor de IA. Escribe tu pregunta o 'exit' para salir.")
    while True:
        user_input = input("📝 Tu pregunta: ").strip()
        if user_input.lower() == "exit":
            print("👋 ¡Hasta luego!")
            break

        try:
            result = await Runner.run(triage_agent, user_input)
            print(f"🤖 Respuesta: {result.final_output}")
        except asyncio.TimeoutError:
            logging.error("⏳ Error: La solicitud tomó demasiado tiempo.")
            print("⏳ Error: La solicitud tomó demasiado tiempo.")
        except ConnectionError:
            logging.error("🚫 Error: No se pudo conectar a la API.")
            print("🚫 Error: No se pudo conectar a la API.")
        except ValueError as ve:
            logging.error(f"❗ Error de valor: {ve}")
            print(f"❗ Error de valor: {ve}")
        except Exception as e:
            logging.error(f"⚠️ Error inesperado en main(): {e}")
            print("⚠️ Ocurrió un error. Revisa el log 'errors.log' para más detalles.")

if __name__ == "__main__":
    asyncio.run(main())
