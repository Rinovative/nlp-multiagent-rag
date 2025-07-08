class GeneratorAgent:
    def __init__(self, client):
        """
        Initialisiert den GeneratorAgent mit einem OpenAI-Client.

        :param client: Der OpenAI-Client, der für die Generierung von Antworten verwendet wird
        """
        self.client = client

    def generate_answer(self, user_query, context, memory):
        """
        Nutzt den Kontext, die Benutzeranfrage und das Gedächtnis, um eine Antwort zu generieren.

        :param context: Der Kontext, der aus den abgerufenen Dokumenten stammt
        :param user_query: Die Benutzeranfrage, auf die eine Antwort generiert werden soll
        :param memory: Der gespeicherte Verlauf oder Kontext des Benutzers (kann leer sein)
        :return: Die generierte Antwort des LLM
        """
        # Erstelle den Prompt mit dem Gedächtnis separat hinzugefügt
        prompt = f"Frage:\n{user_query}"
        if context and context.strip():
            prompt += f"\nBeantworte diese Frage basierend auf den relevanten Dokumenten:\n{context}"
        if memory and len(memory) > 0:
            prompt += f"\nFrage: Erinnere dich dabei an:\n{memory}"
        print(prompt)

        # Berechne die Länge des Prompts (einschließlich Benutzeranfrage, Kontext und Gedächtnis)
        prompt_length = len(prompt.split())

        # Setze max_tokens basierend auf der Länge des Prompts
        max_tokens = max(
            200, 4096 - prompt_length
        )  # Verhindere, dass max_tokens zu klein wird

        # Generiere die Antwort
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Du kannst "gpt-4" oder ein anderes Modell verwenden
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},  # Dynamisch generierter Prompt
            ],
            max_tokens=max_tokens,  # Dynamisch berechneter Wert
            temperature=0.7,
        )

        return response.choices[0].message.content
