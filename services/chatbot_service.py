import json

from openai import OpenAI


class Chatbot:

    @classmethod
    def search(cls, openai_client: OpenAI, data, message):
        """
        Uses openAI API
        provide data from the retrieval and the user message
        """

        system_prompt = cls.build_system_prompt(data)
        system_message = {"role": "system", "content": system_prompt}

        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message, message],
            response_format={"type": "json_object"}
        )

        answer = json.loads(completion.choices[0].message.content)['answer']
        return answer

    @staticmethod
    def build_system_prompt(data):
        """Simply building the system prompt as it was provided"""

        return f"""You are a Q&A chatbot that answers questions about a start-ups. 

Your whole knowledge is the following information:

###
   {data}
###
 
Provide an answer in the following JSON format:
{{
    "answer": string (your answer to the user)
}}
"""
