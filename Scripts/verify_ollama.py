"""
Purpose:
Verify that the Ollama server and local LLM are working correctly.

Inputs:
A model name (currently Mistral) and a simple test prompt.

Outputs:
Prints the model's response to the terminal.

When do I run this?
After installing Ollama, changing models, or troubleshooting the local LLM.
"""

from ollama import chat


def verify_ollama():
    print("Checking Ollama connection...\n")

    response = chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Ollama is working!"
            }
        ]
    )

    print("Model Response:")
    print("-" * 40)
    print(response["message"]["content"])
    print("-" * 40)
    print("\nSuccess! Ollama is connected.")


if __name__ == "__main__":
    verify_ollama()