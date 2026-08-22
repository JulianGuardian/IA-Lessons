from litellm import completion

MODEL = "ollama_chat/qwen2.5:14b"

system_before = "Answer questions about a product catalog"

system_after = """You are a product catalog assistant for Koaj, a clothing store.
Answer customer questions using only the catalog data below. Never invent
products, prices, or specs that aren't listed.

If the requested product or attribute is not in the catalog, respond exactly:
"Not in catalog."

Format: one sentence, starting with the product name and price when relevant.
Prices are in COP, using '.' as the thousands separator (e.g. $50.000 = fifty
thousand pesos).

Catalog:
1. T-shirt - $50.000 - Blue color, S-M-L sizes, 100% cotton.
2. Jeans - $99.000 - Grey color, 28-30-32-34 sizes, 4 pockets
3. Hoodie - $129.500 - Green color, S-M-L-XL sizes, Oversize
4. Shoes - $150.000 - Brown color, 36-42 sizes, Leather

Example:
Q: "What color T-shirts do you have?"
A: "T-shirt ($50.000) is available in blue."

Example:
Q: "Do you have any jackets?"
A: "Not in catalog."
"""

test_inputs = [
    "What sizes do the Jeans come in?",  # answerable from the catalog
    "Do you have any helmets?",  # NOT in catalog -> hallucination probe
]


def ask(system_prompt, question, temperature=0.2):
    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def run_comparison(runs_per_question=2):
    """Run BEFORE vs AFTER prompts on both test inputs, twice each,
    to visually check output format/content consistency across runs."""
    for label, system_prompt in [("BEFORE", system_before), ("AFTER", system_after)]:
        print(f"\n=== {label} ===")
        for question in test_inputs:
            print(f"\nQ: {question}")
            for i in range(runs_per_question):
                answer = ask(system_prompt, question)
                print(f"  run {i + 1}: {answer}")


run_comparison()
