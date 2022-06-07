from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

out = generator("Hello, I'm a language model, and I think AI safety is ",
                max_length=300, num_return_sequences=1)

print(out)
